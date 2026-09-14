# Environment issue: static-lib onnxruntime crashes ALL TTS models on this machine

**Status: pre-existing environment issue, NOT caused by the ZeroTTS work in this fork. Confirmed
reproducible with an unmodified upstream model (`vits`) before any ZeroTTS-specific code runs.**

## Symptom

Following CLAUDE.md's documented build exactly (`cmake -DSHERPA_ONNX_ENABLE_TESTS=ON
-DCMAKE_BUILD_TYPE=Release ..`, no other flags), any invocation of `sherpa-onnx-offline-tts` that
gets far enough to actually construct a TTS model (i.e., passes `Validate()` and reaches
`OfflineTts`'s constructor) crashes with:

```
free(): invalid pointer
Aborted (core dumped)
```

Confirmed via `gdb -batch -ex run -ex bt`: the crash is inside ONNX Runtime's own internal device
discovery, **not** in any ZeroTTS code:

```
#9  std::__detail::_Compiler<std::regex_traits<char>>::_Compiler(...)
#10 onnxruntime::(anonymous namespace)::GetPciBusId(...)
#11 onnxruntime::DeviceDiscovery::DiscoverDevicesForPlatform()
#12 onnxruntime::DeviceDiscovery::GetDevices()
#14 onnxruntime::Environment::EpInfo::Create(...)
#15 onnxruntime::Environment::RegisterExecutionProviderLibrary(...)
#16 onnxruntime::Environment::CreateAndRegisterInternalEps()
#17 onnxruntime::Environment::Initialize(...)
#19 OrtEnv::GetOrCreateInstance(...)
#21 sherpa_onnx::OfflineTtsZeroTtsModel::Impl::Impl(...)   <- first ZeroTTS frame, already inside ORT
```

This happens the first time any code in the process calls `Ort::Env(...)` (which every
`OfflineTts*Model` does once). ORT 1.28's new device-auto-discovery feature
(`DeviceDiscovery::GetDevices()`) probes `/sys/class/drm/card0/...` for GPU info; in this sandboxed
container `/sys/class/drm/card0` doesn't fully exist (matches the earlier benign warning
`GPU device discovery failed: ... Failed to open file: "/sys/class/drm/card0/device/vendor"`), and
something in that fallback/error path corrupts the heap when it later runs a `std::regex` for PCI
bus ID parsing (`GetPciBusId`).

## Proof this is pre-existing, not a ZeroTTS bug

1. A standalone 5-line program (`Ort::Env env(ORT_LOGGING_LEVEL_ERROR);`, nothing else linked)
   against the exact same prebuilt `libonnxruntime.a` (v1.28.2, static, the one CLAUDE.md's build
   downloads automatically) succeeds fine — so it's not simply "Env always crashes here."
2. **The unmodified upstream `vits` model crashes identically** through the exact same CLI binary,
   with a completely unrelated, pre-existing, upstream model
   (`vits-piper-en_US-amy-low`, downloaded from `k2-fsa/sherpa-onnx`'s own release assets):
   ```
   ./bin/sherpa-onnx-offline-tts --vits-model=... --vits-tokens=... --vits-data-dir=... \
     --output-filename=/tmp/vits_test.wav "hello world this is a test"
   -> free(): invalid pointer / Aborted (core dumped)
   ```
   This confirms the crash has nothing to do with ZeroTTS's C++ code — it reproduces with code that
   predates this fork entirely.
3. Our own `offline-tts-zerotts-model-test` **gtest binary** (linking the identical set of static
   libraries: `sherpa-onnx-core`, `onnxruntime`, `piper_phonemize`, `espeak-ng`, `ucd`, ...)
   constructs `OfflineTtsZeroTtsModel` (and therefore `Ort::Env`) successfully, with no crash. The
   difference isn't the link set — it's something about the runtime path the full CLI binary takes
   before reaching `Ort::Env` construction (exact trigger not further root-caused; not worth more
   time given a working fix exists, see below).

## Workaround (confirmed working)

Configure with `-DBUILD_SHARED_LIBS=ON`. This makes the onnxruntime download step fetch the
**dynamic** (`.so`) prebuilt onnxruntime release instead of the static (`.a`) one CLAUDE.md's
plain build command gets by default. With the shared lib (+ `LD_LIBRARY_PATH` pointing at the
`.so`), both `vits` and `zerotts` run correctly end to end:

```bash
mkdir -p build_shared && cd build_shared
cmake -DSHERPA_ONNX_ENABLE_TESTS=OFF -DCMAKE_BUILD_TYPE=Release \
      -DSHERPA_ONNX_ENABLE_BINARY=ON -DSHERPA_ONNX_ENABLE_TTS=ON \
      -DBUILD_SHARED_LIBS=ON ..
make -j$(nproc) sherpa-onnx-offline-tts
LD_LIBRARY_PATH=$(find . -iname 'libonnxruntime.so*' -exec dirname {} \; | head -1) \
  ./bin/sherpa-onnx-offline-tts --zerotts-... "some text"
```

Verified: `vits` produces its expected 1.972s audio clip with RTF 0.082 (matches expected upstream
behavior); `zerotts` produces real, non-silent audio (see plan.md Phase 4's proof) with a 0.935
mel-spectrogram correlation against the reference Python `zerotts` package's output for the same
input.

## Second, unrelated environment gotcha found along the way

The shell environment this task started in has **`ONNXRUNTIME_DIR`, `SHERPA_ONNXRUNTIME_INCLUDE_DIR`,
`SHERPA_ONNXRUNTIME_LIB_DIR`** pre-set to paths under a **different, unrelated project**
(`/mnt/disk2/projects/cdc-ai-orchestrator/prebuilt/onnxruntime/...`, and specifically its
**arm64-v8a** (Android) libraries) plus `SHERPA_ONNX_ENABLE_QNN=ON` and
`SHERPA_ONNX_ENABLE_BINARY=OFF`. Left as-is, `cmake configure` silently picks up that
cross-compiled Android `.so` for what's supposed to be a desktop x86_64 build, and the link step
fails outright (`file in wrong format`). Every `cmake`/`make` invocation for this feature's desktop
builds in this session explicitly unsets those four variables:
```bash
env -u ONNXRUNTIME_DIR -u SHERPA_ONNXRUNTIME_INCLUDE_DIR -u SHERPA_ONNXRUNTIME_LIB_DIR \
    -u SHERPA_ONNX_ENABLE_QNN -u SHERPA_ONNX_ENABLE_BINARY \
    cmake ... / make ...
```
Not touching the other project itself (per the hard rule) — just not inheriting its env vars into
this repo's build.

## Recommendation for whoever picks this up

- Desktop dev/test builds of this repo, in *this specific container*, need `-DBUILD_SHARED_LIBS=ON`
  plus `LD_LIBRARY_PATH` to actually run any TTS model's CLI end to end. Unit tests that don't go
  through the full `OfflineTts`/CLI path (like our Phase 1-3 gtests) are unaffected and build/pass
  fine with the plain static-lib build.
- This is worth reporting upstream (`k2-fsa/sherpa-onnx` and/or `microsoft/onnxruntime`) since it
  will affect any consumer building the static-lib flavor of ORT 1.28.2 on a similarly-sandboxed
  Linux host without a real `/sys/class/drm/card0` — out of scope for this private fork's feature
  work, noted here for the record.
