# Environment note: Android NDK executable linking fails against this sandbox's prebuilt onnxruntime.so

**Symptom:** `build-android-arm64-v8a.sh` with `SHERPA_ONNX_ENABLE_BINARY=ON` builds
`libsherpa-onnx-jni.so` fine, but fails to link any standalone executable
(`sherpa-onnx`, `sherpa-onnx-offline`, `sherpa-onnx-offline-audio-tagging`,
`sherpa-onnx-offline-tts`, ...) with errors like:

```
ld.lld: error: undefined reference due to --no-allow-shlib-undefined: __fwrite_chk@LIBC_N
>>> referenced by /mnt/disk2/projects/cdc-ai-orchestrator/prebuilt/onnxruntime/libs/arm64-v8a/libonnxruntime.so
```

(also seen for `__write_chk`, `__register_atfork`, `__gnu_strerror_r`, `stderr`)

**Root cause:** the prebuilt `libonnxruntime.so` used in this sandbox
(`$SHERPA_ONNXRUNTIME_LIB_DIR`, actually belonging to a different project --
`cdc-ai-orchestrator` -- per `CLAUDE.md`) references fortified libc symbols that this NDK's
executable link step won't resolve under the default `--no-allow-shlib-undefined` linker
behavior. This is an ABI/link-time-strictness mismatch between that prebuilt `.so` and the
NDK toolchain here, **not a ZeroTTS bug** -- it reproduces identically for stock upstream
binaries with zero ZeroTTS code involved. Notably it does *not* affect the JNI `.so` target
(shared libraries aren't linked with the same strict undefined-symbol check here).

**Fix:** relax the check for executables specifically, then rebuild:

```bash
cd build-android-arm64-v8a
cmake -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" .
make -j$(nproc) sherpa-onnx-offline-tts   # or: make -j$(nproc) for everything
```

This is safe here because those symbols are provided by the real device's system `libc.so`
at runtime (confirmed: the resulting `sherpa-onnx-offline-tts` binary ran and produced correct
audio on a real Pixel 6 Pro over `adb shell`) -- the NDK's stub sysroot just doesn't expose them
at static-link time for this particular prebuilt onnxruntime build.
