# mobile/ — on-device Android test harness

A minimal Flutter app that proves the **on-device audio loop** of the speaker
assistant on a real phone: streaming speech recognition (**Listen**) and speech
synthesis (**Speak**), both running fully locally via
[`sherpa-onnx`](https://github.com/k2-fsa/sherpa-onnx) with **no network at
runtime**.

This is Phase 4 groundwork from `docs/target_architecture.md` — deliberately
scoped to ASR + TTS. The LLM and the `always_on_agent` brain are **not** wired
in yet; this exists to validate that on-device speech and the build-and-install
pipeline work end-to-end first.

## Get an APK onto your phone (no local toolchain needed)

1. Push to `claude/nice-planck-ZDr90` (or run the **Build Android APK** workflow
   manually from the Actions tab).
2. Open the finished run, download the **speaker-mobile-debug-apk** artifact.
3. Copy the `.apk` to your Android phone and tap to install (you may need to
   allow *Install unknown apps* for your browser/file manager).

The CI job downloads the models, generates the Android project, and builds the
APK — the repo itself stays free of model binaries and platform scaffolding.

## Build locally instead

Requires the Flutter SDK and an Android toolchain.

```bash
cd mobile
bash tool/setup_android.sh        # generate android/ + patch mic permission
bash tool/download-models.sh      # fetch ASR + TTS models into assets/
python3 tool/generate-asset-list.py
flutter pub get
flutter run                       # or: flutter build apk --debug
```

## Models

Downloaded at build time into `assets/` (git-ignored):

- **ASR:** `sherpa-onnx-streaming-zipformer-en-2023-06-26` (English, streaming)
- **TTS:** `vits-piper-en_US-amy-low` (English)

Swap them by editing `tool/download-models.sh` and the paths in
`lib/asr_model.dart` / `lib/tts_model.dart`.

## Caveat

The Dart UI mirrors the official sherpa-onnx Flutter examples, but this app has
not yet been compiled — the CI run is its first real build. If the first build
fails, it's almost certainly a version/toolchain pin in `pubspec.yaml` or
`tool/setup_android.sh`, not the app logic.
