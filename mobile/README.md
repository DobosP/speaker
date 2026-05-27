# mobile/ — on-device Android test harness

A Flutter app that runs the speaker assistant loop on a real phone, fully
on-device:

- **Assistant** — speak or type, a small **Gemma 3 (1B, int4)** generates the
  reply on-device (GPU via MediaPipe / `flutter_gemma`), and the answer is
  spoken back. ASR + LLM + TTS, all local.
- **Listen** — streaming speech recognition via
  [`sherpa-onnx`](https://github.com/k2-fsa/sherpa-onnx).
- **Speak** — local text-to-speech via sherpa-onnx.

Inference is **fully offline**. The only network use is a one-time model
download on first launch (see *The LLM model* below).

### The LLM model (download on first launch)

The Gemma weights (~0.5 GB) are **not** bundled in the APK — that would push it
past a gigabyte. Instead the app downloads the model once on first use, caches
it on device, and runs offline thereafter.

Gemma is **license-gated**, so the app does not carry a HuggingFace token.
Instead, CI republishes the model to a *public* GitHub release that the app
pulls from. One-time setup:

1. On a HuggingFace account, accept the Gemma license at
   `huggingface.co/litert-community/Gemma3-1B-IT` and create a read token.
2. Add it as the repo secret **`HF_TOKEN`**.
3. Run the **Publish Gemma model** workflow (Actions tab → Run workflow). It
   fetches the gated model and publishes it to the `gemma-model` release.

> See [`CREDENTIALS.md`](../CREDENTIALS.md) for the full token reference
> (`HF_TOKEN`, `HUGGINGFACE_TOKEN`, `GIT_HUB_TOKEN`, `LIVEKIT_*`) — where each
> lives and what it unlocks.

The app downloads from
`releases/download/gemma-model/Gemma3-1B-IT-q4.task` (the URL constant in
`lib/llm.dart`). Until that release exists, the app installs and the ASR/TTS
tabs work, but the Assistant tab's download will fail. Once on the device the
model is reused across reinstalls and can be sideloaded over USB — see *Fast
test loop* below.

## Get an APK onto your phone (no local toolchain needed)

Pushing app changes under `mobile/**` to `main` (or running **Build Android
APK** manually from the Actions tab) builds and publishes the APK. Download it
on your phone from
`https://github.com/DobosP/speaker/releases/download/android-latest/speaker-android.apk`
and tap to install (allow *Install unknown apps* if prompted).

The CI job downloads the ASR/TTS models, generates the Android project, and
builds the APK — the repo itself stays free of model binaries and platform
scaffolding. The Gemma LLM is fetched on the phone at first launch, not baked
into the APK (see above).

## Fast test loop (USB + adb)

Reinstalling a build and re-downloading the ~550 MB model every time is slow.
With a USB cable and `adb` it becomes: **push the model once, then reinstall
the APK freely** — the model is never fetched over the network again.

```bash
cd mobile
bash tool/dev-install.sh        # install/replace the app, keeping its data, then launch
bash tool/push-model.sh         # one time: cache the model locally + push it to the phone
```

How it works:

- **The model lives at a stable on-device path** (`lib/model_store.dart`): the
  app's external files dir, `/sdcard/Android/data/<applicationId>/files/models/`.
  The app loads the weights from there (`.fromFile`) and only downloads when the
  file is genuinely missing — so a normal reinstall reuses it.
- **`tool/dev-install.sh`** runs `adb install -r` (replace in place, **keep app
  data**), so the model survives the reinstall. If a build's signature doesn't
  match the installed one it offers a clean reinstall and **re-pushes the model
  for you**. Pass a path to install a specific APK, `--with-model` to also ensure
  the model is present, `-y` to skip the clean-reinstall prompt.
- **`tool/push-model.sh`** downloads the weights into `.dev-cache/` (git-ignored)
  once, then `adb push`es them to that path (with a `run-as` fallback for devices
  that block direct pushes to `/Android/data`). Re-pushing after a clean install
  is a local copy over USB — seconds, no internet. Use `--model <file>` to push
  an already-local `.task`.

Override the package id or repo with `SPEAKER_APP_ID` / `SPEAKER_REPO`; point
`adb` elsewhere with `$ADB`. Inside the app, the Assistant tab shows **Load
model** (not *Download*) once the weights are on disk.

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

The ASR/TTS UI mirrors the official sherpa-onnx Flutter examples; the Assistant
tab wires in `flutter_gemma`. On-device LLM runtime behaviour (GPU init, model
load) can only be verified on a real phone — CI confirms it compiles and
produces an APK, not that Gemma runs on your specific device. If a build fails,
it's most likely a version/toolchain pin in `pubspec.yaml` or
`tool/setup_android.sh` (e.g. `minSdk`/`compileSdk`), not the app logic.
