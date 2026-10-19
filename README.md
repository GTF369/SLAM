# Face Pattern AR (Android)

An Android prototype for placing a transparent pattern over a live face using ARCore Augmented Faces and OpenGL ES. Users can select a PNG/JPG, drag it, pinch to resize, rotate with two fingers, and reset the placement.

## Build
The GitHub Actions workflow builds a debug APK on each push and on manual dispatch. Open **Actions → Build Android APK → Run workflow** (or push a commit), then download the `face-pattern-debug-apk` artifact from a successful run.

For a local build, use JDK 17 and Gradle 8.9 with Android SDK platform 35 installed, then run `gradle assembleDebug`.

## Device requirements
- Android 7.0 (API 24) or later.
- A device supported by ARCore with a working front-facing camera. ARCore availability varies by model and region.
- Camera permission.

## Prototype notes
This is an initial implementation. It renders a pattern in the local XY coordinates of the tracked face mesh, so the texture follows the mesh surface. The drag mapping is an approximate screen-to-face mapping, not calibrated ray/triangle picking. Camera texture orientation and placement should be validated on real devices. The current prototype does not save/export a composited photo or video and does not include production-level error reporting. It is not yet a tested release build.
