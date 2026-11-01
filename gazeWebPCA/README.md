# PCA-Enhanced Webcam Gaze Tracker

PCA-Enhanced Webcam Gaze Tracker is a privacy-first, browser-native eye-tracking research prototype built with HTML, CSS, and JavaScript. It combines MediaPipe Face Landmarker, OpenCV.js eye preprocessing, native JavaScript PCA, personalized calibration, head-pose compensation, validation, and an in-page gaze cursor.

> **Research prototype:** This project is not a medical device, accessibility certification tool, or production replacement for dedicated eye-tracking hardware.

## Highlights

- Browser-based webcam capture with MediaPipe facial, eye, and iris landmarks
- OpenCV.js `warpAffine` and grayscale preprocessing for canonical eye and face patches
- Native JavaScript PCA with visible mean-feature and principal-component images
- Dense 17-point personalized calibration with fixation selection, blink rejection, balanced samples, and target retry
- Ridge regression, joint input/output PCA, residual outlier rejection, RLS click adaptation, and model ensembling
- Head-pose and eye-rotation features with a translation-aware 3D gaze-ray model
- Rotation-assisted output for improved robustness to moderate head movement
- Homography correction using Hartley-normalized DLT and JavaScript RANSAC
- One-Euro gaze smoothing and signal-loss/reacquisition handling
- Transactional five-point validation: a failed validation does not replace a working calibration
- Accuracy in pixels and estimated visual degrees, robust MAD precision, map stability, FPS benchmarks, and JSON export

## Gaze-estimation pipeline

1. MediaPipe detects facial, eyelid, eye, and iris landmarks.
2. OpenCV.js normalizes grayscale face and eye patches.
3. The app calculates iris position, eyelid opening, head pose, geometry, and 3D eye-ray features.
4. Native JavaScript PCA compresses the image features.
5. Personalized ridge regression and joint PCA estimate gaze.
6. A rotation-assisted model compensates for moderate head movement.
7. A homography maps predictions to normalized screen coordinates.
8. A One-Euro filter stabilizes the in-page gaze cursor.

## Technology

- HTML5, CSS3, and modern JavaScript modules
- [MediaPipe Tasks Vision](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
- OpenCV.js
- Web Workers and WebAssembly
- Browser-native numerical routines for PCA, ridge regression, RLS, homography fitting, and filtering

MediaPipe Tasks Vision, the Face Landmarker model, OpenCV.js, and the required WebAssembly files are included under `vendor/`, allowing the tracking pipeline to run without a remote inference service.

## Run locally

The app is static, but browser security requires it to be served over HTTP or HTTPS. Opening `index.html` through a `file://` URL will prevent JavaScript modules, Web Workers, WebAssembly, and camera access from loading correctly.

```powershell
npm start
```

Then open:

```text
http://127.0.0.1:8080/
```

Run the numeric regression tests with:

```powershell
npm test
```

## Browser and hardware notes

- Use a current desktop browser with camera, WebAssembly, Web Worker, and ES module support.
- Camera access requires `localhost` or HTTPS.
- Results depend on camera placement, lighting, glasses, screen dimensions, viewing distance, and calibration quality.
- Recalibration may be required after moving the camera, changing seating position, or significantly changing lighting.

## Limitations

- The model is calibrated per user and is not a generic pretrained gaze-estimation model.
- Rotation assistance improves moderate head-motion robustness but does not guarantee accurate tracking at extreme or previously unseen poses.
- Browser webcam testing cannot reproduce the precision of dedicated infrared eye-tracking hardware.
- No third-party proprietary eye-tracking binaries, private trained models, private assets, or telemetry are included.


