# Hybrid Webcam Gaze Tracker

Hybrid Webcam Gaze Tracker is a browser-based webcam eye-tracking app for exploring gaze,
head movement, and visual attention. It provides guided calibration, a live
gaze cursor, heatmaps, scan paths, and downloadable session data. Face and eye
processing use MediaPipe Face Landmarker and an optional hybrid pupil-detection pipeline.

## Hybrid eye-tracking approach

The app's core feature combines geometric eye/head measurements with a learned
mapping to screen coordinates:

1. MediaPipe detects facial and iris landmarks from webcam frames.
2. Geometry estimates head pose and eye direction. Eye-sphere calculations
   support the estimated direction rays shown in the diagnostic overlay.
3. Personal calibration learns a mapping from eye and head measurements to
   screen gaze coordinates.
4. Temporal filtering stabilizes measurements and output. Optional adaptive
   drift correction adjusts the calibrated mapping using verified targets.

Screen gaze is therefore a hybrid of geometry and personal calibration.
It is not obtained solely by intersecting a physically measured 3D eye ray
with a known monitor plane. The displayed eye rays are diagnostic estimates;
the calibrated screen mapping produces the gaze cursor coordinates.

Hybrid pupil mode additionally combines grayscale eye-image pupil detection
with landmark measurements and fallback behavior. This optional detector
combination is distinct from the geometry-plus-calibration architecture.

The original procedural head mesh supplies a modest fallback head-pose smoothing
reference; it is not the main gaze estimator. Registration captures a personal
landmark geometry profile for the current user.

## Features

- Hybrid gaze estimation combining eye/head geometry and personal calibration.
- Live gaze coordinates, head position estimates, and yaw/pitch/roll telemetry.
- Face mesh, pupil markers, and estimated eye-direction overlays.
- Guided multi-point calibration with neutral head registration, head-movement
  phases, lighting checks, and calibration accuracy reports.
- Regional One Euro landmark smoothing, with lighter filtering around the eyes
  and iris, plus an optional filter on the final gaze output.
- A personal landmark geometry profile captured during head registration.
  An original procedural head mesh provides a modest fallback reference.
- Adaptive drift correction learned from explicitly armed, user-confirmed targets,
  with quality gates, verification, rollback, Undo, and Reset.
- Live gaze bubble, heatmap, and scan-path controls.
- Fixation and saccade classification from gaze samples.
- CSV and JSON session exports, heatmap images, calibration accuracy images,
  and optional webcam recording, subject to browser support.
- Session import for comparing fixation heatmaps.

## Use the app

1. Open the site in a modern desktop browser and allow webcam access.
2. Select the camera resolution and tracking engine.
3. Face the camera, look forward, and register your head pose.
4. Run calibration and follow the animated targets and movement cues.
5. Review calibration accuracy, then start tracking.
6. Enable the gaze bubble, heatmap, or scan path as needed.
7. Start recording to collect a session, then stop and export the results.

For adaptive drift correction, enable Adaptive Drift, arm a target, hold your gaze
on a known screen point, and click that point. Continue looking at it while the
app verifies the correction. Accepted corrections apply to subsequent live
tracking; adaptive learning lasts for the current page session.

Recalibrate after changing the tracking engine, camera configuration, or other
settings that materially change the measurement pipeline. Accuracy depends on
lighting, camera quality, head position, glasses, and calibration quality.

## Run locally

From the parent project directory, if serve-local.cjs is present:

```powershell
npm start
```

Then open http://127.0.0.1:4173/.

Run the numeric regression tests with:

```powershell
npm test
```

## Technology and model provenance

- Plain HTML, CSS, and JavaScript modules.
- MediaPipe Tasks Vision 1.1.1, Face Landmarker, and browser camera/canvas APIs.
- Browser grayscale eye analysis and pupil tracking in hybrid mode.
- One Euro temporal filtering and robust recursive least squares drift correction.
- dist/assets/procedural-head.raw is original mathematical geometry with provenance.


## Limitations

Webcam gaze coordinates and head pose are estimates. The app is experimental,
and automated code checks do not establish real-world gaze accuracy or latency.
Eye-direction overlays are estimates rather than measured physical gaze rays.
Browser video codecs and performance vary. Calibration and drift quality checks
can reject samples when the signal is weak.
