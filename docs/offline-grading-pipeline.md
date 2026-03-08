# Offline Grading Pipeline

Lap-Trackr grading is offline and accuracy-first. Once a session has been exported to transformed video and depth artifacts, the grader should use the full temporal context of the procedure instead of a tiered low-latency pipeline.

## Problem Framing

The grading task is to recover highly accurate, identity-stable 3D trajectories for two laparoscopic instrument tips:

- `green_tip`
- `pink_tip`

The output must be good enough for downstream grading metrics such as path length, economy of motion, smoothness, task time, and time spent in regions of interest. That shifts the design toward multi-pass analysis, global consistency, and explicit confidence scoring.

## Primary Model Stack

The target offline stack is:

- **SAM2** for dense video segmentation in each camera view
- **CoTracker3** for sub-pixel point tracking and occlusion-aware temporal refinement
- **Color-based analysis** for gap filling and identity verification

Optional auxiliary models may still be used, but they are not the primary source of truth:

- **YOLO11 pose** for tip proposal, re-acquisition after long losses, and QA checks

## Pass Structure

### Pass 1: Dense Segmentation With SAM2

Run SAM2 independently on each camera view. Prompt it with one green-tape region and one pink-tape region on an initialization frame, then propagate both mask tracks through the video using the model's offline temporal memory.

This pass should produce:

- one green mask track per view
- one pink mask track per view
- explicit gap intervals for full occlusion, smoke, or tool exit

SAM2 provides the spatial extent of the taped region and should be treated as the primary segmentation signal.

### Pass 2: Point Refinement With CoTracker3

Run CoTracker3 from manually confirmed tip points on the same initialization frame(s). Track the exact tip point for both instruments in each view.

This pass should produce:

- sub-pixel 2D tip tracks
- visibility / occlusion confidence
- long-range temporal continuity through partial occlusion

CoTracker3 is the primary point-level tracker. The tracked point should remain inside or on the boundary of the corresponding SAM2 mask. Frames that violate that agreement should be flagged for review or lower confidence.

### Pass 3: Color-Based Gap Filling

For frames where both SAM2 and CoTracker3 lose the tool, fall back to color analysis on the green and pink tape.

Because this is offline, color thresholds should be adapted from the full video rather than fixed once at startup. The preferred approach is:

- collect confirmed green and pink regions from successful SAM2 segments
- estimate stable hue / saturation statistics over time
- use those statistics to fill gaps conservatively

Color is also the strongest identity cue because there is never more than one green instrument and one pink instrument.

### Pass 4: Multi-View Triangulation

Do not treat the ZED depth map as the final 3D answer. Use matched 2D tip observations and calibrated camera matrices to triangulate 3D positions directly.

With two stereo ZED cameras, the system can use up to four synchronized views. The preferred formulation is overdetermined triangulation using all valid views for a frame, with reprojection error retained as a confidence signal.

This pass should produce:

- per-frame 3D positions for `green_tip` and `pink_tip`
- per-frame reprojection residuals
- per-view outlier rejection when one camera view is inconsistent

### Pass 5: Trajectory Smoothing and Global Optimization

After triangulation, smooth the full 3D trajectories with a fixed-interval method that can use future frames as well as past frames.

Recommended options:

- smoothing spline
- Rauch-Tung-Striebel (RTS) smoother
- sparse batch optimization minimizing reprojection error plus trajectory smoothness

If a trocar / fulcrum point is available, add it as a constraint during optimization. That materially improves conditioning and reduces implausible tool motion.

This pass should prioritize:

- low jitter
- no phase lag
- identity stability through crossings
- physically plausible trajectories

### Pass 6: Identity Verification

Run a final pass to confirm that green and pink identities never swap.

Recommended checks:

- compare color histograms inside the SAM2 masks over time
- check continuity against the smoothed 3D trajectory
- reject identity flips during crossings unless evidence is overwhelming

Temporary blanks are better than silent identity swaps.

## Calibration Requirements

Offline grading makes calibration accuracy more important, not less.

Required calibration inputs:

- per-camera intrinsics
- stereo extrinsics between the ZED units
- any board-to-camera or world-frame transforms used for reporting

Recommended practice:

- maintain the standard ChArUco-based stereo calibration
- record a dedicated calibration motion sequence with a known target moving through the workspace
- quantify reprojection error empirically at the working distance of the instruments

Also prefer recording at the highest practical sensor resolution and frame rate. Downsampling can happen later; missing spatial detail cannot be recovered.

## Output Contract

The grader should emit more than just final metrics.

Required outputs:

- final smoothed 3D trajectories for `green_tip` and `pink_tip`
- timestamps for every frame
- per-frame confidence values
- per-frame provenance describing which pass supplied the observation
- reprojection residuals for triangulated points
- review flags for disagreement between segmentation, tracking, and color identity

Derived grading metrics such as path length, smoothness / jerk, time to task completion, and economy of motion should be computed from these cleaned 3D trajectories rather than from raw per-frame detections.

## Practical Notes

- The normal grader input is the exported artifact set, not the raw SVO2 file path.
- The offline pipeline should use the transformed exports the user already validated during initialization.
- SAM2 and CoTracker3 are the primary grading models.
- YOLO11 pose is optional support only; it should not replace the offline multi-pass stack.
