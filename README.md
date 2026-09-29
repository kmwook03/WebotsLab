# TECH WEEK Autonomous Search and Rescue

This Webots R2025a project implements the hackathon mission from the supplied
plan: build a map in an unknown environment, find three visually specified
targets, reach each unique target without collisions, and return to the known
starting pose.

The challenge world includes three independently moving people with different
crossing directions and sinusoidal lateral motion. The robot controller never
reads Webots ground-truth position, target position,
or object labels. A separate supervisor only observes the run and reports the
score; it does not send information to the robot.

## Architecture

The stack is deliberately split along interfaces used in production mobile
robotics:

1. **Sensors and actuation** - wheel encoders, gyro, LDS-01 LiDAR, RGB camera,
   and differential-drive motor commands.
2. **State estimation** - encoder/gyro dead reckoning with gated correlative
   LiDAR scan matching and a maintained 3x3 pose covariance.
3. **Mapping** - NumPy log-odds occupancy grid, ray casting, evidence clamping,
   stale-obstacle decay, dynamic-hit exclusion, and footprint inflation.
4. **Perception** - raw-camera red-target segmentation, denoising, bounding-box
   extraction, temporal confidence, and LiDAR range association. Webots
   recognition metadata is not used.
5. **Global planning** - connected frontier selection, information-gain scoring,
   8-connected A*, clearance-aware costs, and line-of-sight path smoothing.
6. **Dynamic tracking and local planning** - ego-motion-compensated compact
   LiDAR cluster tracking plus acceleration-constrained DWA. Candidate rollouts
   reject both static returns and predicted moving-obstacle positions. A
   motion-consistent two-hit planning track gives DWA early warning before the
   stricter confirmed track is used for mapping and telemetry; short occlusions
   are projected forward from the last observation time.
7. **Safety and mission** - an independent predictive TTC guard, scored
   emergency manoeuvres, release hysteresis, stuck recovery,
   visited-target position/bearing de-duplication, and an explicit `BOOTSTRAP ->
   EXPLORE -> TARGET_APPROACH -> CONFIRM_TARGET` loop followed by `RETURN_HOME ->
   COMPLETE` after all three unique targets are confirmed. A locked approach
   accepts only directionally consistent, unvisited observations, and direct
   visual servoing takes priority while that target remains visible.

This decomposition keeps mission policy out of estimation and control, makes
the algorithms unit-testable without Webots, and prevents the local planner from
bypassing the hard safety layer.

## Run

Install NumPy in the Python interpreter configured in Webots, then open:

```text
worlds/amr_search_rescue.wbt
```

For a command-line run on Windows:

```powershell
& 'C:\Program Files\Webots\msys64\mingw64\bin\webots.exe' --batch --mode=fast --stdout --stderr worlds\amr_search_rescue.wbt
```

Expected controller milestones are printed as `[MISSION] ...`. The independent
supervisor prints `EVALUATION: PASS` after all three target arrivals and safe
return.

## Tests

From this directory, with NumPy available:

```powershell
python -m unittest discover -s tests -v
```

The tests cover geometry, occupancy updates, A* around an obstacle, frontier
selection, dynamic-cluster tracking, predicted crossing rejection, safety
hysteresis, dynamic-map exclusion, target detection, visited-target rejection,
target-lock continuity, and mission transitions.

Status messages include confirmed/planning/active track counts, predicted TTC, the
active safety reason, and compact track position/velocity summaries. The map
display draws confirmed tracks and their one-second velocity vectors in orange.

## Tunable parameters

All physical and planner constants are in
`controllers/amr_search_rescue/config.py`. Sensor-facing code is confined to
`amr_search_rescue.py`; the other modules accept plain data structures and can
be reused on another differential-drive robot.

