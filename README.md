# TECH WEEK Autonomous Search and Rescue

This Webots R2025a project implements the hackathon mission from the supplied
plan: build a map in an unknown environment, find a visually specified target,
reach it without collisions, and return to the known starting pose.

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
   stale-obstacle decay, and footprint inflation.
4. **Perception** - raw-camera red-target segmentation, denoising, bounding-box
   extraction, temporal confidence, and LiDAR range association. Webots
   recognition metadata is not used.
5. **Global planning** - connected frontier selection, information-gain scoring,
   8-connected A*, clearance-aware costs, and line-of-sight path smoothing.
6. **Local planning** - acceleration-constrained Dynamic Window Approach (DWA)
   with trajectory, heading, clearance, progress, and braking critics.
7. **Safety and mission** - an independent emergency/TTC guard, stuck recovery,
   and an explicit `BOOTSTRAP -> EXPLORE -> TARGET_APPROACH -> CONFIRM_TARGET ->
   RETURN_HOME -> COMPLETE` state machine.

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
supervisor prints `EVALUATION: PASS` after target arrival and safe return.

## Tests

From this directory, with NumPy available:

```powershell
python -m unittest discover -s tests -v
```

The tests cover geometry, occupancy updates, A* around an obstacle, frontier
selection, DWA collision rejection, 360-degree proximity escape, target
detection, and the mission state transitions.

## Tunable parameters

All physical and planner constants are in
`controllers/amr_search_rescue/config.py`. Sensor-facing code is confined to
`amr_search_rescue.py`; the other modules accept plain data structures and can
be reused on another differential-drive robot.

