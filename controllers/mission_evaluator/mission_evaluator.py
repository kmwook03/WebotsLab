"""Independent, one-way mission evaluator. It never informs the AMR."""

import json
import math

from controller import Supervisor


class MissionEvaluator:
    def __init__(self):
        self.supervisor = Supervisor()
        self.time_step = int(self.supervisor.getBasicTimeStep())
        self.receiver = self.supervisor.getDevice("evaluator receiver")
        self.receiver.enable(self.time_step)
        self.robot = self.supervisor.getFromDef("AMR")
        self.targets = [
            self.supervisor.getFromDef("RESCUE_TARGET"),
            self.supervisor.getFromDef("RESCUE_TARGET_2"),
            self.supervisor.getFromDef("RESCUE_TARGET_3"),
        ]
        self.humans = [
            self.supervisor.getFromDef("MOVING_PERSON"),
            self.supervisor.getFromDef("MOVING_PERSON_2"),
            self.supervisor.getFromDef("MOVING_PERSON_3"),
        ]
        self.human_translations = [human.getField("translation") for human in self.humans]
        self.start = tuple(self.robot.getField("translation").getSFVec3f()[:2])
        self.target_reached = [False for _ in self.targets]
        self.closest_human = math.inf
        self.closest_humans = [math.inf for _ in self.humans]
        self.reported_pass = False
        self.reported_complete_pose = False
        self.last_phase = "UNKNOWN"

    @staticmethod
    def _distance(first, second):
        return math.hypot(first[0] - second[0], first[1] - second[1])

    @staticmethod
    def _triangle(value):
        phase = value % 2.0
        return phase if phase <= 1.0 else 2.0 - phase

    def run(self):
        print("[EVALUATOR] Ground truth is isolated from the robot controller.")
        while self.supervisor.step(self.time_step) != -1:
            now = self.supervisor.getTime()
            # Three independently moving people cross different useful routes.
            # Their mild lateral weave prevents a controller from succeeding by
            # memorising one constant-velocity line. No state is sent to the AMR.
            lower_span = 3.65
            lower_x = -0.75 + lower_span * self._triangle(now * 0.30 / lower_span)
            lower_y = -1.75 + 0.18 * math.sin(1.35 * now)

            right_span = 4.05
            right_x = 1.98 + 0.12 * math.sin(1.10 * now + 0.70)
            right_y = -1.15 + right_span * self._triangle(now * 0.30 / right_span)

            # Cross the only useful top connector, but do not camp directly in
            # front of either rescue object at x=+-2.7. This tests prediction
            # and waiting behaviour instead of turning target vision off.
            upper_span = 3.00
            upper_x = -1.50 + upper_span * self._triangle((now + 1.40) * 0.52 / upper_span)
            upper_y = 3.05 + 0.12 * math.sin(1.70 * now + 1.10)

            positions = (
                (lower_x, lower_y, 0.62),
                (right_x, right_y, 0.62),
                (upper_x, upper_y, 0.62),
            )
            for field, position in zip(self.human_translations, positions):
                field.setSFVec3f(list(position))
            while self.receiver.getQueueLength() > 0:
                try:
                    payload = json.loads(self.receiver.getString())
                    self.last_phase = payload.get("phase", self.last_phase)
                except (ValueError, TypeError):
                    pass
                self.receiver.nextPacket()

            robot_position = self.robot.getPosition()
            target_distances = [
                self._distance(robot_position, target.getPosition()) for target in self.targets
            ]
            human_distances = [self._distance(robot_position, human.getPosition()) for human in self.humans]
            human_distance = min(human_distances)
            self.closest_human = min(self.closest_human, human_distance)
            self.closest_humans = [
                min(previous, current)
                for previous, current in zip(self.closest_humans, human_distances)
            ]

            for index, target_distance in enumerate(target_distances):
                if not self.target_reached[index] and target_distance < 0.68:
                    self.target_reached[index] = True
                    print(f"[EVALUATOR] target {index + 1}/{len(self.targets)} reached at t={now:.1f}s")

            home_distance = self._distance(robot_position, self.start)
            if self.last_phase == "COMPLETE" and not self.reported_complete_pose:
                self.reported_complete_pose = True
                print(f"[EVALUATOR] controller complete; actual home distance={home_distance:.3f}m")
            if all(self.target_reached) and home_distance < 0.32 and self.last_phase == "COMPLETE":
                print(
                    f"EVALUATION: PASS | all {len(self.targets)} targets reached, returned home, "
                    f"closest moving-person separation={self.closest_human:.3f}m "
                    f"across {len(self.humans)} people",
                    flush=True,
                )
                self.supervisor.step(self.time_step)
                self.reported_pass = True
                self.supervisor.simulationQuit(0)
                return

            if human_distance < 0.27:
                person_index = human_distances.index(human_distance) + 1
                print(
                    f"EVALUATION: FAIL | unsafe separation from person {person_index}: "
                    f"{human_distance:.3f}m",
                    flush=True,
                )
                self.supervisor.step(self.time_step)
                self.supervisor.simulationQuit(2)
                return
            if now > 440.0:
                print(
                    f"EVALUATION: TIMEOUT | phase={self.last_phase}, "
                    f"targets_reached={self.target_reached}, home_distance={home_distance:.2f}m"
                )
                self.supervisor.simulationQuit(3)
                return


if __name__ == "__main__":
    MissionEvaluator().run()

