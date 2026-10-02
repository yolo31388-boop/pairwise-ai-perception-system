"""
pairwise-ai-perception-system - AIPerceptionSystem

A perception system for game AI with five subsystems:

- vision: distance / FOV / occlusion checks, peripheral falloff, time accumulation
- hearing: distance attenuation, obstacle dampening, direction, memory
- smell: wind propagation, concentration decay, lingering scent trail
- memory: confidence decay, per-sense reliability, position prediction
- stimulus: priority, filtering and alert thresholds
"""

import math
import time


def _distance(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _angle_to(origin, target):
    return math.degrees(math.atan2(target[1] - origin[1], target[0] - origin[0]))


def _angle_diff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _segment_circle_blocked(p1, p2, center, radius):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return _distance(p1, center) <= radius
    t = max(0.0, min(1.0, ((center[0] - p1[0]) * dx + (center[1] - p1[1]) * dy) / length_sq))
    closest = (p1[0] + t * dx, p1[1] + t * dy)
    return _distance(closest, center) <= radius


def _obstacles_between(p1, p2, obstacles):
    hits = 0
    for obstacle in obstacles or []:
        if isinstance(obstacle, dict):
            center = tuple(obstacle.get("position", (0.0, 0.0)))
            radius = float(obstacle.get("radius", 1.0))
        else:
            center, radius = tuple(obstacle[0]), float(obstacle[1])
        if _segment_circle_blocked(p1, p2, center, radius):
            hits += 1
    return hits


class AIPerceptionSystem:
    VISION_PRIORITY = 3
    HEARING_PRIORITY = 2
    SMELL_PRIORITY = 1

    SENSE_CONFIDENCE = {"vision": 1.0, "hearing": 0.6, "smell": 0.4}
    MEMORY_DECAY_RATES = {"vision": 0.05, "hearing": 0.15, "smell": 0.10}
    STIMULUS_THRESHOLDS = {"vision": 0.2, "hearing": 0.15, "smell": 0.25}
    STIMULUS_PRIORITIES = {"vision": VISION_PRIORITY, "hearing": HEARING_PRIORITY, "smell": SMELL_PRIORITY}

    def __init__(self):
        self.state = {}
        self.memory = []
        self.vision_accumulator = {}
        self.scent_trail = []
        self.time = 0.0

    def calculate_vision(self, observer=None, target=None, obstacles=None, dt=1.0,
                         view_distance=50.0, fov_degrees=120.0, time_to_detect=1.0):
        """Vision with distance, occlusion, peripheral falloff and time accumulation."""
        observer = observer or {"position": (0.0, 0.0), "facing": 0.0}
        target = target or {"position": (10.0, 0.0), "id": "default"}
        self.time += dt

        observer_pos = tuple(observer.get("position", (0.0, 0.0)))
        facing = float(observer.get("facing", 0.0))
        target_pos = tuple(target.get("position", (0.0, 0.0)))
        target_id = target.get("id", "default")

        distance = _distance(observer_pos, target_pos)
        angle = _angle_diff(_angle_to(observer_pos, target_pos), facing)
        half_fov = fov_degrees / 2.0

        in_range = distance <= view_distance
        in_fov = angle <= half_fov
        blocked = _obstacles_between(observer_pos, target_pos, obstacles) > 0 if (in_range and in_fov) else False
        visible = in_range and in_fov and not blocked

        distance_factor = max(0.0, 1.0 - distance / view_distance) if in_range else 0.0
        angle_factor = max(0.0, 1.0 - (angle / half_fov) ** 2) if in_fov else 0.0
        clarity = distance_factor * angle_factor

        previous = self.vision_accumulator.get(target_id, 0.0)
        if visible:
            detection = min(1.0, previous + clarity * dt / max(time_to_detect, 1e-6))
        else:
            detection = max(0.0, previous - 0.5 * dt)
        self.vision_accumulator[target_id] = detection

        detected = detection >= 1.0
        if detected:
            self._remember(target_pos, "vision", velocity=target.get("velocity"))

        return {
            "visible": visible,
            "detected": detected,
            "detection": detection,
            "distance": distance,
            "angle": angle,
            "distance_factor": distance_factor,
            "angle_factor": angle_factor,
            "occluded": blocked,
        }

    def calculate_hearing(self, listener=None, sound=None, obstacles=None, dt=1.0,
                          hearing_range=40.0, obstacle_dampening=0.5):
        """Hearing with distance attenuation, obstacle dampening, direction and memory."""
        listener = listener or {"position": (0.0, 0.0), "facing": 0.0}
        sound = sound or {"position": (5.0, 0.0), "volume": 0.8}
        self.time += dt

        listener_pos = tuple(listener.get("position", (0.0, 0.0)))
        facing = float(listener.get("facing", 0.0))
        sound_pos = tuple(sound.get("position", (0.0, 0.0)))
        volume = float(sound.get("volume", 0.5))

        distance = _distance(listener_pos, sound_pos)
        distance_factor = max(0.0, 1.0 - distance / hearing_range)
        walls = _obstacles_between(listener_pos, sound_pos, obstacles)
        occlusion_factor = obstacle_dampening ** walls
        effective_volume = volume * distance_factor * occlusion_factor

        heard = effective_volume >= self.STIMULUS_THRESHOLDS["hearing"]
        absolute_bearing = _angle_to(listener_pos, sound_pos)
        direction = _angle_diff(absolute_bearing, facing)
        if absolute_bearing - facing < -180.0 or (absolute_bearing - facing) % 360.0 > 180.0:
            direction = -direction

        if heard:
            self._remember(sound_pos, "hearing")

        return {
            "heard": heard,
            "effective_volume": effective_volume,
            "distance": distance,
            "direction": direction,
            "bearing": absolute_bearing,
            "occlusion": walls,
        }

    def calculate_smell(self, smeller=None, source=None, wind=None, dt=1.0,
                        smell_range=20.0, decay_rate=0.2, trail_lifetime=30.0):
        """Smell with wind propagation, concentration decay and a lingering scent trail."""
        smeller = smeller or {"position": (0.0, 0.0)}
        source = source or {"position": (5.0, 0.0), "intensity": 1.0}
        wind = wind or (0.0, 0.0)
        self.time += dt

        smeller_pos = tuple(smeller.get("position", (0.0, 0.0)))
        source_pos = tuple(source.get("position", (0.0, 0.0)))
        intensity = float(source.get("intensity", 1.0))

        self.scent_trail.append({
            "position": source_pos,
            "intensity": intensity,
            "timestamp": self.time,
        })
        self.scent_trail = [
            s for s in self.scent_trail
            if self.time - s["timestamp"] <= trail_lifetime
        ]

        wind_speed = math.hypot(wind[0], wind[1])
        best = 0.0
        best_position = None
        for scent in self.scent_trail:
            age = self.time - scent["timestamp"]
            drifted = (
                scent["position"][0] + wind[0] * age,
                scent["position"][1] + wind[1] * age,
            )
            distance = _distance(smeller_pos, drifted)
            if distance > smell_range:
                continue
            concentration = (
                scent["intensity"]
                * math.exp(-decay_rate * distance)
                * math.exp(-age / max(trail_lifetime, 1e-6))
            )
            if concentration > best:
                best = concentration
                best_position = drifted

        detected = best >= self.STIMULUS_THRESHOLDS["smell"]
        if detected:
            self._remember(best_position, "smell")

        return {
            "detected": detected,
            "concentration": best,
            "source_position": best_position,
            "wind_speed": wind_speed,
            "trail_length": len(self.scent_trail),
        }

    def update_memory(self, stimulus=None, dt=1.0):
        """Memory with confidence decay, per-sense reliability and position prediction."""
        self.time += dt
        if stimulus is not None:
            self._remember(
                tuple(stimulus.get("position", (0.0, 0.0))),
                stimulus.get("sense", "vision"),
                velocity=stimulus.get("velocity"),
            )

        active = []
        for entry in self.memory:
            elapsed = self.time - entry["timestamp"]
            decay = self.MEMORY_DECAY_RATES.get(entry["sense"], 0.1)
            entry["confidence"] = max(
                0.0,
                self.SENSE_CONFIDENCE.get(entry["sense"], 0.5) * math.exp(-decay * elapsed),
            )
            if entry["confidence"] <= 0.01:
                continue
            velocity = entry.get("velocity") or (0.0, 0.0)
            predicted = (
                entry["position"][0] + velocity[0] * elapsed,
                entry["position"][1] + velocity[1] * elapsed,
            )
            active.append({**entry, "predicted_position": predicted})
        self.memory = [e for e in self.memory if e["confidence"] > 0.01]
        return active

    def process_stimulus(self, stimuli=None, thresholds=None):
        """Stimulus processing with priority, filtering and alert thresholds."""
        thresholds = thresholds or self.STIMULUS_THRESHOLDS
        stimuli = stimuli if stimuli is not None else []

        filtered = []
        ignored = []
        for stimulus in stimuli:
            sense = stimulus.get("type", stimulus.get("sense", "vision"))
            intensity = float(stimulus.get("intensity", 0.0))
            threshold = thresholds.get(sense, 0.2)
            if intensity >= threshold:
                filtered.append((sense, intensity, stimulus))
            else:
                ignored.append(stimulus)

        selected = None
        if filtered:
            filtered.sort(
                key=lambda item: (
                    self.STIMULUS_PRIORITIES.get(item[0], 0),
                    item[1],
                ),
                reverse=True,
            )
            selected = filtered[0][2]

        return {
            "selected": selected,
            "candidates": [item[2] for item in filtered],
            "filtered_out": ignored,
            "alert": selected is not None,
        }

    def _remember(self, position, sense, velocity=None):
        entry = {
            "position": tuple(position),
            "sense": sense,
            "confidence": self.SENSE_CONFIDENCE.get(sense, 0.5),
            "timestamp": self.time,
            "velocity": tuple(velocity) if velocity else None,
        }
        for existing in self.memory:
            if existing["sense"] == sense and _distance(existing["position"], entry["position"]) < 1.0:
                existing.update(entry)
                return entry
        self.memory.append(entry)
        return entry
