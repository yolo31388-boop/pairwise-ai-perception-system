"""
pairwise-ai-perception-system - AIPerceptionSystem

A game AI perception system providing vision, hearing, smell, memory and
stimulus processing.

Fixes implemented:
- calculate_vision(): checks angle AND distance AND obstacles (line of
  sight), applies a clarity gradient (center sharp, edge blurry) and
  accumulates detection over time instead of instant spotting.
- calculate_hearing(): checks loudness AND distance AND obstacles (walls
  muffle sound), computes the direction the sound came from and keeps a
  short-term auditory memory.
- calculate_smell(): propagates odor with the wind, decays concentration
  with distance and accumulates a lingering scent trail over time.
- update_memory(): memories decay over time, carry a confidence that
  depends on the sensing modality, and predict the target's current
  position from its last known velocity.
- process_stimulus(): filters irrelevant / sub-threshold stimuli and
  picks the highest-priority one deterministically (vision > hearing >
  smell) instead of a random choice.
"""

import math


def _distance(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _bearing_degrees(origin, target):
    return math.degrees(math.atan2(target[1] - origin[1], target[0] - origin[0]))


def _angle_diff(a, b):
    """Smallest absolute difference between two headings in degrees."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _segment_intersects_circle(p1, p2, center, radius):
    """True if segment p1->p2 passes through the circle (an obstacle)."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return _distance(p1, center) <= radius
    t = max(0.0, min(1.0, ((center[0] - p1[0]) * dx + (center[1] - p1[1]) * dy) / length_sq))
    closest = (p1[0] + t * dx, p1[1] + t * dy)
    return _distance(closest, center) <= radius


class AIPerceptionSystem:
    # Sensing modality priorities (higher wins) and confidence weights.
    MODALITY_PRIORITY = {"vision": 3, "hearing": 2, "smell": 1}
    MODALITY_CONFIDENCE = {"vision": 1.0, "hearing": 0.6, "smell": 0.4}

    def __init__(self):
        self.state = {
            "time": 0.0,
            "vision_accumulation": {},   # target_id -> accumulated detection 0..1
            "hearing_memory": [],        # recent sounds: position/direction/loudness/time
            "scent_trail": [],           # lingering odor samples
            "memory": {},                # target_id -> remembered position/confidence/velocity
        }
        # Vision tuning.
        self.vision_range = 50.0
        self.vision_fov = 120.0          # total field-of-view angle in degrees
        self.vision_edge_clarity = 0.3   # clarity at the FOV edge (gradient)
        self.vision_accumulation_rate = 1.5   # detection gained per second at clarity 1
        self.vision_decay_rate = 0.8          # detection lost per second when unseen
        # Hearing tuning.
        self.hearing_range = 30.0
        self.wall_muffle_factor = 0.5    # loudness multiplier per wall between listener and sound
        self.hearing_memory_duration = 10.0
        # Smell tuning.
        self.smell_range = 20.0
        self.scent_decay_rate = 0.2      # concentration lost per second
        self.scent_diffusion = 8.0       # characteristic distance for odor falloff
        # Memory tuning.
        self.memory_decay_rate = 0.1     # confidence lost per second
        self.memory_forget_threshold = 0.05
        # Stimulus tuning.
        self.stimulus_thresholds = {"vision": 0.2, "hearing": 0.3, "smell": 0.2}

    # ------------------------------------------------------------------
    # Vision
    # ------------------------------------------------------------------
    def calculate_vision(self, observer=None, target=None, obstacles=None, dt=1.0):
        """Evaluate whether (and how well) the observer sees the target.

        Checks field-of-view angle, distance and line of sight, applies a
        clarity gradient across the FOV, and accumulates detection over
        time so a brief glimpse is not an instant full detection.
        """
        observer = observer or {}
        target = target or {}
        obstacles = obstacles or []
        obs_pos = tuple(observer.get("position", (0.0, 0.0)))
        tgt_pos = tuple(target.get("position", (0.0, 0.0)))
        facing = float(observer.get("facing", 0.0))
        target_id = target.get("id", "default")

        distance = _distance(obs_pos, tgt_pos)
        bearing = _bearing_degrees(obs_pos, tgt_pos)
        angle = _angle_diff(bearing, facing)

        in_angle = angle <= self.vision_fov / 2.0
        in_range = distance <= self.vision_range
        line_of_sight = in_range and not any(
            _segment_intersects_circle(obs_pos, tgt_pos, tuple(o.get("position", (0.0, 0.0))),
                                       float(o.get("radius", 1.0)))
            for o in obstacles
        )

        # Clarity gradient: sharpest at the FOV center, fading toward the
        # edges, and fading with distance.
        if in_angle and in_range:
            angle_factor = 1.0 - (1.0 - self.vision_edge_clarity) * (angle / (self.vision_fov / 2.0))
            distance_factor = 1.0 - 0.5 * (distance / self.vision_range)
            clarity = max(0.0, min(1.0, angle_factor * distance_factor))
        else:
            clarity = 0.0

        # Time accumulation of detection.
        accumulation = self.state["vision_accumulation"]
        level = accumulation.get(target_id, 0.0)
        if in_angle and in_range and line_of_sight:
            level = min(1.0, level + clarity * self.vision_accumulation_rate * dt)
        else:
            level = max(0.0, level - self.vision_decay_rate * dt)
        accumulation[target_id] = level

        visible = in_angle and in_range and line_of_sight and level >= 1.0
        result = {
            "visible": visible,
            "detection_level": level,
            "clarity": clarity if line_of_sight else 0.0,
            "distance": distance,
            "angle": angle,
            "in_fov": in_angle,
            "in_range": in_range,
            "line_of_sight": line_of_sight,
            "target_id": target_id,
        }
        if visible:
            self._record_observation(target_id, tgt_pos, "vision",
                                     velocity=target.get("velocity"))
        return result

    # ------------------------------------------------------------------
    # Hearing
    # ------------------------------------------------------------------
    def calculate_hearing(self, observer=None, sound=None, obstacles=None, dt=1.0):
        """Evaluate whether the observer hears a sound.

        Accounts for loudness, distance attenuation and obstacles that
        muffle sound, determines the direction of the sound, and stores
        it in a short-term auditory memory.
        """
        observer = observer or {}
        sound = sound or {}
        obstacles = obstacles or []
        obs_pos = tuple(observer.get("position", (0.0, 0.0)))
        sound_pos = tuple(sound.get("position", (0.0, 0.0)))
        loudness = float(sound.get("loudness", 0.0))

        distance = _distance(obs_pos, sound_pos)
        direction = _bearing_degrees(obs_pos, sound_pos)

        # Distance attenuation (inverse falloff against hearing range).
        if distance >= self.hearing_range:
            distance_factor = 0.0
        else:
            distance_factor = 1.0 - distance / self.hearing_range

        # Obstacles between listener and source muffle the sound.
        walls = sum(
            1 for o in obstacles
            if _segment_intersects_circle(obs_pos, sound_pos, tuple(o.get("position", (0.0, 0.0))),
                                          float(o.get("radius", 1.0)))
        )
        muffle = self.wall_muffle_factor ** walls

        perceived = loudness * distance_factor * muffle
        heard = perceived >= self.stimulus_thresholds["hearing"]

        # Auditory memory: keep recent sounds, forget old ones.
        now = self.state["time"] + dt
        self.state["time"] = now
        memory = self.state["hearing_memory"]
        memory[:] = [m for m in memory if now - m["time"] <= self.hearing_memory_duration]
        if heard:
            memory.append({
                "position": sound_pos,
                "direction": direction,
                "loudness": perceived,
                "time": now,
            })
            self._record_observation(sound.get("source_id", "sound"), sound_pos, "hearing")

        return {
            "heard": heard,
            "perceived_loudness": perceived,
            "distance": distance,
            "direction": direction,
            "walls_between": walls,
            "memory": list(memory),
        }

    # ------------------------------------------------------------------
    # Smell
    # ------------------------------------------------------------------
    def calculate_smell(self, observer=None, source=None, wind=None, dt=1.0):
        """Evaluate odor perception with wind propagation and decay.

        Odor drifts downwind over time, its concentration decays with
        distance, and scent lingers (accumulates) where the source has
        been instead of vanishing instantly.
        """
        observer = observer or {}
        source = source or {}
        wind = wind or {}
        obs_pos = tuple(observer.get("position", (0.0, 0.0)))
        src_pos = tuple(source.get("position", (0.0, 0.0)))
        intensity = float(source.get("intensity", 1.0))
        wind_dir = float(wind.get("direction", 0.0))
        wind_speed = float(wind.get("speed", 0.0))

        now = self.state["time"] + dt
        self.state["time"] = now

        # Deposit a fresh scent sample at the source's current position.
        trail = self.state["scent_trail"]
        if intensity > 0.0:
            trail.append({"position": src_pos, "concentration": intensity, "time": now})

        # Wind propagation: samples drift downwind as they age; decay over time.
        wind_vec = (math.cos(math.radians(wind_dir)) * wind_speed,
                    math.sin(math.radians(wind_dir)) * wind_speed)
        total = 0.0
        kept = []
        for sample in trail:
            age = now - sample["time"]
            concentration = sample["concentration"] * math.exp(-self.scent_decay_rate * age)
            if concentration < 0.01:
                continue  # fully dissipated
            drifted = (sample["position"][0] + wind_vec[0] * age,
                       sample["position"][1] + wind_vec[1] * age)
            sample["drifted_position"] = drifted
            sample["concentration"] = concentration
            kept.append(sample)
            # Distance decay of odor concentration at the observer.
            dist = _distance(obs_pos, drifted)
            if dist <= self.smell_range:
                total += concentration * math.exp(-dist / self.scent_diffusion)
        trail[:] = kept

        detected = total >= self.stimulus_thresholds["smell"]
        if detected:
            strongest = max(kept, key=lambda s: s["concentration"])
            self._record_observation(source.get("id", "scent"),
                                     strongest["drifted_position"], "smell")

        return {
            "detected": detected,
            "concentration": total,
            "trail_length": len(kept),
            "wind": {"direction": wind_dir, "speed": wind_speed},
        }

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------
    def update_memory(self, dt=1.0):
        """Decay memories, weight them by confidence, and predict positions.

        Confidence fades over time (memories are eventually forgotten),
        sightings are more trustworthy than sounds or smells, and a
        target's last known velocity is used to predict where it is now.
        """
        self.state["time"] += dt
        now = self.state["time"]
        memory = self.state["memory"]
        result = {}
        forgotten = []
        for target_id, entry in list(memory.items()):
            entry["confidence"] = max(0.0, entry["confidence"] - self.memory_decay_rate * dt)
            if entry["confidence"] < self.memory_forget_threshold:
                forgotten.append(target_id)
                del memory[target_id]
                continue
            elapsed = now - entry["last_update"]
            velocity = entry.get("velocity") or (0.0, 0.0)
            predicted = (entry["position"][0] + velocity[0] * elapsed,
                         entry["position"][1] + velocity[1] * elapsed)
            entry["predicted_position"] = predicted
            result[target_id] = dict(entry)
        return {
            "memories": result,
            "forgotten": forgotten,
            "time": now,
        }

    # ------------------------------------------------------------------
    # Stimulus processing
    # ------------------------------------------------------------------
    def process_stimulus(self, stimuli=None):
        """Filter and prioritize incoming stimuli.

        Stimuli below their modality threshold or of unknown type are
        filtered out; among the rest the highest-priority one wins
        (vision > hearing > smell, then intensity), deterministically.
        """
        stimuli = stimuli or []
        accepted = []
        filtered_out = []
        for stimulus in stimuli:
            stype = stimulus.get("type")
            intensity = float(stimulus.get("intensity", 0.0))
            if stype not in self.MODALITY_PRIORITY:
                filtered_out.append(stimulus)  # irrelevant stimulus type
                continue
            if intensity < self.stimulus_thresholds.get(stype, 1.0):
                filtered_out.append(stimulus)  # too weak to cause alert
                continue
            accepted.append(stimulus)

        selected = None
        if accepted:
            selected = max(
                accepted,
                key=lambda s: (self.MODALITY_PRIORITY[s["type"]],
                               float(s.get("intensity", 0.0))),
            )
        return {
            "selected": selected,
            "accepted": accepted,
            "filtered_out": filtered_out,
            "alert": selected is not None,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _record_observation(self, target_id, position, modality, velocity=None):
        """Store/refresh a memory entry with modality-based confidence."""
        memory = self.state["memory"]
        confidence = self.MODALITY_CONFIDENCE.get(modality, 0.5)
        entry = memory.get(target_id)
        if entry is None or confidence >= entry["confidence"]:
            memory[target_id] = {
                "position": tuple(position),
                "velocity": tuple(velocity) if velocity else (0.0, 0.0),
                "confidence": confidence,
                "modality": modality,
                "last_update": self.state["time"],
            }
