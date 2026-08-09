"""Online fixed-budget controller for CoFrame."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any

from .mesh import adaptive_mesh_indices, frame_neighbors


@dataclass(frozen=True)
class ControllerConfig:
    total_budget: int = 9
    validator_count: int = 1
    ema_decay: float = 0.65
    exploration_weight: float = 0.05
    prior_weight: float = 0.35
    age_weight: float = 0.02
    distance_power: float = 2.0
    inertia: float = 0.15
    update_every_blocks: int = 1
    spread_decay: float = 0.65

    def validate(self, num_frames: int) -> None:
        if not 3 <= self.total_budget <= num_frames:
            raise ValueError(
                f"total_budget must be in [3, {num_frames}], got {self.total_budget}"
            )
        if not 1 <= self.validator_count <= self.total_budget - 2:
            raise ValueError(
                "validator_count must leave at least two core endpoints: "
                f"{self.validator_count}/{self.total_budget}"
            )
        if self.update_every_blocks < 1:
            raise ValueError("update_every_blocks must be >= 1")


@dataclass(frozen=True)
class Selection:
    block_index: int
    core: tuple[int, ...]
    validators: tuple[int, ...]
    selected: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class CoFrameController:
    """Rhyme-initialized, defect-corrected temporal mesh controller.

    ``total_budget`` includes validator frames.  Therefore CoFrame is compared
    with fixed and Rhyme baselines at the same exact number of block-evaluated
    frames; self-validation is not hidden as extra compute.
    """

    def __init__(
        self,
        *,
        num_frames: int,
        initial_core: list[int],
        prior_scores: list[float],
        config: ControllerConfig | None = None,
    ) -> None:
        self.num_frames = int(num_frames)
        self.config = config or ControllerConfig()
        self.config.validate(self.num_frames)
        self.core_budget = self.config.total_budget - self.config.validator_count
        if len(prior_scores) != self.num_frames:
            raise ValueError(
                f"expected {self.num_frames} prior scores, got {len(prior_scores)}"
            )
        self.prior = [max(float(value), 0.0) for value in prior_scores]
        self.risk = [0.0 for _ in range(self.num_frames)]
        self.visits = [0 for _ in range(self.num_frames)]
        self.last_validated = [-1 for _ in range(self.num_frames)]
        self.history: list[dict[str, Any]] = []

        core = sorted(set(int(index) for index in initial_core))
        core = sorted(set(core + [0, self.num_frames - 1]))
        if len(core) != self.core_budget:
            seed_scores = [
                self.config.prior_weight * value + (1.0 if index in core else 0.0)
                for index, value in enumerate(self.prior)
            ]
            core = adaptive_mesh_indices(
                self.num_frames,
                self.core_budget,
                seed_scores,
                current=core,
                distance_power=self.config.distance_power,
                inertia=self.config.inertia,
            )
        self.core = core

    def _candidate_utility(self, frame: int, block_index: int) -> float:
        left, right = frame_neighbors(self.core, frame)
        interval_width = right - left
        exploration = self.config.exploration_weight / sqrt(self.visits[frame] + 1)
        age = block_index - self.last_validated[frame]
        age_bonus = self.config.age_weight * max(age, 0)
        local_risk = max(self.risk[left : right + 1], default=0.0)
        semantic = self.config.prior_weight * self.prior[frame]
        return (
            local_risk + self.risk[frame] + semantic + exploration + age_bonus + 1e-6
        ) * (interval_width**self.config.distance_power)

    def select(self, block_index: int) -> Selection:
        candidates = [
            frame
            for frame in range(1, self.num_frames - 1)
            if frame not in self.core
        ]
        validators: list[int] = []
        for _ in range(self.config.validator_count):
            available = [frame for frame in candidates if frame not in validators]
            if not available:
                break
            chosen = max(
                available,
                key=lambda frame: (self._candidate_utility(frame, block_index), -frame),
            )
            validators.append(chosen)
        selected = sorted(set(self.core + validators))
        if len(selected) != self.config.total_budget:
            raise RuntimeError(
                f"controller emitted {len(selected)} frames for budget {self.config.total_budget}: {selected}"
            )
        return Selection(
            block_index=int(block_index),
            core=tuple(self.core),
            validators=tuple(sorted(validators)),
            selected=tuple(selected),
        )

    def _spread_observation(self, validator: int, defect: float) -> None:
        left, right = frame_neighbors(self.core, validator)
        width = max(right - left, 1)
        for frame in range(left, right + 1):
            distance = abs(frame - validator) / width
            weight = self.config.spread_decay**distance
            observation = defect * weight
            self.risk[frame] = (
                self.config.ema_decay * self.risk[frame]
                + (1.0 - self.config.ema_decay) * observation
            )

    def _reallocate_core(self) -> None:
        scores = []
        max_visit = max(self.visits, default=0)
        for frame in range(self.num_frames):
            stale = max_visit - self.visits[frame]
            score = (
                self.risk[frame]
                + self.config.prior_weight * self.prior[frame]
                + self.config.exploration_weight / sqrt(self.visits[frame] + 1)
                + self.config.age_weight * stale
            )
            scores.append(score)
        self.core = adaptive_mesh_indices(
            self.num_frames,
            self.core_budget,
            scores,
            current=self.core,
            distance_power=self.config.distance_power,
            inertia=self.config.inertia,
        )

    def observe(
        self,
        selection: Selection,
        defects: dict[int, float],
    ) -> None:
        if tuple(self.core) != selection.core:
            raise RuntimeError("controller core changed between select() and observe()")
        for validator in selection.validators:
            if validator not in defects:
                raise ValueError(f"missing defect for validator {validator}")
            value = max(float(defects[validator]), 0.0)
            self.visits[validator] += 1
            self.last_validated[validator] = selection.block_index
            self._spread_observation(validator, value)

        record = selection.as_dict()
        record["defects"] = {str(key): float(value) for key, value in defects.items()}
        record["risk_after_observation"] = list(self.risk)
        self.history.append(record)

        if (selection.block_index + 1) % self.config.update_every_blocks == 0:
            self._reallocate_core()

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_frames": self.num_frames,
            "config": asdict(self.config),
            "core": list(self.core),
            "prior": list(self.prior),
            "risk": list(self.risk),
            "visits": list(self.visits),
            "last_validated": list(self.last_validated),
            "history": list(self.history),
        }
