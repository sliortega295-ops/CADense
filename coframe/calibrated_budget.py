from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite


def schedule_slot_key(value: str) -> tuple[int, int]:
    step, group = str(value).split(":", 1)
    return int(step), int(group)


@dataclass(frozen=True, slots=True)
class ExactBudgetSchedule:
    schedule: dict[str, int]
    objective: float
    uniform_k9_objective: float
    target_total_k: int

    @property
    def average_k(self) -> float:
        return sum(self.schedule.values()) / len(self.schedule)

    @property
    def budget_counts(self) -> dict[int, int]:
        return dict(sorted(Counter(self.schedule.values()).items()))

    @property
    def relative_training_improvement_over_uniform_k9(self) -> float:
        return (self.uniform_k9_objective - self.objective) / (self.uniform_k9_objective + 1.0e-12)

    def to_dict(self) -> dict[str, object]:
        return {
            "schedule": dict(self.schedule),
            "objective": self.objective,
            "uniform_k9_objective": self.uniform_k9_objective,
            "relative_training_improvement_over_uniform_k9": (
                self.relative_training_improvement_over_uniform_k9
            ),
            "target_total_k": self.target_total_k,
            "average_k": self.average_k,
            "budget_counts": {str(key): value for key, value in self.budget_counts.items()},
        }


def optimize_exact_budget_schedule(
    costs: Mapping[str, Mapping[int, float]],
    *,
    budgets: Sequence[int] = (6, 9, 12, 21),
    target_average_k: float = 9.0,
) -> ExactBudgetSchedule:
    """Minimize additive slot cost under an exact total-K constraint.

    Costs must already be computed from training prompts only. Dynamic
    programming is exact; the only tie-breaker after equal objective is fewer
    non-K9 assignments, followed by deterministic iteration order.
    """
    budget_values = tuple(int(value) for value in budgets)
    if budget_values != (6, 9, 12, 21):
        raise ValueError("calibrated schedule is preregistered with K in {6,9,12,21}")
    slots = sorted((str(key) for key in costs), key=schedule_slot_key)
    if not slots:
        raise ValueError("cost surface contains no schedule slots")
    target_total_float = float(target_average_k) * len(slots)
    target_total = int(round(target_total_float))
    if abs(target_total_float - target_total) > 1.0e-9:
        raise ValueError("target average does not produce an integral total budget")

    normalized: dict[str, dict[int, float]] = {}
    for slot in slots:
        values = {int(key): float(value) for key, value in costs[slot].items()}
        if set(values) != set(budget_values):
            raise ValueError(f"slot {slot} does not contain exactly {budget_values}")
        if any(not isfinite(value) or value < 0.0 for value in values.values()):
            raise ValueError(f"slot {slot} contains an invalid error")
        normalized[slot] = values

    # sum_k -> (objective, non_k9_count). Parent maps are kept per stage for
    # exact reconstruction without retaining full paths in every DP state.
    states: dict[int, tuple[float, int]] = {0: (0.0, 0)}
    parents: list[dict[int, tuple[int, int]]] = []
    preference = (9, 6, 12, 21)
    for slot in slots:
        next_states: dict[int, tuple[float, int]] = {}
        parent: dict[int, tuple[int, int]] = {}
        for total, (objective, non_k9) in sorted(states.items()):
            for budget in preference:
                new_total = total + budget
                candidate = (
                    objective + normalized[slot][budget],
                    non_k9 + int(budget != 9),
                )
                previous = next_states.get(new_total)
                better = previous is None or candidate[0] < previous[0] - 1.0e-15
                if previous is not None and abs(candidate[0] - previous[0]) <= 1.0e-15:
                    better = candidate[1] < previous[1]
                if better:
                    next_states[new_total] = candidate
                    parent[new_total] = (total, budget)
        states = next_states
        parents.append(parent)

    if target_total not in states:
        raise ValueError(f"exact total K={target_total} is infeasible")
    chosen: list[int] = []
    total = target_total
    for parent in reversed(parents):
        previous_total, budget = parent[total]
        chosen.append(budget)
        total = previous_total
    chosen.reverse()
    schedule = dict(zip(slots, chosen, strict=True))
    uniform_objective = sum(normalized[slot][9] for slot in slots)
    return ExactBudgetSchedule(
        schedule=schedule,
        objective=states[target_total][0],
        uniform_k9_objective=uniform_objective,
        target_total_k=target_total,
    )
