from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"pattern missing in {path}: {old[:120]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# Adaptive-budget validation must not affect legacy tiny-frame unit tests.
replace_once(
    "coframe/config.py",
    '        if num_frames is not None and any(int(value) > num_frames for value in self.adaptive_k_values):\n'
    '            raise ValueError("adaptive_k_values exceed the latent frame count")\n'
    '        if self.force_boundaries and any(int(value) < 2 for value in self.adaptive_k_values) and (num_frames is None or num_frames > 1):\n'
    '            raise ValueError("adaptive_k_values must be >=2 when force_boundaries=True")\n',
    '        if self.method == "adaptive_k" and num_frames is not None and any(\n'
    '            int(value) > num_frames for value in self.adaptive_k_values\n'
    '        ):\n'
    '            raise ValueError("adaptive_k_values exceed the latent frame count")\n'
    '        if (\n'
    '            self.method == "adaptive_k"\n'
    '            and self.force_boundaries\n'
    '            and any(int(value) < 2 for value in self.adaptive_k_values)\n'
    '            and (num_frames is None or num_frames > 1)\n'
    '        ):\n'
    '            raise ValueError("adaptive_k_values must be >=2 when force_boundaries=True")\n',
)

# The conditional branch owns both remeshing and adaptive-budget updates; CFG
# unconditional replay remains read-only and reuses the recorded anchors.
replace_once(
    "coframe/wan/pipeline.py",
    '                    update_controller=config.method == "coframe",\n',
    '                    update_controller=config.method in {"coframe", "adaptive_k"},\n',
)

# Make the prompt-independent step/block baseline independently budget matched:
# quantile the per-cell calibration means rather than reusing content-adaptive thresholds.
replace_once(
    "scripts/analyze_stage1d_lagged.py",
    '        thresholds = fold["mean_defect_thresholds"]\n'
    '        by_key: dict[str, list[float]] = defaultdict(list)\n'
    '        for item in train:\n'
    '            by_key[str(item["key"])].append(float(item["mean_defect"]))\n'
    '        schedule = {\n'
    '            key: select_budget(mean(values), thresholds, budgets)\n'
    '            for key, values in sorted(by_key.items())\n'
    '        }\n'
    '        fold["step_block_schedule"] = schedule\n'
    '        fold["step_block_calibration_avg_k"] = mean(schedule.values()) if schedule else None\n',
    '        by_key: dict[str, list[float]] = defaultdict(list)\n'
    '        for item in train:\n'
    '            by_key[str(item["key"])].append(float(item["mean_defect"]))\n'
    '        key_means = {key: mean(values) for key, values in sorted(by_key.items())}\n'
    '        schedule_thresholds = [quantile(list(key_means.values()), q) for q in quantiles]\n'
    '        schedule = {\n'
    '            key: select_budget(value, schedule_thresholds, budgets)\n'
    '            for key, value in key_means.items()\n'
    '        }\n'
    '        fold["step_block_thresholds"] = schedule_thresholds\n'
    '        fold["step_block_schedule"] = schedule\n'
    '        fold["step_block_calibration_avg_k"] = mean(schedule.values()) if schedule else None\n',
)

# Keep the group-0 diagnostic causal too: do not overwrite the incoming previous
# step state until all probes for the current step have been emitted.
replace_once(
    "scripts/analyze_stage1d_lagged.py",
    '            if groups.get(max_group):\n'
    '                previous_step_last = groups[max_group]\n\n'
    '            for probe in event.get("probes", []):\n',
    '            incoming_previous_step_last = previous_step_last\n\n'
    '            for probe in event.get("probes", []):\n',
)
replace_once(
    "scripts/analyze_stage1d_lagged.py",
    '                source = groups.get(group - 1) if group > 0 else previous_step_last\n'
    '                if not source:\n'
    '                    continue\n'
    '                probe_records.append({\n',
    '                source = groups.get(group - 1) if group > 0 else incoming_previous_step_last\n'
    '                if not source:\n'
    '                    continue\n'
    '                probe_records.append({\n',
)
replace_once(
    "scripts/analyze_stage1d_lagged.py",
    '                    "propagation_h3": probe.get("propagated_relative_l2_h3"),\n'
    '                })\n'
    '    return probe_records, causal_observations\n',
    '                    "propagation_h3": probe.get("propagated_relative_l2_h3"),\n'
    '                })\n'
    '            if groups.get(max_group):\n'
    '                previous_step_last = groups[max_group]\n'
    '    return probe_records, causal_observations\n',
)

print("Stage-1d post-fixes applied")
