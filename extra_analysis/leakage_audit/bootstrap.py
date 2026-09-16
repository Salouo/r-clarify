from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Iterable


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct
    lower = int(idx)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = idx - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_rate_ci_by_instance(
    rows: Iterable[dict],
    *,
    predicate: Callable[[dict], bool],
    samples: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    by_instance: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_instance[str(row.get("instance_id", row.get("record_id", "")))].append(row)
    instance_ids = list(by_instance)
    if not instance_ids:
        return (0.0, 0.0)

    rng = random.Random(seed)
    rates: list[float] = []
    for _ in range(samples):
        total = 0
        hits = 0
        for instance_id in (rng.choice(instance_ids) for _ in instance_ids):
            instance_rows = by_instance[instance_id]
            total += len(instance_rows)
            hits += sum(1 for row in instance_rows if predicate(row))
        rates.append(hits / total if total else 0.0)
    return (percentile(rates, 0.025), percentile(rates, 0.975))

