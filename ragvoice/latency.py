from __future__ import annotations

import math
from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class StageTimer:
    started_at: float = field(default_factory=perf_counter)

    def stop_ms(self) -> float:
        return round((perf_counter() - self.started_at) * 1000, 3)


def percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil((pct / 100) * len(ordered)) - 1)
    return round(ordered[index], 3)
