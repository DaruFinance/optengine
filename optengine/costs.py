"""optengine.costs — option execution cost model.

The real bid/ask is the dominant cost and the whole point of the study (gross!=net).
Fills cross the quoted spread by `spread_capture` (1.0 = full taker cross, <1 = working
the order). Commission is per contract per leg. $100 OPRA multiplier.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

MULT = 100.0  # OPRA contract multiplier


@dataclass
class OptionCostModel:
    commission: float = 0.65       # $/contract/leg
    exch_fee: float = 0.00         # $/contract/leg (exchange/clearing/regulatory)
    spread_capture: float = 1.0    # 1.0 = pay full half-spread to bid/ask; 0.5 = mid+quarter

    def fill(self, mid, bid, ask, side):
        """Per-contract fill price. side: +1 buy (toward ask), -1 sell (toward bid)."""
        half = np.maximum((ask - bid) / 2.0, 0.0)
        return mid + side * self.spread_capture * half

    def commission_cost(self, qty):
        return (self.commission + self.exch_fee) * abs(qty)
