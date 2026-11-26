"""Opt-in scratch preflight at optimizer boundaries (not a CUDA fit guarantee).

Only *unallocated* persistent state and outstanding GPU commitments are
subtracted from the driver-free snapshot. Existing tensor allocations and the
allocator's reserved-but-unused bytes are not added back. Groups execute
sequentially, so their scratch demands are maxima rather than sums.
"""
from dataclasses import dataclass
import math

MIB = 2**20

class ScratchBudgetError(MemoryError):
    """Policy refusal, not an allocation failure or system RAM exhaustion."""

@dataclass(frozen=True)
class ScratchGroup:
    key: int
    count: int
    bytes_each: int
    ceiling_bytes: int

@dataclass(frozen=True)
class BudgetPlan:
    free_bytes: int
    pending_state_bytes: int
    outstanding_bytes: int
    reserve_bytes: int
    safe_bytes: int
    selected_bytes: int
    chunks: tuple
    caps: tuple

class MemoryBudget:
    def __init__(self, reserve_mib=768, growth_steps=3):
        if not math.isfinite(reserve_mib) or reserve_mib<0 or not isinstance(growth_steps,int) or growth_steps<1:
            raise ValueError('invalid memory budget policy')
        self.reserve_bytes=int(reserve_mib*MIB)
        self.growth_steps=growth_steps
        self.current=None
        self._growth=[]
        self.last_plan=None

    def plan(self, free_bytes, groups, *, pending_state_bytes=0, outstanding_bytes=0):
        groups=tuple(groups)
        if not groups or min(free_bytes,pending_state_bytes,outstanding_bytes)<0:
            raise ValueError('invalid memory snapshot/groups')
        if any(min(g.count,g.bytes_each,g.ceiling_bytes)<1 for g in groups):
            raise ValueError('automatic memory budgeting requires positive finite chunk ceilings')
        available=max(0,int(free_bytes)-int(pending_state_bytes)-int(outstanding_bytes)-self.reserve_bytes)
        safe=min(available,max(g.ceiling_bytes for g in groups))
        if self.current is None or safe<=self.current:
            self.current=safe;self._growth=[]
        else:
            self._growth.append(safe)
            if len(self._growth)>=self.growth_steps:
                self.current=min(self._growth);self._growth=[]
        selected=self.current
        caps=tuple(min(selected,g.ceiling_bytes) for g in groups)
        if any(g.bytes_each>cap for g,cap in zip(groups,caps)):
            self.last_plan=None
            raise ScratchBudgetError('memory preflight: a single matrix/pair exceeds the safe estimated scratch allowance; no update started')
        chunks=tuple(min(g.count,cap//g.bytes_each) for g,cap in zip(groups,caps))
        self.last_plan=BudgetPlan(int(free_bytes),int(pending_state_bytes),int(outstanding_bytes),self.reserve_bytes,safe,selected,chunks,caps)
        return self.last_plan
