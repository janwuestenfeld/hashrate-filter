"""hashrate-filter: state-space correction of the daily Bitcoin network hashrate.

Estimators
----------
pure_filter      one-sided estimate using the day's own blocks; the series
                 used in the paper. two_sided=True adds the two-sided
                 version (RTS smoother), which the paper uses only as a
                 benchmark.
anchored_filter  the extended filter: the pure filter plus the difficulty
                 as an extra observation on reset days, de-lagged by the
                 filter's causal slope. Also exported as extended_filter.

Evidence
--------
daily_evidence builds the daily arrival-time weight pi_t from block header
timestamps.

Constants (engine.py): the observation variance is protocol-fixed, q_jump
is calibrated from the dispersion on days with pi_t > 0.7, and q_quiet is
the single fitted parameter (maximum likelihood on the filter's one-step
prediction errors).
"""

from .engine import (
    DELAG_LAG,
    DELAG_PROJ,
    P0,
    PROTOCOL_OFFSET,
    Q_JUMP,
    R1,
    R2,
    FilterResult,
    fit_q_quiet,
    run_filter,
)
from .evidence import daily_evidence
from .variants import anchored_filter, extended_filter, pure_filter

__version__ = "0.1.0"

__all__ = [
    "pure_filter",
    "anchored_filter",
    "extended_filter",
    "daily_evidence",
    "run_filter",
    "fit_q_quiet",
    "FilterResult",
    "R1",
    "R2",
    "Q_JUMP",
    "P0",
    "PROTOCOL_OFFSET",
    "DELAG_LAG",
    "DELAG_PROJ",
]
