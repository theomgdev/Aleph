"""Aleph — a language model that is one matrix.

Forked from OdyssNet, decoupled: the recurrence is read one token per step,
the token interface is frozen rather than learned, and depth is bought inside
the step instead of by echoing.
"""

__version__ = "0.1.0"

from .core.attention import TemporalAttention
from .core.codebook import Codebook
from .core.core import AlephCore, quantize_ste
from .core.plasticity import Plasticity
from .core.ponder import Ponder
from .train.muon import MuonAdamW
from .train.trainer import Trainer

__all__ = [
    "Codebook", "AlephCore", "TemporalAttention", "Plasticity", "Ponder",
    "quantize_ste", "Trainer", "MuonAdamW", "__version__",
]
