"""Matrix multiplication through the BLAS the machine actually has.

thinc ships its own build of BLIS and routes every ``gemm`` through it. That build
uses a generic kernel: measured on an AMD EPYC 9454P, which has AVX-512, it reached
**36 GFLOPS** while numpy — the same process, the same matrices — reached **2378**
through scipy-openblas. Sixty-five times. The same gap appears on Apple silicon, where
numpy goes through Accelerate.

So the arithmetic is not slow because Python is slow, and not because the CPU lacks
SIMD. It is slow because one dependency does not use it. Overriding the one method
that matters puts the work back on the platform's BLAS: measured end to end on the
Russian pipeline, 5.63 s to 4.19 s for 183 KB, with the findings identical.

``PII_SHIELD_BLAS=blis`` puts it back the way thinc intended, for a machine where that
turns out to be the faster of the two.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_DISABLE_ENV = "PII_SHIELD_BLAS"


def _build_ops_class():
    """The ops class, or None when thinc is absent or the override is switched off."""
    if os.environ.get(_DISABLE_ENV, "").lower() == "blis":
        return None
    try:
        import numpy as np
        from thinc.api import NumpyOps
    except ImportError:  # pragma: no cover - thinc arrives with the ner extra
        return None

    class PlatformBlasOps(NumpyOps):
        """thinc's ops, with ``gemm`` handed to numpy instead of the bundled BLIS."""

        name = "numpy"

        def gemm(self, x, y, *, out=None, trans1=False, trans2=False):
            left = x.T if trans1 else x
            right = y.T if trans2 else y
            result = left @ right
            if out is not None:
                out[...] = result
                return out
            return np.ascontiguousarray(result, dtype=x.dtype)

    return PlatformBlasOps


def use_platform_blas(pipeline) -> int:
    """Point every model in a spaCy pipeline at the platform's BLAS.

    Returns how many model nodes were switched, which is zero when the override is
    unavailable or switched off. Failures are swallowed on purpose: a pipeline that
    cannot be retargeted still works, only slower, and a shield that refuses to start
    over an optimization would be a poor trade.
    """
    ops_class = _build_ops_class()
    if ops_class is None:
        return 0
    try:
        ops = ops_class()
        switched = 0
        for _, component in getattr(pipeline, "pipeline", ()):
            model = getattr(component, "model", None)
            if model is None:
                continue
            for node in model.walk():
                node.ops = ops
                switched += 1
        return switched
    except Exception:  # pragma: no cover - never fatal
        logger.debug("pii-shield: could not switch to the platform BLAS", exc_info=True)
        return 0
