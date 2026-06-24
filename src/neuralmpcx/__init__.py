r"""Neural MPCX (neuralmpcx) is a Model Predictive Control toolkit with neural MPC support (CasADi-friendly).


==================== ========================================================
**Documentation**        https://github.com/hzdr/neural-mpcx/blob/main/README.md

**Source code**          https://github.com/hzdr/neural-mpcx

**Report issues**        https://github.com/hzdr/neural-mpcx/issues
==================== ========================================================
"""

__version__ = "3.0.0"

__all__ = ["Nlp", "Solution"]

from .core.solutions import Solution
from .nlps.nlp import Nlp
