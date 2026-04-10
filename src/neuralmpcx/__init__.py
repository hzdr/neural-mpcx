r"""Neural MPCX (neuralmpcx) is a Model Predictive Control toolkit with neural MPC support (CasADi-friendly).


==================== ========================================================
**Documentation**        https://neuralmpcx.readthedocs.io/en/stable/

**Download**             https://pypi.python.org/pypi/neuralmpcx/

**Source code**          https://github.com/EnioLopes/neural-mpcx/

**Report issues**        https://github.com/EnioLopes/neural-mpcx/issues/
==================== ========================================================
"""

__version__ = "1.1.0"

__all__ = ["Nlp", "Solution"]

from .core.solutions import Solution
from .nlps.nlp import Nlp
