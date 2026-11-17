"""pymspca: sparse PCA with multiple components.

Python port of the ``msPCA`` R package (Cory-Wright & Pauphilet). Cite:

    R. Cory-Wright and J. Pauphilet. Sparse PCA With Multiple Components.
    Operations Research 74(5):2650-2671, 2026. doi:10.1287/opre.2023.0598
"""

from ._api import MSPCAResult, MSPCAWarning, TPMResult, mspca, preprocess, tpm
from ._estimator import MultipleSparsePCA
from ._core import get_num_threads, set_num_threads

__all__ = ["mspca", "tpm", "MultipleSparsePCA", "MSPCAResult", "TPMResult", "MSPCAWarning", "preprocess", "set_num_threads", "get_num_threads"]
__version__ = "0.1.0"
