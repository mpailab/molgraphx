"""MolgraphX package init.

Keep imports lightweight to avoid pulling optional heavy dependencies
at import time (e.g., torch_geometric). Import utilities directly from
their submodules when needed.
"""

__all__ = ["__version__"]

# Single source of truth for the package version
__version__ = "0.1.0"
