"""Runtime initialization: thread caps and PyTensor compile directory.

Importing this package runs the module-level side effects in
``_thread_caps`` and ``_pytensor_compiledir`` (see those modules' docstrings
for the "MUST be first" ordering requirement).
"""
from vesta.runtime import _pytensor_compiledir, _thread_caps  # noqa: F401

__all__ = ["_pytensor_compiledir", "_thread_caps"]
