"""
NumPy 2.0 compatibility shim for ChromaDB. Must be imported before any chromadb imports.
"""
import numpy as np

# NumPy 2.0 removed these aliases; restore them for chromadb compatibility
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'complex_'):
    np.complex_ = np.complex128
if not hasattr(np, 'unicode_'):
    np.unicode_ = np.str_
if not hasattr(np, 'int_'):
    np.int_ = np.int64
if not hasattr(np, 'bool_'):
    np.bool_ = np.bool_
if not hasattr(np, 'object_'):
    np.object_ = np.object_
if not hasattr(np, 'str_'):
    np.str_ = np.str_
