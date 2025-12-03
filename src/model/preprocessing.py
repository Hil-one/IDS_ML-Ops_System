"""
Preprocessing Utilities for IDS Classifier

This module contains reusable preprocessing functions that need to be
accessible when loading serialized models with joblib.

IMPORTANT: Functions here are referenced by serialized sklearn pipelines.
Do not rename or move these functions, as it will break model loading.
"""

import numpy as np
import os
import pandas as pd
import numpy as np
from pathlib import Path

def log2_transform(array):
    """
    Apply log2 transformation to byte features, setting 0 values to 0.

    This transformation is useful for byte-size features (src_bytes, dst_bytes)
    which can span many orders of magnitude. Log transformation helps normalize
    the distribution while preserving zero values.

    Args:
        array: NumPy array of byte values

    Returns:
        Transformed array with log2 applied to non-zero values
    """
    array = array.copy()
    array[array > 0] = np.log2(array[array > 0])
    return array
