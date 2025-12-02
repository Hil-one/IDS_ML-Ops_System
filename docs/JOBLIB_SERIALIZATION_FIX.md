# Joblib Serialization Fix

## Problem

When trying to load the trained model artifacts using `verify_artifacts.py` or any script that loads the preprocessor, the following error occurred:

```
✗ Error loading artifacts: Can't get attribute 'log2_transform' on <module '__main__' from 'verify_artifacts.py'>
```

## Root Cause

The `log2_transform()` function was defined in `train.py` and used in the sklearn `FunctionTransformer` within the preprocessing pipeline:

```python
# In train.py
def log2_transform(array):
    """Apply log2 transformation to byte features."""
    array = array.copy()
    array[array > 0] = np.log2(array[array > 0])
    return array

# Used in pipeline
bytes_size_transform = FunctionTransformer(
    func=log2_transform, ...
)
```

When joblib saves the preprocessor, it **serializes a reference** to the `log2_transform` function as `train.log2_transform`.

When other scripts try to load the preprocessor:
- They import `joblib.load()`
- Joblib tries to find `train.log2_transform`
- But the function is not available in their namespace
- ❌ Loading fails

This is a classic **custom function serialization problem** with sklearn pipelines and joblib.

## Solution

### Step 1: Extract Function to Shared Module

Created `src/model/preprocessing.py` containing shared preprocessing utilities:

```python
# src/model/preprocessing.py
import numpy as np

def log2_transform(array):
    """Apply log2 transformation to byte features, setting 0 values to 0."""
    array = array.copy()
    array[array > 0] = np.log2(array[array > 0])
    return array
```

### Step 2: Update train.py

Modified `train.py` to import from the shared module instead of defining locally:

```python
# Import preprocessing utilities
from preprocessing import log2_transform

# Removed the local definition
# def log2_transform(array): ...  <-- REMOVED
```

Also removed unused `import numpy as np` since it's no longer needed.

### Step 3: Update All Loading Scripts

Added import to all scripts that load the preprocessor:

**inference.py:**
```python
from preprocessing import log2_transform  # noqa: F401
```

**verify_artifacts.py:**
```python
from src.model.preprocessing import log2_transform  # noqa: F401
```

**evaluate.py:**
```python
from src.model.preprocessing import log2_transform  # noqa: F401
```

The `# noqa: F401` comment tells linters to ignore "imported but unused" warnings. The import is necessary for joblib to find the function when deserializing.

### Step 4: Update Package __init__.py

Created `src/model/__init__.py` to properly export the function:

```python
from .preprocessing import log2_transform
from .inference import IDSClassifier

__all__ = ['IDSClassifier', 'log2_transform']
```

## Why This Works

1. **Shared Location**: `log2_transform` is now in a dedicated module that all scripts can import
2. **Consistent Namespace**: When joblib saves, it references `preprocessing.log2_transform`
3. **Available on Load**: When loading, the function is imported and available in the namespace
4. **Single Source of Truth**: The function is defined once and imported everywhere

## Files Modified

1. **Created:**
   - `src/model/preprocessing.py` - Shared preprocessing utilities
   - `src/model/__init__.py` - Package initialization

2. **Modified:**
   - `src/model/train.py` - Import log2_transform instead of defining it
   - `src/model/inference.py` - Import log2_transform for loading
   - `src/model/verify_artifacts.py` - Import log2_transform for loading
   - `src/model/evaluate.py` - Import log2_transform for loading

## Verification

After these changes, you can verify the fix works:

```bash
# 1. Train a model (regenerate artifacts with new reference)
python src/model/train.py

# 2. Verify artifacts load correctly
python src/model/verify_artifacts.py
```

Expected output:
```
✓ VERIFICATION SUCCESSFUL
All model artifacts are present and working correctly.
```

## Best Practices for Future

### ✅ DO:
- Define custom transformation functions in shared modules
- Import them consistently across all scripts
- Use standard sklearn transformers when possible

### ❌ DON'T:
- Define custom functions in the training script
- Use lambda functions in sklearn pipelines (they can't be serialized)
- Define functions inside other functions

### Example of Safe Custom Transformer:

```python
# preprocessing.py
def safe_transform(X):
    """Well-documented, reusable transformation."""
    return X

# train.py
from preprocessing import safe_transform

transformer = FunctionTransformer(func=safe_transform)
```

## Related Issues

This issue could occur with any custom function passed to:
- `FunctionTransformer`
- Custom sklearn estimators
- Custom scoring functions in `cross_validate`

Always ensure custom callables are:
1. Defined in a shared module
2. Imported where artifacts are saved
3. Imported where artifacts are loaded

## References

- [Sklearn FunctionTransformer Docs](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)
- [Joblib Persistence Docs](https://joblib.readthedocs.io/en/latest/persistence.html)
- [Python Pickle Module Warning](https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled)
