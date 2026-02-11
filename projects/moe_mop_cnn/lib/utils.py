"""
Utility Functions
=================

Progress bar wrappers and common data preparation helpers.
"""

import pandas as pd
from tqdm import tqdm


def apply_with_progress(series: pd.Series, func, desc: str = "", **kwargs) -> pd.Series:
    """Apply `func` to every element of `series` with a tqdm progress bar.
    
    Extra keyword arguments are passed through to `func`.
    
    Args:
        series: Pandas Series to process
        func: Function to apply to each element
        desc: Description for the progress bar
        **kwargs: Additional arguments passed to func
        
    Returns:
        pd.Series with results
    """
    results = [None] * len(series)
    for i, value in enumerate(tqdm(series, desc=desc, leave=True)):
        if kwargs:
            results[i] = func(value, **kwargs)
        else:
            results[i] = func(value)
    return pd.Series(results, index=series.index)
