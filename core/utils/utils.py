import json
import re
import ast
import logging
import math
import pandas as pd
from typing import Any, Union, Dict, Set, Tuple


def extract_json_block(raw_text):
    """
    Extracts JSON block from raw_text.
    """
    logging.error("[extract_json_block] Not implemented")
    raise NotImplementedError("Not implemented")

def is_none_value(value: Any) -> bool:
    """
    Check if a value is considered as None/null/empty.
    
    This function handles various representations of None/null values:
    - Python None
    - String representations: "None", "null", "NULL", "nan", "NaN", "NAN"
    - Empty strings: "", "   " (whitespace only)
    - Numeric NaN values
    - Pandas NA values
    
    Args:
        value: Any value to check
        
    Returns:
        bool: True if the value is considered None/null/empty, False otherwise
        
    Examples:
        >>> is_none_value(None)
        True
        >>> is_none_value("None")
        True
        >>> is_none_value("NaN")
        True
        >>> is_none_value("")
        True
        >>> is_none_value("   ")
        True
        >>> is_none_value(float('nan'))
        True
        >>> is_none_value("hello")
        False
        >>> is_none_value(0)
        False
    """
    # Check for Python None
    if value is None:
        return True
    
    # Check for string representations
    if isinstance(value, str):
        # Remove leading/trailing whitespace and convert to lowercase
        cleaned_value = value.strip().lower()
        
        # Check for empty string after stripping
        if not cleaned_value:
            return True
            
        # Check for common None/null string representations
        none_strings = {
            'none', 'null', 'nan', 'na', 'n/a', 
            'nil', 'undefined', 'empty', '<null>',
            '<none>', '<na>', '<nan>'
        }
        if cleaned_value in none_strings:
            return True
    
    # Check for numeric NaN values
    try:
        if isinstance(value, (int, float)) and math.isnan(value):
            return True
    except (TypeError, ValueError):
        pass
    
    return False


def is_null(value: Any) -> bool:
    """Check if a value should be considered null/empty for evaluation purposes.
    
    This function is used across evaluation modules to consistently handle
    null/empty values in predictions and ground truth data.
    
    Args:
        value: Value to check
        
    Returns:
        True if value is considered null, False otherwise
    """
    null_words = {"", "null", "none", "nan", "undisclosed", "unspecified", 
                  "unknown", "n/a", "na", "n.a.", "na.", "n/a.", "not available", 
                  "not applicable"}
    
    if value is None:
        return True
        
    str_value = str(value).strip().lower()
    return str_value in null_words


def is_match(value1: Any, value2: Any) -> bool:
    """Check if two values match. """
    return value1 == value2


def compute_tp_fp_fn(golden: set, output: set):
    golden = set(golden)
    output = set(output)
    _tp = len(golden & output)    # Correctly extracted attributes
    _fp = len(output - golden)    # Incorrectly extracted attributes
    _fn = len(golden - output)    # Missing attributes
    return _tp, _fp, _fn
