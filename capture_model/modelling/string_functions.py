import re
import pandas as pd

def __convert_to_string(string_like):
    if isinstance(string_like, float):
        if pd.isnull(string_like):
            return ''
        else:
            return str(string_like)
    return str(string_like)

def lower(string_like):
    string = __convert_to_string(string_like)
    return string.lower()

def upper(string_like):
    string = __convert_to_string(string_like)
    return string.upper()

def replace(string_like, regex_pattern, replacement):
    regex = re.compile(regex_pattern)
    return regex.sub(replacement, __convert_to_string(string_like))

def strip(string_like):
    string = __convert_to_string(string_like)
    return string.strip()

def lstrip(string_like):
    string = __convert_to_string(string_like)
    return string.lstrip()

def rstrip(string_like):
    string = __convert_to_string(string_like)
    return string.rstrip()

def split(string_like, sep=None):
    string = __convert_to_string(string_like)
    return string.split(sep)

def ljust(string_like, width, fillchar=' '):  # ✅ ADDED FUNCTION
    string = __convert_to_string(string_like)
    return string.ljust(width, fillchar)
