"""
Test imports
"""

try:
    import scikit_ext
    _top_import_error = None
except Exception as e:
    _top_import_error = e

try:
    from scikit_ext import *
    _star_import_error = None
except Exception as e:
    _star_import_error = e

try:
    from scikit_ext import estimators, scorers
    _module_import_error = None
except Exception as e:
    _module_import_error = e

def test_top():
    assert _top_import_error == None

def test_star():
    assert _star_import_error == None

def test_module():
    assert _module_import_error == None
