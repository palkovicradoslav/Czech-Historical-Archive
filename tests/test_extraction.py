import importlib.util
import sys
import os
import unicodedata


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_normalize_json():
    module = load_module(os.path.join('src', 'extraction',
                         'structured_records_extraction.py'), 'extraction_mod')
    data = {
        'a': 'á',
        'b': ['č', {'c': 'ď'}]
    }
    normalized = module.normalize_json(data)
    assert normalized['a'] == unicodedata.normalize('NFD', 'á')
    assert normalized['b'][0] == unicodedata.normalize('NFD', 'č')
    assert normalized['b'][1]['c'] == unicodedata.normalize('NFD', 'ď')
