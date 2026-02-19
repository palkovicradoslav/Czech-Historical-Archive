import importlib.util
import sys
import os


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_generate_surname_versions_simple():
    module = load_module(os.path.join(
        'src', 'genealogy', 'genealogy.py'), 'genealogy')
    versions = module.generate_surname_versions('Novakova')
    assert 'novak' in versions
