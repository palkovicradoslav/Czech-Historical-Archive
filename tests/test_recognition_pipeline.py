import importlib.util
import sys
import os


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_get_frequency_score():
    module = load_module(os.path.join(
        'src', 'recognition', 'pipeline.py'), 'pipeline')
    freq = {'a': 5, 'b': 2}
    score = module.get_frequency_score("A b.", freq)
    assert score == 7
