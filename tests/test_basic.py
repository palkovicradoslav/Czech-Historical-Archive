import sys
import os

# Ensure src is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def test_utils_import():
    import utils as u
    assert hasattr(u, 'setup_logger')


def test_app_import():
    from app import app as flask_app
    assert flask_app is not None


def test_records_indexer_import():
    from app.records_indexer import VitalRecordsIndexer
    idx = VitalRecordsIndexer(index_dir=os.path.join(
        SRC, 'app', 'vital_records_index'))
    assert hasattr(idx, 'build_index')
