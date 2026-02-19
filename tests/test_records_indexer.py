import importlib.util
import sys
import os


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_compute_bounding_box():
    module = load_module(os.path.join(
        'src', 'app', 'records_indexer.py'), 'records_indexer')
    coords = "0,0 10,0 10,5 0,5"
    bbox = module.compute_bounding_box(coords, padding=0)
    assert bbox == (0, 0, 10, 5)


def test_get_region_coordinates(tmp_path):
    module = load_module(os.path.join(
        'src', 'app', 'records_indexer.py'), 'records_indexer')
    xml_content = '''<?xml version="1.0"?>
<Page xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
  <TextRegion id="r1">
    <Coords points="0,0 10,0 10,5 0,5"/>
  </TextRegion>
</Page>
'''
    p = tmp_path / "test_page.xml"
    p.write_text(xml_content, encoding='utf-8')
    points = module.get_region_coordinates(str(p), 'r1')
    assert points == "0,0 10,0 10,5 0,5"
