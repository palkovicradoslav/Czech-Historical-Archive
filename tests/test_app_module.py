import importlib.util
import sys
import os


def load_module(path, name):
    # resolve import issues
    src_root = os.path.abspath(os.path.join(os.getcwd(), 'src'))
    src_app = os.path.abspath(os.path.join(src_root, 'app'))
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    if src_app not in sys.path:
        sys.path.insert(0, src_app)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path[:] = list(sys.path)


def test_extract_text_lines_from_pagexml(tmp_path):
    module = load_module(os.path.join('src', 'app', 'app.py'), 'app_module')
    xml_content = '''<?xml version="1.0"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
  <Page>
    <TextRegion id="r1">
      <TextLine id="l1">
        <Coords points="0,0 1,1"/>
        <TextEquiv><Unicode>Hello</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
'''
    p = tmp_path / "page.xml"
    p.write_text(xml_content, encoding='utf-8')

    lines = module.extract_text_lines_from_pagexml(str(p), 'r1')
    assert isinstance(lines, list)
    assert len(lines) == 1
    assert lines[0]['id'] == 'l1'
    assert lines[0]['text'] == 'Hello'
    assert lines[0]['points'] == '0,0 1,1'
