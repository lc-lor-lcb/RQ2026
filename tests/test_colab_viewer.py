import importlib.util
from pathlib import Path
import socket
import sys
import types
from urllib.request import urlopen

import pytest

spec = importlib.util.spec_from_file_location(
    "launcher", Path(__file__).resolve().parents[1] / "scripts/launch_colab_viewer.py"
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.fixture
def viewer_dir(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<h1>viewer</h1>")
    (tmp_path / "assets/config.json").write_text("{}")
    (tmp_path / "assets/test.wasm").write_bytes(b"\x00asm")
    return tmp_path


def test_requires_completed_build(tmp_path):
    with pytest.raises(FileNotFoundError, match="再学習は不要"):
        launcher.start_viewer_server(tmp_path)


def test_serves_wasm_despite_idle_browser_connection(viewer_dir):
    server = launcher.start_viewer_server(viewer_dir)
    idle = socket.create_connection(server.server_address, timeout=2)
    try:
        with urlopen(f"http://127.0.0.1:{server.server_port}/assets/test.wasm", timeout=2) as r:
            assert r.read() == b"\x00asm"
            assert r.headers["Content-Type"] == "application/wasm"
            assert r.headers["Cache-Control"] == "no-store"
    finally:
        idle.close()
        server.shutdown()
        server.server_close()


def test_explicit_colab_proxy_embedded(viewer_dir, monkeypatch):
    import IPython.display
    displays = []
    monkeypatch.setattr(IPython.display, "display", displays.append)
    output = types.SimpleNamespace(eval_js=lambda code: "https://example.test/?a=1&b=2")
    monkeypatch.setitem(sys.modules, "google.colab", types.SimpleNamespace(output=output))
    server = launcher.launch_colab_viewer(viewer_dir)
    try:
        assert 'src="https://example.test/?a=1&amp;b=2"' in displays[0].data
        assert 'height="620"' in displays[0].data
    finally:
        server.shutdown()
        server.server_close()


def test_proxy_error_is_not_silently_hidden(viewer_dir, monkeypatch):
    def fail(code):
        raise RuntimeError("Colab proxy unavailable")
    monkeypatch.setitem(sys.modules, "google.colab", types.SimpleNamespace(
        output=types.SimpleNamespace(eval_js=fail)
    ))
    with pytest.raises(RuntimeError, match="Colab proxy unavailable"):
        launcher.launch_colab_viewer(viewer_dir)
