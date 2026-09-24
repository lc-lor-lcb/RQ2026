"""Serve a built, single-threaded mjswan viewer inside Google Colab.

This file can also be pasted into a new Colab cell after building the viewer.
It does not rebuild the viewer or modify the trained policy.
"""
from functools import partial
from html import escape
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from urllib.request import urlopen


class ViewerHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "application/javascript",
        ".mjs": "application/javascript",
        ".wasm": "application/wasm",
    }

    def end_headers(self):
        # Each rebuild may replace config and policy files at the same URL.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format, *args):
        pass


def start_viewer_server(directory):
    directory = Path(directory).resolve()
    for required in ("index.html", "assets/config.json"):
        if not (directory / required).is_file():
            raise FileNotFoundError(
                f"ビューアーのファイルがありません: {directory / required}\n"
                "先にビューアーのビルドを完了してください。再学習は不要です。"
            )
    # Bind before displaying the iframe. Port 0 allocates a free port atomically.
    # A threaded server avoids one idle browser connection blocking all assets.
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(ViewerHandler, directory=str(directory))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with urlopen(f"http://127.0.0.1:{server.server_port}/index.html", timeout=10) as response:
            if response.status != 200:
                raise RuntimeError("ビューアーのHTTP起動確認に失敗しました。")
    except Exception:
        server.shutdown()
        server.server_close()
        raise
    return server


def launch_colab_viewer(directory="/tmp/rq_walk_dist", height=620):
    from google.colab import output
    from IPython.display import HTML, display

    server = start_viewer_server(directory)
    try:
        # Resolve explicitly: errors reach the cell instead of leaving an empty
        # async Javascript output. Use the authenticated Colab proxy, not a tunnel.
        url = output.eval_js(f"google.colab.kernel.proxyPort({server.server_port})")
        if not isinstance(url, str) or not url.startswith("https://"):
            raise RuntimeError("ColabのビューアーURLを取得できませんでした。")
        display(HTML(
            '<p>3Dビューアーを読み込みます。初回はWASMの転送に時間がかかります。</p>'
            f'<iframe src="{escape(url, quote=True)}" width="100%" '
            f'height="{int(height)}" style="border:0" '
            'title="RoboQuest 学習済みモデル" allow="autoplay; fullscreen"></iframe>'
        ))
    except Exception:
        server.shutdown()
        server.server_close()
        raise
    print("✅ 表示サーバー起動・Colab接続確認済み（3D描画完了の確認は画面で行ってください）")
    return server


if __name__ == "__main__":
    # Retain the handle in the notebook so rerunning this recovery cell releases
    # its previous server without touching the training environment.
    if "rq_recovery_server" in globals():
        rq_recovery_server.shutdown()
        rq_recovery_server.server_close()
    rq_recovery_server = launch_colab_viewer()
