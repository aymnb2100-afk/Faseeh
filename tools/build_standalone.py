"""Build standalone/Faseeh.html: one file that runs offline by double-clicking it.

index.html loads its data with fetch("data/...json"), which browsers block for pages opened
from disk (file://). This script copies index.html, embeds every data file (and the
articulation drawings as data: URIs) in a <script type="application/json">, and adds a small
fetch shim that serves those files, so the application code itself is unchanged.

Usage (from the repository root):
    python3 tools/build_standalone.py
"""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_FILES = ["quran", "rules", "articulation", "ai-demo-response", "reciters"]

SHIM = """<script>
/* Standalone build: the data files are embedded here and served to fetch(), so the app code is unchanged. */
(function () {
  const FILES = JSON.parse(document.getElementById("faseeh-data").textContent);
  const realFetch = window.fetch ? window.fetch.bind(window) : null;
  window.fetch = function (url, opts) {
    const key = String(url).replace(/^\\.\\//, "").replace(/\\?.*$/, "");
    if (Object.prototype.hasOwnProperty.call(FILES, key)) {
      return Promise.resolve(new Response(JSON.stringify(FILES[key]), { headers: { "Content-Type": "application/json" } }));
    }
    return realFetch(url, opts);
  };
})();
</script>
"""


def embed_images(value):
    """Replace "images/x.svg" paths inside the data with data: URIs."""
    if isinstance(value, dict):
        return {k: embed_images(v) for k, v in value.items()}
    if isinstance(value, list):
        return [embed_images(v) for v in value]
    if isinstance(value, str) and value.startswith("images/") and (ROOT / value).exists():
        return "data:image/svg+xml;base64," + base64.b64encode((ROOT / value).read_bytes()).decode()
    return value


def main():
    page = (ROOT / "index.html").read_text(encoding="utf-8")
    files = {f"data/{n}.json": json.loads((ROOT / "data" / f"{n}.json").read_text(encoding="utf-8")) for n in DATA_FILES}
    payload = json.dumps(embed_images(files), ensure_ascii=False).replace("</", "<\\/")
    block = f'<script type="application/json" id="faseeh-data">{payload}</script>\n{SHIM}'
    # The data and the shim must run before the app script, which is the first <script> in <body>.
    body = page.index("<body>")
    first_script = page.index("<script>", body)
    out = page[:first_script] + block + page[first_script:]
    target = ROOT / "standalone" / "Faseeh.html"
    target.parent.mkdir(exist_ok=True)
    target.write_text(out, encoding="utf-8")
    print(f"Wrote {target.relative_to(ROOT)} ({len(out.encode('utf-8')) // 1024} KB)")


if __name__ == "__main__":
    main()
