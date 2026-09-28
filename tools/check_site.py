"""Check every link and asset referenced by site/index.html (local files exist; external URLs answer).

Usage: python tools/check_site.py [--offline]
"""

from __future__ import annotations

import re
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"


class Refs(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.refs: list[tuple[str, str]] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        if tag == "link" and a.get("rel") in ("preconnect", "dns-prefetch"):
            return  # connection hints name an origin, not a document
        for k in ("href", "src", "poster"):
            if a.get(k):
                self.refs.append((tag, a[k]))


def main() -> int:
    offline = "--offline" in sys.argv
    html = (SITE / "index.html").read_text(encoding="utf-8")
    p = Refs()
    p.feed(html)
    css_urls = []
    for css in (SITE / "assets" / "css").glob("*.css"):
        css_urls += [(f"css:{css.name}", u) for u in re.findall(r"url\(['\"]?([^)'\"]+)", css.read_text(encoding="utf-8")) if not u.startswith("data:")]
    bad, seen = [], set()
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass
    for tag, ref in p.refs + css_urls:
        if ref in seen:
            continue
        seen.add(ref)
        u = urlparse(ref)
        if ref.startswith("#"):
            ok = ref[1:] in p.ids
            print(("OK  " if ok else "BAD ") + ref)
            bad += [] if ok else [ref]
        elif u.scheme in ("http", "https"):
            if offline:
                print("SKIP " + ref)
                continue
            status = None
            for method in ("HEAD", "GET"):
                try:
                    req = urllib.request.Request(ref, method=method, headers={"User-Agent": "Mozilla/5.0 (link check)"})
                    with urllib.request.urlopen(req, timeout=20) as r:
                        status = r.status
                    break
                except Exception as e:  # noqa: BLE001
                    status = getattr(e, "code", type(e).__name__)
            ok = status == 200 or (isinstance(status, int) and status in (999,))  # LinkedIn answers bots with 999
            print(f"{'OK  ' if status == 200 else 'WARN' if ok else 'BAD '} {status} {ref}")
            if not ok:
                bad.append(ref)
        elif u.scheme in ("mailto",):
            print("OK   " + ref)
        else:
            target = (SITE / u.path).resolve()  # not urljoin: it drops a leading "../"
            ok = target.exists()
            print(("OK  " if ok else "BAD ") + ref)
            bad += [] if ok else [ref]
    print(f"\n{len(seen)} references, {len(bad)} broken")
    for b in bad:
        print("  broken:", b)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
