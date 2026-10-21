"""Ultimate Guitar helper: a shortlist of good Guitar Pro tabs, and sorting what you downloaded.

    python -m riffgen.tools.ultimate_guitar list https://www.ultimate-guitar.com/artist/meshuggah_3907
    python -m riffgen.tools.ultimate_guitar list --file bands.txt --min-rating 4.6 --best-only
    python -m riffgen.tools.ultimate_guitar sort --downloads ~/Downloads --dest "data/own/Compound Structure/influences"

What it does and does not do (on purpose):

* ``list`` reads the public **artist pages** (Guitar Pro filter) and keeps the community Guitar
  Pro tabs rated above ``--min-rating``. It waits a long random time between requests, caches
  pages for a week, checks robots.txt before every request, and stops at the first refusal
  (403/429/challenge page). It uses **no account and no cookies**.
* It does **not search** (robots.txt disallows ``/search.php``) and does **not download tabs**:
  ``tabs.ultimate-guitar.com/robots.txt`` disallows ``/tab/download`` for automated agents. You
  click "Download" yourself on each tab page; the shortlist page makes that quick.
* ``sort`` never touches the network: it matches the files in your Downloads folder to the
  shortlist, renames them ``Artist - Song.gp5`` / ``Artist - Song__v2.gp5`` (versions of one
  song form one song family, D9) and files them into the album folder.

Paid tabs (Official, Tab Pro) are never listed. Use the files for personal, non-commercial use
(Ultimate Guitar terms, §6.3).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from riffgen import REPO_DIR

USER_AGENT = "Mozilla/5.0 (compatible; RiffGen-shortlist/0.1; personal use, low rate)"
GP_EXTS = (".gp3", ".gp4", ".gp5", ".gpx", ".gp")
OLD_GP = (".gp3", ".gp4", ".gp5")
DEFAULT_OUT = REPO_DIR / "reports" / "ug_shortlist"


class Refused(Exception):
    """The site refused or challenged a request: stop, do not retry."""


# ---------------------------------------------------------------------------
# polite fetching

class PoliteFetcher:
    def __init__(self, delay: tuple[float, float], user_agent: str, cache_dir: Path, cache_days: float, seed: int = 0):
        self.delay = delay
        self.ua = user_agent
        self.cache_dir = cache_dir
        self.cache_s = cache_days * 86400
        self.rng = random.Random(seed or None)
        self.last = 0.0
        self.robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self.requests = 0

    def _wait(self) -> None:
        if self.last:
            pause = self.rng.uniform(*self.delay)
            left = self.last + pause - time.time()
            if left > 0:
                print(f"  waiting {left:.0f} s ...", flush=True)
                time.sleep(left)

    def _get(self, url: str) -> str:
        self._wait()
        req = urllib.request.Request(url, headers={"User-Agent": self.ua, "Accept-Language": "en"})
        self.requests += 1
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                body = r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            raise Refused(f"HTTP {e.code} for {url}") from e
        finally:
            self.last = time.time()
        return body

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        root = f"{parts.scheme}://{parts.netloc}"
        if root not in self.robots:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(root + "/robots.txt")
            try:
                rp.parse(self._get(root + "/robots.txt").splitlines())
            except Refused:
                rp.disallow_all = True
            self.robots[root] = rp
        return self.robots[root].can_fetch(self.ua, url)

    def page(self, url: str) -> str:
        key = self.cache_dir / (hashlib.sha1(url.encode()).hexdigest()[:20] + ".html")
        if key.is_file() and time.time() - key.stat().st_mtime < self.cache_s:
            return key.read_text(encoding="utf-8")
        if not self.allowed(url):
            raise Refused(f"robots.txt disallows {url}")
        body = self._get(url)
        if 'class="js-store"' not in body:
            raise Refused(f"unexpected page (challenge or new layout) for {url}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key.write_text(body, encoding="utf-8")
        return body


# ---------------------------------------------------------------------------
# parsing artist pages

def page_data(body: str) -> dict:
    m = re.search(r'class="js-store"\s+data-content="([^"]*)"', body)
    if not m:
        raise ValueError("no js-store data on the page")
    return json.loads(html.unescape(m.group(1)))["store"]["page"]["data"]


def artist_base(url: str) -> str:
    """https://xx.ultimate-guitar.com/artist/meshuggah_3907?page=2 -> https://www.ultimate-guitar.com/artist/meshuggah_3907"""
    parts = urlsplit(url.strip())
    if "/artist/" not in parts.path:
        raise ValueError(f"not an artist page: {url} (open the band on ultimate-guitar.com and copy the address)")
    return urlunsplit(("https", "www.ultimate-guitar.com", parts.path.rstrip("/"), "", ""))


def community_gp_tabs(data: dict) -> list[dict]:
    """Free community Guitar Pro tabs on one listing page (no Official / Tab Pro)."""
    out = []
    for key in ("album_tabs", "other_tabs", "feat_tabs", "top_tabs", "latest_tabs"):
        for t in data.get(key) or []:
            if t.get("type") == "Pro" and t.get("tab_access_type", "public") == "public" and t.get("tab_url"):
                out.append(t)
    return out


def collect_artist(fetcher: PoliteFetcher, url: str) -> list[dict]:
    base = artist_base(url) + "?filter=guitar_pro"
    data = page_data(fetcher.page(base))
    pages = [p["url"] for p in (data.get("pagination") or {}).get("pages", [])][1:]
    tabs = community_gp_tabs(data)
    for p in pages:
        full = "https://www.ultimate-guitar.com" + p if p.startswith("/") else p
        tabs += community_gp_tabs(page_data(fetcher.page(full)))
    seen, unique = set(), []
    for t in tabs:
        if t["id"] not in seen:
            seen.add(t["id"])
            unique.append(t)
    return unique


def shortlist(tabs: list[dict], min_rating: float, min_votes: int, best_only: bool) -> list[dict]:
    keep = [t for t in tabs if float(t.get("rating") or 0) > min_rating and int(t.get("votes") or 0) >= min_votes]
    if best_only:
        best: dict[tuple, dict] = {}
        for t in keep:
            k = (t["artist_name"].lower(), t["song_name"].lower())
            if k not in best or (t["rating"], t["votes"]) > (best[k]["rating"], best[k]["votes"]):
                best[k] = t
        keep = list(best.values())
    rows = [{"id": int(t["id"]), "artist": t["artist_name"], "song": t["song_name"], "version": int(t.get("version") or 1),
             "rating": round(float(t["rating"]), 2), "votes": int(t["votes"]), "tuning": t.get("tuning") or "",
             "tab_url": t["tab_url"], "status": "to download"} for t in keep]
    rows.sort(key=lambda r: (r["artist"].lower(), r["song"].lower(), r["version"]))
    return rows


# ---------------------------------------------------------------------------
# outputs

HTML_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Tab shortlist</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ --bg:#fafaf8; --fg:#1d1d1b; --muted:#6b6b66; --line:#e3e2dc; --done:#e8f1e4; --accent:#2f5d8a; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1b1b1a; --fg:#ecebe6; --muted:#a3a29b; --line:#33332f; --done:#24331f; --accent:#8fb6dd; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.45 system-ui, sans-serif; margin:0 auto; max-width:1000px; padding:24px 16px; }}
h1 {{ font-size:20px; margin:0 0 4px; }} p {{ color:var(--muted); margin:0 0 16px; }}
table {{ border-collapse:collapse; width:100%; }} th, td {{ border-bottom:1px solid var(--line); padding:6px 8px; text-align:left; }}
th {{ font-weight:600; color:var(--muted); font-size:13px; }} td.n {{ text-align:right; font-variant-numeric:tabular-nums; }}
tr.done td {{ background:var(--done); }} a {{ color:var(--accent); }}
.wrap {{ overflow-x:auto; }}
</style></head><body>
<h1>Guitar Pro shortlist</h1>
<p>{summary} Open a tab, click Download, tick it. Ticks are remembered in this browser. Then run
<code>python -m riffgen.tools.ultimate_guitar sort</code>.</p>
<div class="wrap"><table><thead><tr><th></th><th>Artist</th><th>Song</th><th>Ver</th><th>Rating</th><th>Votes</th><th>Tuning</th><th>Status</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<script>
const KEY = "ug-shortlist-done";
let done = {{}}; try {{ done = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{}}
for (const box of document.querySelectorAll("input[data-id]")) {{
  const row = box.closest("tr"); const id = box.dataset.id;
  if (done[id] || row.dataset.status === "sorted") {{ box.checked = true; row.classList.add("done"); }}
  box.addEventListener("change", () => {{
    done[id] = box.checked; row.classList.toggle("done", box.checked);
    try {{ localStorage.setItem(KEY, JSON.stringify(done)); }} catch (e) {{}}
  }});
}}
</script></body></html>
"""


def write_outputs(rows: list[dict], out: Path, summary: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "shortlist.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    with open(out / "shortlist.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "artist", "song", "version", "rating", "votes", "tuning", "tab_url",
                                          "status", "file"], restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    trs = []
    for r in rows:
        e = {k: html.escape(str(v)) for k, v in r.items()}
        trs.append(f'<tr data-status="{e["status"]}"><td><input type="checkbox" data-id="{e["id"]}"></td>'
                   f'<td>{e["artist"]}</td><td><a href="{e["tab_url"]}" target="_blank" rel="noopener">{e["song"]}</a></td>'
                   f'<td class="n">{e["version"]}</td><td class="n">{e["rating"]}</td><td class="n">{e["votes"]}</td>'
                   f'<td>{e["tuning"]}</td><td>{e["status"]}</td></tr>')
    (out / "shortlist.html").write_text(HTML_PAGE.format(summary=html.escape(summary), rows="\n".join(trs)),
                                        encoding="utf-8")


# ---------------------------------------------------------------------------
# sorting downloads

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def safe(s: str) -> str:
    return re.sub(r'[<>:"/\\|?*]+', "", s).strip().rstrip(".")


def target_name(row: dict, ext: str) -> str:
    suffix = "" if row["version"] <= 1 else f"__v{row['version']}"
    return f"{safe(row['artist'])} - {safe(row['song'])}{suffix}{ext}"


def gp_metadata(path: Path) -> tuple[str, str]:
    if path.suffix.lower() not in OLD_GP:
        return "", ""
    try:
        import guitarpro
        s = guitarpro.parse(str(path))
        return s.artist or "", s.title or ""
    except Exception:
        return "", ""


def match_file(path: Path, rows: list[dict]) -> tuple[dict | None, str]:
    stem = norm(path.stem)
    cands = [r for r in rows if norm(r["artist"]) in stem and norm(r["song"]) in stem]
    if not cands:
        artist, title = gp_metadata(path)
        if title:
            cands = [r for r in rows if norm(r["song"]) == norm(title)
                     and (not artist or norm(r["artist"]) in norm(artist) or norm(artist) in norm(r["artist"]))]
    if not cands:
        return None, "no shortlist entry matches"
    # prefer the longest song name ("Bleed" vs "Bleed (Live)")
    best_len = max(len(norm(r["song"])) for r in cands)
    cands = [r for r in cands if len(norm(r["song"])) == best_len]
    if len(cands) == 1:
        return cands[0], ""
    m = re.search(r"(?:ver(?:sion)?|v)\s*\.?\s*(\d+)|\((\d+)\)", path.stem, re.I)
    if m:
        v = int(m.group(1) or m.group(2))
        exact = [r for r in cands if r["version"] == v]
        if len(exact) == 1:
            return exact[0], ""
    left = [r for r in cands if r["status"] != "sorted"]
    if len(left) == 1:
        return left[0], ""
    return None, f"ambiguous: versions {sorted(r['version'] for r in cands)} of {cands[0]['song']}"


def cmd_sort(args) -> None:
    sl = Path(args.shortlist)
    rows = json.loads(sl.read_text(encoding="utf-8"))
    downloads = Path(args.downloads).expanduser()
    dest = Path(args.dest)
    if not dest.is_absolute():
        dest = REPO_DIR / dest
    files = sorted(p for p in downloads.iterdir() if p.is_file() and p.suffix.lower() in GP_EXTS)
    done, problems, export = [], [], []
    for p in files:
        row, why = match_file(p, rows)
        if row is None:
            problems.append(f"{p.name}: {why}")
            continue
        out = dest / safe(row["artist"]) / target_name(row, p.suffix.lower())
        if out.exists():
            problems.append(f"{p.name}: {out.name} already exists, skipped")
            row["status"] = "sorted"
            continue
        if not args.dry_run:
            out.parent.mkdir(parents=True, exist_ok=True)
            (shutil.move if args.move else shutil.copy2)(str(p), str(out))
            row["status"] = "sorted"
            row["file"] = str(out)
        done.append(f"{p.name} -> {out.relative_to(dest)}")
        if p.suffix.lower() not in OLD_GP:
            export.append(str(out))
    missing = [r for r in rows if r["status"] != "sorted"]
    if not args.dry_run:
        sl.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        write_outputs(rows, sl.parent, f"{len(rows) - len(missing)} of {len(rows)} sorted.")
    print(f"{'would sort' if args.dry_run else 'sorted'} {len(done)} files into {dest}")
    for line in done:
        print("  " + line)
    if export:
        print(f"\n{len(export)} files are Guitar Pro 6/7 (.gpx/.gp): open them in Guitar Pro 7 and export to GP5 (D6):")
        for e in export:
            print("  " + e)
    if problems:
        print(f"\n{len(problems)} files not sorted:")
        for line in problems:
            print("  " + line)
    print(f"\n{len(missing)} shortlisted tabs still to download (see {sl.parent / 'shortlist.html'})")


def cmd_list(args) -> None:
    urls = list(args.urls)
    if args.file:
        urls += [u.strip() for u in Path(args.file).read_text(encoding="utf-8").splitlines()
                 if u.strip() and not u.strip().startswith("#")]
    if not urls:
        sys.exit("give one or more artist page URLs (or --file)")
    if args.delay[0] < 10:
        sys.exit("--delay below 10 s is not allowed; keep the load on the site low")
    out = Path(args.out)
    fetcher = PoliteFetcher(tuple(args.delay), args.user_agent, out / "cache", args.cache_days)
    all_rows: list[dict] = []
    try:
        for url in urls:
            print(f"{artist_base(url)}")
            tabs = collect_artist(fetcher, url)
            rows = shortlist(tabs, args.min_rating, args.min_votes, args.best_only)
            print(f"  {len(tabs)} free community Guitar Pro tabs, {len(rows)} rated above {args.min_rating}")
            all_rows += rows
    except Refused as e:
        print(f"\nstopped: {e}\nThe site refused or challenged the request. Not retrying; try again another day "
              f"with a longer --delay, or open the pages in your browser.")
    old = {}
    if (out / "shortlist.json").is_file():
        old = {r["id"]: r for r in json.loads((out / "shortlist.json").read_text(encoding="utf-8"))}
    merged = {r["id"]: {**r, **{k: old[r["id"]][k] for k in ("status", "file") if k in old.get(r["id"], {})}}
              for r in list(old.values()) + all_rows}
    rows = sorted(merged.values(), key=lambda r: (r["artist"].lower(), r["song"].lower(), r["version"]))
    write_outputs(rows, out, f"{len(rows)} community Guitar Pro tabs rated above {args.min_rating} "
                             f"({sum(r['status'] == 'sorted' for r in rows)} sorted).")
    print(f"\n{len(rows)} tabs in {out / 'shortlist.html'} ({fetcher.requests} requests)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("list", help="shortlist good Guitar Pro tabs from artist pages")
    a.add_argument("urls", nargs="*", help="artist page URLs, e.g. https://www.ultimate-guitar.com/artist/meshuggah_3907")
    a.add_argument("--file", help="text file with one artist URL per line")
    a.add_argument("--min-rating", type=float, default=4.4, help="keep tabs rated above this (default 4.4)")
    a.add_argument("--min-votes", type=int, default=1, help="ignore ratings with fewer votes")
    a.add_argument("--best-only", action="store_true", help="only the best-rated version of each song")
    a.add_argument("--delay", type=float, nargs=2, default=[30.0, 60.0], metavar=("MIN", "MAX"),
                   help="random pause between requests in seconds (default 30-60, minimum 10)")
    a.add_argument("--cache-days", type=float, default=7.0)
    a.add_argument("--user-agent", default=USER_AGENT)
    a.add_argument("--out", default=str(DEFAULT_OUT))
    b = sub.add_parser("sort", help="file your downloaded tabs into an album folder")
    b.add_argument("--shortlist", default=str(DEFAULT_OUT / "shortlist.json"))
    b.add_argument("--downloads", default="~/Downloads")
    b.add_argument("--dest", required=True, help='e.g. "data/own/Compound Structure/influences"')
    b.add_argument("--move", action="store_true", help="move instead of copy")
    b.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    {"list": cmd_list, "sort": cmd_sort}[args.cmd](args)


if __name__ == "__main__":
    main()
