#!/usr/bin/env python
"""Shortlist the most-played tabs of a genre from Ultimate Guitar's explore pages into data/list.csv.

    python data_explore.py "https://www.ultimate-guitar.com/explore?genres[]=45&order=hitstotal_desc&type[]=Pro" --songs 500

``data_dl.py list`` reads artist pages, which is the wrong shape for "the N most famous songs of a genre":
that question is answered by /explore, whose rows are already ordered by total hits and are all free
community Guitar Pro tabs. This adds only the paging and the row-to-list conversion, and reuses
data_dl.py's PoliteFetcher for everything that touches the network - robots.txt checked before every
request, 7.5-15 s between them, pages cached for a week, and a stop at the first refusal.

Of each song it keeps the single most-played version (the first row the explore order gives), so
``--songs 500`` means 500 distinct songs and not 500 tabs of which some are the same song. Nothing is
downloaded here: the rows land in data/list.csv with status "to download", for data_dl.py to fetch.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_dl as dl  # noqa: E402


def page_url(url: str, page: int) -> str:
    """The same explore URL at page N, keeping every filter (genres[], type[], order) as it is."""
    parts = urlsplit(url)
    q = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != "page"]
    if page > 1:
        q.append(("page", str(page)))
    return urlunsplit((parts.scheme or "https", parts.netloc, parts.path, urlencode(q), ""))


def collect(fetcher: dl.PoliteFetcher, url: str, songs: int, max_pages: int,
            exclude: tuple[str, ...] = ()) -> list[dict]:
    """Rows for the `songs` most-played distinct songs, in the order explore lists them.

    ``exclude`` drops artists by name: the genre tags on UG are crowd-sourced and a genre listing carries
    a few plainly wrong ones (genre 45, hip hop, lists a French black metal band and a post-hardcore one),
    which a "most famous hip hop songs" shortlist should not spend its slots on.
    """
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    dropped: dict[str, int] = {}
    for page in range(1, max_pages + 1):
        data = dl.page_data(fetcher.page(page_url(url, page)))
        tabs = data["data"]["tabs"]
        total, pages = data.get("totalResults"), (data.get("pagination") or {}).get("pages")
        if page == 1:
            print(f"  {total} tabs match, {pages} pages reachable ({len(tabs)} per page)")
        kept = 0
        for t in tabs:
            # explore with type[]=Pro already returns only free community Guitar Pro tabs, but check anyway:
            # the filter lives in the URL, and a mistyped one would otherwise shortlist paid tabs
            if t.get("type") != "Pro" or t.get("tab_access_type", "public") != "public" or not t.get("tab_url"):
                continue
            key = (str(t.get("artist_name", "")).strip().lower(), str(t.get("song_name", "")).strip().lower())
            if key in seen:
                continue
            if any(x in key[0] for x in exclude):
                dropped[t["artist_name"]] = dropped.get(t["artist_name"], 0) + 1
                continue
            seen.add(key)
            out.append({"id": int(t["id"]), "artist": t["artist_name"], "song": t["song_name"],
                        "version": int(t.get("version") or 1), "rating": round(float(t.get("rating") or 0), 2),
                        "votes": int(t.get("votes") or 0), "tuning": t.get("tuning") or "",
                        "tab_url": t["tab_url"], "status": "to download"})
            kept += 1
        print(f"  page {page}: +{kept} new songs ({len(out)} of {songs})")
        if len(out) >= songs or (isinstance(pages, int) and page >= pages) or not tabs:
            break
    if dropped:
        print("  excluded: " + ", ".join(f"{a} ({n})" for a, n in sorted(dropped.items(), key=lambda x: -x[1])))
    return out[:songs]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", help="an /explore URL with its filters, e.g. ...?genres[]=45&order=hitstotal_desc&type[]=Pro")
    ap.add_argument("--songs", type=int, default=500, help="how many distinct songs to shortlist (default 500)")
    ap.add_argument("--max-pages", type=int, default=40)
    ap.add_argument("--exclude", default="", help="comma-separated artist names to skip (mis-tagged genres)")
    ap.add_argument("--genre", default="", help="genre to record on the rows (used when filing by genre folder)")
    ap.add_argument("--delay", type=float, nargs=2, default=[7.5, 15.0], metavar=("MIN", "MAX"))
    ap.add_argument("--cache-days", type=float, default=7.0)
    ap.add_argument("--out", default=str(dl.WORK), help="folder holding list.csv and the page cache (default data/)")
    a = ap.parse_args(argv)
    if "/explore" not in urlsplit(a.url).path:
        sys.exit("not an explore URL; for a band use: python data_dl.py list <artist page>")
    if a.delay[0] < 5:
        sys.exit("--delay below 5 s is not allowed; keep the load on the site low")

    out = Path(a.out)
    fetcher = dl.PoliteFetcher(tuple(a.delay), dl.LIST_UA, out / "cache", a.cache_days)
    try:
        exclude = tuple(x.strip().lower() for x in a.exclude.split(",") if x.strip())
        rows = collect(fetcher, a.url, a.songs, a.max_pages, exclude)
    except dl.Refused as e:
        print(f"\nstopped: {e}\nThe site refused or challenged the request. Not retrying; try again another "
              f"day with a longer --delay.")
        return 3
    if a.genre:
        rows = [r | {"genre": a.genre} for r in rows]
    lst = out / "list.csv"
    merged = dl.merge_into_list(rows, lst)
    print(f"\n{len(rows)} songs shortlisted, {len(merged)} rows in {lst} ({fetcher.requests} requests)")
    print("next: python data_dl.py --no-login --dest <folder>   (uses the browser profile's cookies)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
