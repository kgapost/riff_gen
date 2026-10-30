"""Everything that collects tabs from Ultimate Guitar: the tab list, the download, the sorting.

    python data/dl.py list https://www.ultimate-guitar.com/artist/meshuggah_3907
    python data/dl.py list --file bands.txt --min-rating 4.6 --best-only

``list`` reads the public artist pages (Guitar Pro filter) and adds the free community Guitar Pro tabs
rated above ``--min-rating`` to data/list.csv. It waits 7.5-15 s between requests, caches pages for a
week, checks robots.txt before every request, stops at the first refusal (403/429/challenge page) and
uses no account and no cookies. A row already in the list keeps its status, its file and every column
only other tools write (genre, difficulty, popularity).

    set UG_PASSWORD=...            (or pass --password)
    python data/dl.py --user konstant_apost --limit 3
    python data/dl.py --only-first 11218
    python data/dl.py --retry-failed
    python data/dl.py --report

Without a command, the tabs of data/list.csv are downloaded with a real browser: a headed Chromium
logs in to Ultimate Guitar once, then walks the tab pages in order: render the page, press the
"Download (gp3/gp4/gp5/gpx)" button, save the file, wait 2.5-5 s (plus a 15-20 s pause every 100 tabs
and a 5-10 s pause every 1000), next. Progress is kept in data/state.json, so the run can be stopped
and resumed. tabs.ultimate-guitar.com/robots.txt disallows /tab/download for automated agents: this
uses your own account, which is what is at risk.

Every download is curated on the spot, the way the training set was cleaned (``--no-curate`` skips this):
one version per song is chosen before downloading (the best-rated, then most votes, of a song not downloaded
or on disk yet); the file must open in PyGuitarPro; it is converted to GP5 (8-string tracks are cut to 7
strings, per track the cheapest of: drop the highest string, drop the lowest, shift an octave); instrument
changes are replaced by one track per instrument (when the MIDI channels run out, the first instrument of each
track is kept for the whole tab); 6-string guitars get a 7th string (A); it must pass the training filters of
riff.py prepare (a guitar track, a tuning and time signatures riff.py can encode, no bar too dense), else it is
"rejected" and the song's next best version is tried on the next run; the name is cleaned (no genre tag, no
__v/__id marker, "Last, First" -> "First Last" when the library already spells that artist so); a byte-identical
copy is dropped, and only the largest version of a song stays. The tab ends up in the genre folder of its artist
(the folder where most of the artist's tabs already are), else the one for the list's genre, else
"tabs - metal generic". Anything pushed out or rejected goes to data/__removed, nothing is deleted.
The rules live in data/curate.py.

    python data/dl.py curate some.gpx "some folder"        # the same for files you add by hand

    python data/dl.py get https://tabs.ultimate-guitar.com/tab/meshuggah/bleed-guitar-pro-1234567 --user ...
    python data/dl.py get https://www.ultimate-guitar.com/artist/meshuggah_3907 --genre Metal --user ...

``get`` downloads one tab page, or every song of a band (artist page: its free community Guitar Pro tabs,
``--min-rating`` narrows them), with the same browser, pacing and curation. Of a band, only the best-rated
version of each song not on disk yet is fetched, as always; a tab page is fetched even when the song is
(``--all-versions`` does that for a band too). The tabs are recorded in data/list.csv. Download options pass through.
To curate a collection of your own without touching data/, use data/curate.py.

    python data/dl.py sort --downloads ~/Downloads --dest "data/own/Compound Structure/influences"
    python data/dl.py sort --downloads "data/tabs - generic" --manifest data/state.json --dest ... --move

``sort`` never touches the network: it matches downloaded files to the list, renames them
``Artist - Song.gp5`` / ``Artist - Song__v2.gp5`` (versions of one song form one song family) and files
them into a folder. ``--manifest data/state.json`` names every file exactly, even when two tabs share
artist, song and version.

Paid tabs (Official, Tab Pro) are never listed. Use the files for personal, non-commercial use
(Ultimate Guitar terms, §6.3).
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import html
import json
import os
import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import urllib.robotparser
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

if TYPE_CHECKING:  # playwright is only needed to download, so it is imported there
    from playwright.sync_api import BrowserContext, Page

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curate as cur  # noqa: E402 - the curation rules (and data/convert.py) live next to this file
from curate import GENRE_SUFFIX, GENRES, GP_EXTENSIONS, NoChannel, clean, convert_module, gp, stem_key  # noqa: E402,F401


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
WORK = ROOT / "data"
PROFILE = WORK / "profile"
DOWNLOAD_DIR = WORK / "tabs - generic"      # new tabs land here
ERROR_DIR = WORK / "errors"
STATE_PATH = WORK / "state.json"
LOG_PATH = WORK / "downloads.log"
LIST_PATH = WORK / "list.csv"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36")

# The button label carries the format: "Download (gp5)", "Download (gpx)", ...
BUTTON_SELECTOR = 'button:has-text("Download (gp")'

# Text that means "stop, do not keep hammering": daily quota or a paywall.
STOP_MARKERS = (
    "daily download limit",
    "download limit",
    "you have reached your",
    "limit of downloads",
    "too many downloads",
)


def song_key(artist: str, song: str) -> str:
    return re.sub(r"^the", "", re.sub(r"[^a-z0-9]+", "", clean(f"{artist}{song}").lower()))


@dataclass
class Target:
    url: str
    tab_id: str = ""
    artist: str = ""
    song: str = ""
    version: int = 1
    list_status: str = "to download"   # the list.csv status column; anything else = do not fetch
    label: str = ""
    rating: float = 0.0
    votes: int = 0
    genre: str = ""

    @property
    def stem(self) -> str:
        if self.artist and self.song:
            suffix = "" if self.version <= 1 else f"__v{self.version}"
            return f"{safe(self.artist)} - {safe(self.song)}{suffix}"
        return f"tab-{self.tab_id or 'unknown'}"


@dataclass
class Run:
    ok: int = 0
    failed: int = 0
    blocked: int = 0
    skipped: int = 0
    stopped: str = ""
    started: float = field(default_factory=time.time)


def safe(s: str) -> str:
    return clean(s)      # no special characters in file names


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# list parsing


def tab_id_from_url(url: str) -> str:
    """.../tab/a-perfect-circle/3-libras-guitar-pro-214593 -> '214593'; .../tab/4835048 -> '4835048'.

    Splitting on the last dash is wrong for short URLs, because the host name has one too.
    """
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    m = re.search(r"(\d+)$", tail)
    return m.group(1) if m else ""


def nice_slug(url: str, last: bool = False) -> str:
    """a-perfect-circle/3-libras-guitar-pro-214593 -> 'A Perfect Circle' / '3 Libras'"""
    parts = url.split("/tab/", 1)[-1].split("/")
    if last:
        slug = parts[-1] if parts else ""
        slug = re.sub(r"-guitar-pro-\d+$|-power-tab-\d+$|-\d+$", "", slug)
    else:
        slug = parts[0] if parts else ""
        slug = re.sub(r"_\d+$", "", slug)
    return slug.replace("-", " ").title()


def load_targets(path: Path) -> list[Target]:
    """The tabs of a list .csv (id, artist, song, version, rating, votes, tab_url, ...), in file order."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    seen, out = set(), []
    for r in rows:
        url = r["tab_url"]
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(Target(
            url=url,
            tab_id=r.get("id") or tab_id_from_url(url),
            artist=r.get("artist") or nice_slug(url),
            song=r.get("song") or nice_slug(url, last=True),
            version=int(float(r.get("version") or 1)),
            rating=float(r.get("rating") or 0),
            votes=int(float(r.get("votes") or 0)),
            genre=(r.get("genre") or "").strip(),
            list_status=(r.get("status") or "to download").strip(),
        ))
    return out


def keep_best_versions(targets: list[Target], n: int) -> tuple[list[Target], int, int]:
    """Keep the n best-rated versions of each song; songs with n or fewer rows are left alone.

    Songs are (artist, song) case-insensitively; "best" is rating, then votes. Returns the kept
    targets in their original order, how many rows were dropped, and how many songs were affected.
    """
    groups: dict[tuple[str, str], list[Target]] = {}
    for t in targets:
        groups.setdefault((t.artist.strip().lower(), t.song.strip().lower()), []).append(t)

    keep_urls, dropped, affected = set(), 0, 0
    for rows in groups.values():
        if len(rows) <= n:
            keep_urls.update(t.url for t in rows)
            continue
        affected += 1
        best = sorted(rows, key=lambda t: (-t.rating, -t.votes))[:n]
        keep_urls.update(t.url for t in best)
        dropped += len(rows) - n
    return [t for t in targets if t.url in keep_urls], dropped, affected


# ---------------------------------------------------------------------------
# state


def load_state() -> dict:
    if STATE_PATH.is_file():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    for attempt in range(20):
        try:
            tmp.replace(STATE_PATH)
            return
        except PermissionError:  # another process has the file open for a moment
            time.sleep(0.5)
    STATE_PATH.write_text(tmp.read_text(encoding="utf-8"), encoding="utf-8")
    tmp.unlink(missing_ok=True)


def append_log(row: dict) -> None:
    new = not LOG_PATH.is_file()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["at", "tab_id", "artist", "song", "version", "status",
                                          "button", "file", "note"])
        if new:
            w.writeheader()
        w.writerow(row)


def write_progress(run: "Run", done: int, total: int, target: "Target | None",
                   elapsed: float, finished: bool = False) -> None:
    """A small file the user (or a monitoring script) can read while the run goes on."""
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "progress.json").write_text(json.dumps({
        "updated": datetime.now().isoformat(timespec="seconds"),
        "done": done, "total": total, "ok": run.ok, "failed": run.failed, "blocked": run.blocked,
        "current": target.stem if target else None,
        "current_url": target.url if target else None,
        "stopped": run.stopped, "finished": finished, "elapsed_s": round(elapsed),
    }, indent=1), encoding="utf-8")


# ---------------------------------------------------------------------------
# browser


def login_controls(page: Page) -> list:
    """The 'Log in' link/button, when it is on the page (= not signed in)."""
    out = []
    for el in page.query_selector_all("button, a"):
        try:
            if (el.inner_text() or "").strip().lower() in ("log in", "login"):
                out.append(el)
        except Exception:
            continue
    return out


def login(ctx: BrowserContext, user: str, password: str) -> bool:
    """Sign in, unless the browser profile already is (it keeps its cookies)."""
    from playwright.sync_api import TimeoutError as PWTimeout

    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    page.goto("https://www.ultimate-guitar.com/", wait_until="domcontentloaded", timeout=60_000)
    page.wait_for_timeout(2500)

    if page.query_selector("#login-form-password") is None:
        controls = login_controls(page)
        if not controls:
            log("already logged in (cookies from an earlier run)")
            return True
        try:
            controls[0].click()
        except Exception:
            pass
        page.wait_for_timeout(2500)

    try:
        page.wait_for_selector("#login-form-email", timeout=15_000)
        page.fill("#login-form-email", user)
        page.fill("#login-form-password", password)
    except PWTimeout:
        ERROR_DIR.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(ERROR_DIR / "login_failed.png"))
        (ERROR_DIR / "login_failed.html").write_text(page.content(), encoding="utf-8")
        log("login FAILED: no login form (see data/errors/login_failed.png)")
        return False

    page.wait_for_timeout(600)
    page.keyboard.press("Enter")
    page.wait_for_timeout(6000)

    # success = the form is gone and no 'Log in' control came back
    if page.query_selector("#login-form-password") is None and not login_controls(page):
        log("logged in")
        return True
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(ERROR_DIR / "login_failed.png"))
    (ERROR_DIR / "login_failed.html").write_text(page.content(), encoding="utf-8")
    log("login FAILED (see data/errors/login_failed.png)")
    return False


def check_file(path: Path) -> str:
    """Return '' if the download looks like a Guitar Pro file, else a reason."""
    data = path.read_bytes()
    head = data[:64]
    # gp3/gp4/gp5 start with a length byte then the magic, e.g. b'\x18FICHIER GUITAR PRO v5.00'
    if b"FICHIER GUITAR PRO" in head:
        return ""
    # gpx (Guitar Pro 6) is not a zip: b'BCFZ' then 7 fixed bytes at offset 8; the 8th varies
    if head.startswith(b"BCFZ") and head[8:15] == b"\x68\x48\x68\xcd\x4c\x00\x01":
        return ""
    if head.startswith(b"PK"):  # .gp (Guitar Pro 7) is a zip
        return ""
    if head.lstrip().startswith(b"<"):
        return "server sent an HTML page"
    if len(data) < 512:
        return f"suspiciously small file ({len(data)} bytes)"
    return f"unexpected file header {head[:16]!r}"


def page_blocked(page: Page) -> str:
    """Why the page has no download button, if the site says so itself."""
    try:
        body = page.inner_text("body")[:6000].lower()
    except Exception:
        return ""
    for needle, why in (
        ("can't show this tab", "artist does not allow this tab to be shown"),
        ("cannot show this tab", "artist does not allow this tab to be shown"),
        ("has been blocked", "tab has been blocked"),
        ("tab has been removed", "tab has been removed"),
        ("not available in your country", "not available in this country"),
        ("this tab is not available", "tab is not available"),
    ):
        if needle in body:
            return why
    return ""


def goto_with_retry(page: Page, url: str, tries: int = 2, timeout_ms: int = 90_000) -> str:
    """Navigate, retrying once: the site drops a page now and then (DNS, connection reset)."""
    last = ""
    for attempt in range(1, tries + 1):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            return ""
        except Exception as e:
            last = f"{type(e).__name__}: {str(e).splitlines()[0][:150]}"
            if attempt < tries:
                log(f"    navigation failed ({last}), retrying")
                time.sleep(5)
    return f"goto failed {tries}x: {last}"


def wait_for_button_or_block(page: Page, timeout_s: float = 25.0):
    """The download button, or a string saying why the page will never have one, or None."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            button = page.query_selector(BUTTON_SELECTOR)
        except Exception:  # mid-navigation
            button = None
        if button is not None:
            return button
        why = page_blocked(page)
        if why:
            return why
        page.wait_for_timeout(500)
    return None


def grab(ctx: BrowserContext, page: Page, t: Target, dest: Path,
         name_owner: dict[str, str] | None = None, wait_s: int = 90) -> tuple[str, str, str]:
    """Open t.url, press the download button, save the file. Returns (status, file name, note)."""
    failed = goto_with_retry(page, t.url)
    if failed:
        return "failed", "", failed

    found = wait_for_button_or_block(page)
    if isinstance(found, str):
        return "blocked", "", found
    if found is None:
        return "failed", "", "no download button on the page"
    button = found

    # a short "/tab/<id>" URL carries no names; the page's keywords meta does
    if t.artist.isdigit() or t.song.isdigit():
        meta = page.query_selector('meta[name="keywords"]')
        keywords = (meta.get_attribute("content") or "") if meta else ""
        if " - " in keywords and " (" in keywords:
            artist, rest = keywords.split(" - ", 1)
            song = rest.split(" (", 1)[0]
            if artist.strip() and song.strip():
                t = replace(t, artist=artist.strip(), song=song.strip())
    label = (button.inner_text() or "").strip()
    ext = re.sub(r"[()]", "", label.split("Download")[-1]).strip() or "gp"
    if not ext.startswith("."):
        ext = "." + ext

    downloads: list = []

    def on_download(d) -> None:
        downloads.append(d)

    pages_before = set(ctx.pages)
    ctx.on("download", on_download)

    try:
        button.evaluate("el => el.click()")
        deadline = time.time() + wait_s
        extra_page = None
        while time.time() < deadline and not downloads:
            new_pages = [p for p in ctx.pages if p not in pages_before]
            if new_pages:
                extra_page = new_pages[-1]
            page.wait_for_timeout(400)

        if not downloads:
            note = "no download event"
            if extra_page is not None:
                try:
                    extra_page.wait_for_load_state("domcontentloaded", timeout=5000)
                    body = (extra_page.inner_text("body") or "")[:400].replace("\n", " ")
                    note = f"opened page instead: {extra_page.url} :: {body}"
                    if any(m in body.lower() for m in STOP_MARKERS):
                        return "stopped", "", note
                except Exception:
                    pass
            return "failed", "", note

        dl = downloads[0]
        suggested = dl.suggested_filename or f"tab{t.tab_id}{ext}"
        dest.mkdir(parents=True, exist_ok=True)
        name = f"{t.stem}{ext}"
        owner = (name_owner or {}).get(name)
        if (owner and owner != t.url) or (owner is None and (dest / safe(name)).exists()):
            # another tab (or a file this tool did not fetch) already has this name: keep the id
            name = f"{t.stem}__id{t.tab_id}{ext}"
        target = dest / safe(name)
        replaced = target.exists()  # only ever our own earlier download of this very tab
        if replaced:
            target.unlink()
        dl.save_as(str(target))
        problem = check_file(target)
        if problem:
            ERROR_DIR.mkdir(parents=True, exist_ok=True)
            broken = ERROR_DIR / f"{t.tab_id or 'x'}{ext}.bad"
            target.replace(broken)
            return "failed", "", f"{problem} -> {broken.name}"
        if name_owner is not None:
            name_owner[target.name] = t.url
        return "ok", target.name, f"from {suggested}" + (" (replaced)" if replaced else "")
    finally:
        try:
            ctx.remove_listener("download", on_download)
        except Exception:
            pass
        for p in set(ctx.pages) - pages_before:
            try:
                p.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# list: shortlist good tabs from the public artist pages (no account, no cookies)

LIST_UA = "Mozilla/5.0 (compatible; RiffGen-shortlist/0.1; personal use, low rate)"


class Refused(Exception):
    """The site refused or challenged a request: stop, do not retry."""


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


# The tab list is one CSV (data/list.csv). Columns other tools add (genre, difficulty, popularity, ...)
# are kept as they are: a rewrite never drops a column it does not know.
LIST_COLUMNS = ["id", "artist", "song", "version", "rating", "votes", "tuning", "tab_url", "status", "file"]
INT_COLUMNS = ("id", "version", "votes", "popularity")


def read_list(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in INT_COLUMNS:
            if r.get(k):
                r[k] = int(float(r[k]))
        r["rating"] = float(r.get("rating") or 0)
        r["version"] = r.get("version") or 1
        r["status"] = r.get("status") or "to download"
    return rows


def write_list(rows: list[dict], path: Path) -> None:
    columns = list(LIST_COLUMNS)
    for r in rows:
        columns += [k for k in r if k not in columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, restval="")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def cmd_list(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="dl.py list", description="shortlist good Guitar Pro tabs from artist pages")
    ap.add_argument("urls", nargs="*", help="artist page URLs, e.g. https://www.ultimate-guitar.com/artist/meshuggah_3907")
    ap.add_argument("--file", help="text file with one artist URL per line")
    ap.add_argument("--min-rating", type=float, default=4.4, help="keep tabs rated above this (default 4.4)")
    ap.add_argument("--min-votes", type=int, default=1, help="ignore ratings with fewer votes")
    ap.add_argument("--best-only", action="store_true", help="only the best-rated version of each song")
    ap.add_argument("--delay", type=float, nargs=2, default=[7.5, 15.0], metavar=("MIN", "MAX"),
                    help="random pause between requests in seconds (default 7.5-15, minimum 5)")
    ap.add_argument("--cache-days", type=float, default=7.0)
    ap.add_argument("--user-agent", default=LIST_UA)
    ap.add_argument("--out", default=str(WORK), help="folder of list.csv and the page cache (default data/)")
    args = ap.parse_args(argv)

    urls = list(args.urls)
    if args.file:
        urls += [u.strip() for u in Path(args.file).read_text(encoding="utf-8").splitlines()
                 if u.strip() and not u.strip().startswith("#")]
    if not urls:
        sys.exit("give one or more artist page URLs (or --file)")
    if args.delay[0] < 5:
        sys.exit("--delay below 5 s is not allowed; keep the load on the site low")
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
    lst = out / "list.csv"
    rows = merge_into_list(all_rows, lst)
    print(f"\n{len(rows)} tabs in {lst} ({sum(r['status'] == 'sorted' for r in rows)} sorted, "
          f"{fetcher.requests} requests)")
    return 0


def merge_into_list(new_rows: list[dict], lst: Path) -> list[dict]:
    """Add rows to a tab list. Fresh data updates rating, votes, ...; a row already in the list keeps its status
    and file and every column only other tools write (genre, difficulty, popularity)."""
    old = {r["id"]: r for r in read_list(lst)} if lst.is_file() else {}
    merged = {}
    for r in list(old.values()) + new_rows:
        prev = old.get(r["id"], {})
        merged[r["id"]] = {**prev, **r, **{k: prev[k] for k in ("status", "file") if k in prev}}
    rows = sorted(merged.values(), key=lambda r: (r["artist"].lower(), r["song"].lower(), int(r["version"])))
    write_list(rows, lst)
    return rows


# ---------------------------------------------------------------------------
# get: one tab, or every tab of a band, straight to the download


def tab_row(url: str) -> dict:
    """A list row for one tab page URL; artist and song come from the address."""
    parts = urlsplit(url.strip())
    if "ultimate-guitar.com" not in parts.netloc or "/tab/" not in parts.path:
        raise ValueError(f"not a tab page: {url}")
    clean_url = urlunsplit((parts.scheme or "https", parts.netloc, parts.path.rstrip("/"), "", ""))
    tid = tab_id_from_url(clean_url)
    if not tid:
        raise ValueError(f"no tab id in {url}")
    return {"id": int(tid), "artist": nice_slug(clean_url), "song": nice_slug(clean_url, last=True), "version": 1,
            "rating": 0.0, "votes": 0, "tuning": "", "tab_url": clean_url, "status": "to download"}


def cmd_get(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="dl.py get", description="Download one tab, or every free community Guitar Pro tab of a band. "
        "Any option of the plain download (--user, --password, --no-login, --no-curate, --dry-run, --limit, "
        "--dest, ...) can be added.")
    ap.add_argument("urls", nargs="*", help="tab pages (https://tabs.ultimate-guitar.com/tab/...) and/or artist "
                                            "pages (https://www.ultimate-guitar.com/artist/...)")
    ap.add_argument("--file", help="text file with one URL per line")
    ap.add_argument("--min-rating", type=float, default=-1.0,
                    help="of a band, only the tabs rated above this (default: all, unrated ones too)")
    ap.add_argument("--min-votes", type=int, default=0)
    ap.add_argument("--best-only", action="store_true", help="of a band, only the best-rated version of each song")
    ap.add_argument("--genre", default="", help="genre for tabs whose artist is not in the library yet (e.g. Metal)")
    ap.add_argument("--delay", type=float, nargs=2, default=[7.5, 15.0], metavar=("MIN", "MAX"),
                    help="pause between artist page requests (default 7.5-15 s, minimum 5)")
    ap.add_argument("--cache-days", type=float, default=7.0)
    args, rest = ap.parse_known_args(argv)

    urls = list(args.urls)
    if args.file:
        urls += [u.strip() for u in Path(args.file).read_text(encoding="utf-8").splitlines()
                 if u.strip() and not u.strip().startswith("#")]
    if not urls:
        sys.exit("give one or more tab or artist page URLs (or --file)")
    if args.delay[0] < 5:
        sys.exit("--delay below 5 s is not allowed; keep the load on the site low")
    old = {r["id"]: r for r in read_list(LIST_PATH)} if LIST_PATH.is_file() else {}
    fetcher = PoliteFetcher(tuple(args.delay), LIST_UA, WORK / "cache", args.cache_days)
    rows: list[dict] = []
    picked = True                                       # only tab pages: the user chose these versions
    for url in urls:
        try:
            if "/artist/" in urlsplit(url).path:
                picked = False
                tabs = collect_artist(fetcher, url)
                found = shortlist(tabs, args.min_rating, args.min_votes, args.best_only)
                log(f"{artist_base(url)}: {len(tabs)} free community Guitar Pro tabs, {len(found)} to get")
            else:
                r = tab_row(url)
                found = [{**old[r["id"]], "tab_url": r["tab_url"]} if r["id"] in old else r]   # the list's names are better
                log(f"{r['tab_url']}: {found[0]['artist']} - {found[0]['song']}")
        except (ValueError, Refused) as e:
            log(f"skipped: {e}")
            continue
        rows += [{**r, "genre": r.get("genre") or args.genre} for r in found]
    if not rows:
        return 1
    merge_into_list(rows, LIST_PATH)                    # recorded in data/list.csv, statuses untouched
    run_list = WORK / "cache" / "get.csv"               # what this run fetches
    write_list([r | {"status": "to download"} for r in rows], run_list)
    # a band: the best-rated version of each song not on disk yet, as always; tab pages: exactly those
    return cmd_download(["--list", str(run_list), *(["--all-versions"] if picked else []), *rest])


# ---------------------------------------------------------------------------
# sort: rename downloaded files after the list and file them into a folder (no network)

OLD_GP = (".gp3", ".gp4", ".gp5")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def target_name(row: dict, ext: str) -> str:
    suffix = "" if int(row["version"]) <= 1 else f"__v{row['version']}"
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


def prepare_rows(rows: list[dict]) -> list[tuple[str, str, int, dict]]:
    """(normalised artist, normalised song, length of the song, row) — computed once per sort."""
    return [(norm(r["artist"]), norm(r["song"]), len(norm(r["song"])), r) for r in rows]


def match_file(path: Path, rows: list[dict], prepared=None, by_id=None) -> tuple[dict | None, str]:
    if prepared is None:
        prepared = prepare_rows(rows)
    # "Artist - Song__v2__id123456.gp5" (written when two tabs of one song would get the same name)
    m = re.search(r"__id(\d+)", path.stem)
    if m:
        hit = (by_id or {}).get(m.group(1))
        if hit:
            return hit, ""
    stem = norm(path.stem)
    cands = [(length, r) for na, ns, length, r in prepared if na in stem and ns in stem]
    if not cands:
        artist, title = gp_metadata(path)
        if title:
            ntitle, nartist = norm(title), norm(artist)
            cands = [(length, r) for na, ns, length, r in prepared if ns == ntitle
                     and (not nartist or na in nartist or nartist in na)]
    if not cands:
        return None, "no list entry matches"
    # prefer the longest song name ("Bleed" vs "Bleed (Live)")
    best_len = max(length for length, _ in cands)
    cands = [r for length, r in cands if length == best_len]
    if len(cands) == 1:
        return cands[0], ""
    m = re.search(r"(?:ver(?:sion)?|v)\s*\.?\s*(\d+)|\((\d+)\)", path.stem, re.I)
    if m:
        v = int(m.group(1) or m.group(2))
        exact = [r for r in cands if int(r["version"]) == v]
        if len(exact) == 1:
            return exact[0], ""
    left = [r for r in cands if r["status"] != "sorted"]
    if len(left) == 1:
        return left[0], ""
    return None, f"ambiguous: versions {sorted(int(r['version']) for r in cands)} of {cands[0]['song']}"


def read_manifest(path: Path) -> dict[str, str]:
    """File name -> tab id, from the downloader's state file (data/state.json).

    Two tabs of one song can share artist/song/version, which makes the name
    ambiguous; the downloader knows exactly which tab id each file came from.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for url, entry in data.items():
        saved = entry.get("path")
        if saved and entry.get("status") == "ok":
            out[Path(saved).name] = tab_id_from_url(url)
    return out


def cmd_sort(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="dl.py sort", description="rename downloaded tabs after the list and file them")
    ap.add_argument("--list", "--shortlist", dest="shortlist", default=str(LIST_PATH))
    ap.add_argument("--downloads", default="~/Downloads")
    ap.add_argument("--dest", required=True, help='e.g. "data/own/Compound Structure/influences"')
    ap.add_argument("--move", action="store_true", help="move instead of copy")
    ap.add_argument("--manifest", help="data/state.json: resolves files that the name alone cannot")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    sl = Path(args.shortlist)
    rows = read_list(sl)
    downloads = Path(args.downloads).expanduser()
    if not downloads.is_dir():
        sys.exit(f"no such folder: {downloads}")
    dest = Path(args.dest)
    if not dest.is_absolute():
        dest = ROOT / dest
    by_id = {str(r["id"]): r for r in rows}
    manifest: dict[str, dict] = {}
    if args.manifest:
        for name, tab_id in read_manifest(Path(args.manifest)).items():
            if tab_id in by_id:
                manifest[name] = by_id[tab_id]
        print(f"{len(manifest)} files resolve through {Path(args.manifest).name}")
    files = sorted(p for p in downloads.iterdir() if p.is_file() and p.suffix.lower() in GP_EXTENSIONS)
    prepared = prepare_rows(rows)
    done, problems, export = [], [], []
    for p in files:
        row, why = manifest.get(p.name), ""
        if row is None:
            row, why = match_file(p, rows, prepared, by_id)
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
        write_list(rows, sl)
    print(f"{'would sort' if args.dry_run else 'sorted'} {len(done)} files into {dest}")
    for line in done:
        print("  " + line)
    if export:
        print(f"\n{len(export)} files are Guitar Pro 6/7 (.gpx/.gp): convert them with python data/convert.py")
        for e in export:
            print("  " + e)
    if problems:
        print(f"\n{len(problems)} files not sorted:")
        for line in problems:
            print("  " + line)
    print(f"\n{len(missing)} listed tabs still to download (see {sl})")
    return 0


# ---------------------------------------------------------------------------
# download


def report(last: int) -> None:
    """Summarise state.json: how many tabs are done and why the rest failed."""
    def bucket(note: str) -> str:
        for needle, label in (("Page.goto: Timeout", "goto timeout"), ("no download button", "no download button"),
                              ("blocked", "blocked / removed"), ("HTML page", "HTML page instead of a file"),
                              ("Timeout", "other timeout"), ("unexpected file header", "unknown file format")):
            if needle in note:
                return label
        return note[:70] or "(no note)"

    if not STATE_PATH.is_file():
        print(f"no ledger at {STATE_PATH}; the downloader writes one as it runs")
        return
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    bad = {u: v for u, v in state.items() if v.get("status") != "ok"}
    print(f"{len(state)} urls tracked: {len(state) - len(bad)} ok, {len(bad)} not ok")
    if bad:
        print("\nwhy:")
        for reason, count in collections.Counter(bucket(v.get("note", "")) for v in bad.values()).most_common():
            print(f"  {count:5}  {reason}")
        print(f"\nlast {min(last, len(bad))} failures:")
        for url, entry in list(bad.items())[-last:]:
            print(f"  {url}\n      {entry['status']}: {entry.get('note', '')[:110]}")


# ---------------------------------------------------------------------------
# curation: every new tab is cleaned the way the training set was cleaned

KEEP_VERSIONS = 1                           # of one song, the largest different versions that stay
TRAIN_FILTERS = True                        # reject what riff.py prepare would skip (needs numpy and torch)


def Library(work: Path | None = None) -> cur.Library:
    """Index of the tabs* folders of data/ (songs, artists, file names), kept current as files come and go."""
    return cur.Library(WORK if work is None else work, genre_folders=True, pattern="tabs*")


def curate_file(src: Path, genre: str, lib: cur.Library) -> tuple[str, str, str]:
    """Clean one downloaded tab and file it (data/curate.py). -> (status, final path, note); status is
    ok / duplicate (not kept: same file, or not among the best versions; the file is in __removed) / rejected
    (fails the training filters; in __removed/rejected) / failed."""
    return cur.curate_file(src, genre, lib, KEEP_VERSIONS, TRAIN_FILTERS)


def curate_download(src: Path, t: Target, lib: cur.Library) -> tuple[str, str, str]:
    """curate_file for a fresh download; a file that cannot be curated goes to data/errors for a look."""
    status, final, note = curate_file(src, t.genre, lib)
    if status == "failed":
        ERROR_DIR.mkdir(parents=True, exist_ok=True)
        bad = ERROR_DIR / f"{t.tab_id or 'x'}{src.suffix}.bad"
        if src.exists():
            shutil.move(str(src), str(bad))
        return "failed", "", f"curate: {note} -> {bad.name}"
    src.unlink(missing_ok=True)
    return status, final, note


def cmd_curate(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="dl.py curate", description="Curate Guitar Pro files that were not downloaded by this tool.")
    ap.add_argument("paths", nargs="+", help="files, or folders (all Guitar Pro files in them)")
    ap.add_argument("--genre", default="", help="the Deezer genre to use when the artist is not in the library")
    args = ap.parse_args(argv)
    files = []
    for s in args.paths:
        p = Path(s)
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in GP_EXTENSIONS) if p.is_dir() else [p]
    lib, count = Library(), collections.Counter()
    for f in files:
        status, final, note = curate_file(f, args.genre, lib)
        count[status] += 1
        log(f"{status}: {f.name} :: {note}")
        if status != "failed":
            f.unlink(missing_ok=True)
    log(f"{dict(count)}")
    return 0 if not count["failed"] else 1


def main(argv=None) -> int:
    # tab names contain anything (accents, CJK, dashes); a redirected console must not choke on them
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in ("list", "sort", "curate", "get"):
        return {"list": cmd_list, "sort": cmd_sort, "curate": cmd_curate, "get": cmd_get}[argv[0]](argv[1:])
    return cmd_download(argv)


def cmd_download(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", "--shortlist", dest="shortlist", default=str(LIST_PATH),
                    help="the tab list .csv to download from (default data/list.csv)")
    ap.add_argument("--dest", default=str(DOWNLOAD_DIR))
    ap.add_argument("--limit", type=int, default=0, help="stop after N tabs (0 = no limit)")
    ap.add_argument("--only-first", type=int, default=0,
                    help="only the first N tabs of the list (e.g. half of it)")
    ap.add_argument("--max-versions", type=int, default=0, metavar="N",
                    help="keep only the N best-rated versions of each song (0 = keep all)")
    ap.add_argument("--gap", type=float, nargs=2, default=[2.5, 5.0], metavar=("MIN", "MAX"))
    ap.add_argument("--pause-100", type=float, nargs=2, default=[15.0, 20.0], metavar=("MIN", "MAX"),
                    help="extra pause every 100 tabs")
    ap.add_argument("--pause-1000", type=float, nargs=2, default=[5.0, 10.0], metavar=("MIN", "MAX"),
                    help="extra pause every 1000 tabs")
    ap.add_argument("--max-failures", type=int, default=10,
                    help="stop after this many failures in a row (default 10)")
    ap.add_argument("--user", default=os.environ.get("UG_USER", ""))
    ap.add_argument("--password", default=os.environ.get("UG_PASSWORD", ""))
    ap.add_argument("--no-login", action="store_true", help="reuse the browser profile's cookies only")
    ap.add_argument("--retry-failed", action="store_true",
                    help="retry tabs recorded as failed or stopped")
    ap.add_argument("--retry-blocked", action="store_true",
                    help="retry tabs the site refuses to show (usually pointless, they do not come back)")
    ap.add_argument("--no-curate", action="store_true",
                    help="leave downloads as they come (default: convert, clean and file each one on the spot)")
    ap.add_argument("--all-versions", action="store_true",
                    help="fetch every version of a song, even when the song is already on disk "
                         "(curation still keeps only the largest versions)")
    ap.add_argument("--dry-run", action="store_true", help="only list what would be downloaded")
    ap.add_argument("--start-at", type=int, default=0, help="skip the first N targets")
    ap.add_argument("--report", nargs="?", const=15, type=int, metavar="LAST",
                    help="only summarise state.json (how many tabs are done, why the rest failed, the last LAST failures)")
    args = ap.parse_args(argv)
    if args.report is not None:
        report(args.report)
        return 0

    shortlist = Path(args.shortlist)
    if not shortlist.is_absolute():
        shortlist = (ROOT / shortlist).resolve()
    if not shortlist.is_file():
        sys.exit(f"no such list: {shortlist}")

    targets = load_targets(shortlist)
    log(f"{len(targets)} tabs in {shortlist.name}")
    if args.only_first:
        targets = targets[: args.only_first]
        log(f"limited to the first {len(targets)}")
    if args.max_versions:
        targets, dropped, affected = keep_best_versions(targets, args.max_versions)
        log(f"keeping the best {args.max_versions} versions of each song: {affected} songs had more, "
            f"{dropped} rows dropped, {len(targets)} left")
    if args.start_at:
        targets = targets[args.start_at:]

    dest = Path(args.dest)
    if not dest.is_absolute():
        dest = (ROOT / dest).resolve()

    state = load_state()
    if not args.no_curate and not args.dry_run:
        if gp is None:
            sys.exit("curating needs PyGuitarPro (pip install PyGuitarPro), or pass --no-curate")
        if TRAIN_FILTERS:
            try:
                cur.training_check(gp.Song())       # loads riff.py (numpy, torch) once, before the first download
            except ImportError as e:
                sys.exit(f"the training filters need riff.py's packages ({e}); install them or pass --no-curate")
            except Exception:                       # noqa: BLE001 - an empty song may upset the encoder; it is loaded
                pass
        lib = Library()
    else:
        lib = None
    # which tab already owns which file name (so two tabs of one song do not overwrite each other)
    name_owner = {Path(v["path"]).name: u for u, v in state.items() if v.get("path")}

    # a downloaded file may have been moved between folders since it was fetched, so look for it
    # wherever our files live, not only at the path recorded back then
    known_dirs = [d for d in dict.fromkeys([dest, *sorted(WORK.glob("tabs*"))]) if d.is_dir()]

    def present(entry: dict) -> bool:
        saved = entry.get("path")
        if not saved:
            return True
        path = Path(saved)
        return path.is_file() or any((d / path.name).is_file() for d in known_dirs)

    # "already downloaded" is judged by tab id, not by URL: the same tab shows up in more than one
    # list, with different slugs in the address
    done_ids = set()
    for url, entry in state.items():
        status = entry.get("status")
        if status == "ok" and present(entry):
            done_ids.add(tab_id_from_url(url))
    # ... and by what already sits in the folders, for files this tool did not fetch
    on_disk: dict[str, Path] = {}
    for d in known_dirs:
        if d.name.lower() == "tabs - generic":  # a mixed collection, not just our downloads
            continue
        for p in d.iterdir():
            if p.suffix.lower() in GP_EXTENSIONS:
                on_disk.setdefault(re.sub(r"__id\d+$", "", p.stem), p)

    todo, skipped = [], collections.Counter()
    reconciled = 0
    for t in targets:
        st = state.get(t.url, {})
        if t.list_status != "to download":  # marked in list.csv: already have this song, or sorted
            skipped["marked in list.csv"] += 1
            continue
        if st.get("status") == "ok" and present(st):
            skipped["downloaded"] += 1
            continue
        if st.get("status") == "blocked" and not args.retry_blocked:
            continue
        if st.get("status") == "rejected":           # fails the training filters; it will not change
            skipped["rejected by the training filters"] += 1
            continue
        if st.get("status") in ("failed", "stopped") and not args.retry_failed:
            continue
        if t.tab_id and t.tab_id in done_ids:
            skipped["downloaded"] += 1
            continue
        found = None if args.all_versions else on_disk.get(re.sub(r"__id\d+$", "", t.stem))
        if found is not None:
            skipped["already on disk"] += 1
            if st.get("status") != "ok" and found.parent.name == "tabs":
                # heal the ledger (only for our own old downloads; other folders are mixed collections): the file is here, so record which tab it is (sort --manifest
                # needs this mapping, and the downloader needs it to skip by tab id later)
                state[t.url] = {"status": "ok", "file": found.name, "path": str(found),
                                "note": "found on disk at startup",
                                "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                reconciled += 1
            continue
        todo.append(t)
    if reconciled and not args.dry_run:
        save_state(state)
        log(f"scan: recorded {reconciled} already-downloaded tabs in {STATE_PATH.name}")

    # song-level pass: the same song under another tab id, another version, or another collection
    song_dirs = [d for d in WORK.glob("tabs*") if d.is_dir()]
    have_songs = {stem_key(p.stem) for d in song_dirs for p in d.iterdir() if p.suffix.lower() in GP_EXTENSIONS}
    done_songs = {song_key(t.artist, t.song) for t in targets if t.tab_id and t.tab_id in done_ids}
    # one version per song, chosen before downloading: the best-rated one (then most votes) of the versions not
    # fetched yet; a song already downloaded or on disk gets none (a rejected or blocked best version makes the
    # next best one the choice of the next run)
    queue, seen_song = [], {}
    for t in sorted(todo, key=lambda t: (-t.rating, -t.votes, -t.version)):
        if args.all_versions:
            queue.append(t)
            continue
        k = song_key(t.artist, t.song)
        if k in done_songs:
            skipped["song already downloaded"] += 1
            continue
        if k in have_songs:
            skipped["song already on disk"] += 1
            continue
        if k in seen_song:
            skipped["other version of a song chosen in this run"] += 1
            continue
        seen_song[k] = t
        queue.append(t)
    order = {t.url: i for i, t in enumerate(todo)}
    todo = sorted(queue, key=lambda t: order[t.url])

    detail = ", ".join(f"{n} {why}" for why, n in skipped.most_common()) or "none"
    log(f"{len(targets)} tabs in the list, {len(targets) - len(todo)} skipped ({detail})")
    if args.limit:
        todo = todo[: args.limit]

    # up-front plan: how many, and how long this should take at the configured pacing
    per_tab = sum(args.gap) / 2 + 1.5  # wait between tabs + about 1.5 s to open the page and save
    est_total = (len(todo) * per_tab + (len(todo) // 100) * sum(args.pause_100) / 2
                 + (len(todo) // 1000) * sum(args.pause_1000) / 2)
    log(f"PLAN: {len(todo)} tabs to download (no duplicates, nothing already downloaded). "
        f"Estimated {int(est_total // 3600)}h {int(est_total % 3600 // 60):02d}m, "
        f"finishing around {datetime.fromtimestamp(time.time() + est_total):%H:%M}")

    if args.dry_run:
        for t in todo[:20]:
            print(f"  {t.tab_id:>8}  {t.stem}")
        return 0

    from playwright.sync_api import sync_playwright

    run = Run()
    consecutive = 0
    PROFILE.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            str(PROFILE), headless=False, accept_downloads=True,
            viewport={"width": 1360, "height": 900},
            args=[f"--user-agent={UA}", "--disable-blink-features=AutomationControlled"],
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()

        if not args.no_login:
            if not args.user or not args.password:
                ctx.close()
                sys.exit("need --user and --password (or UG_USER / UG_PASSWORD)")
            if not login(ctx, args.user, args.password):
                ctx.close()
                return 2

        try:
            for i, t in enumerate(todo, 1):
                elapsed = time.time() - run.started
                left = len(todo) - i + 1
                eta = elapsed / (i - 1) * left if i > 1 else est_total
                countdown = (f"{left} left, {int(eta // 3600)}h {int(eta % 3600 // 60):02d}m to go, "
                             f"done ~{datetime.fromtimestamp(time.time() + eta):%H:%M}")
                log(f"[{i}/{len(todo)}] {t.stem or t.url}  ({countdown})")
                try:
                    status, name, note = grab(ctx, page, t, dest, name_owner)
                except Exception as e:  # a dead page, a crashed renderer, a timeout
                    status, name, note = "failed", "", f"{type(e).__name__}: {str(e)[:200]}"

                final_path = str(dest / name) if name else ""
                if status == "ok" and lib is not None:
                    cstatus, final, cnote = curate_download(dest / name, t, lib)
                    if cstatus == "ok":
                        name, final_path, note = Path(final).name, final, f"{note}; {cnote}"
                    elif cstatus == "duplicate":   # recorded as done, so it is not fetched again
                        name, final_path, note = "", "", f"{note}; dropped: {cnote}"
                        run.skipped += 1
                    elif cstatus == "rejected":    # never fetched again; the song's next best version may be
                        status, name, final_path, note = "rejected", "", "", cnote
                        run.skipped += 1
                    else:
                        status, name, final_path, note = "failed", "", "", cnote

                entry = {
                    "status": status, "file": name, "note": note,
                    "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "path": final_path,
                }
                state[t.url] = entry
                save_state(state)
                append_log({"at": entry["at"], "tab_id": t.tab_id, "artist": t.artist, "song": t.song,
                            "version": t.version, "status": status, "button": "", "file": name, "note": note})
                write_progress(run, i, len(todo), t, elapsed)

                if status == "ok":
                    run.ok += 1
                    consecutive = 0
                    log(f"    ok: {name or note}")
                elif status == "stopped":
                    run.stopped = note
                    log(f"    STOP: {note}")
                    break
                elif status == "blocked":
                    # the site refuses this tab for good; an artist can have twenty of them in a row
                    run.blocked += 1
                    log(f"    blocked: {note}")
                elif status == "rejected":                     # a fine download the training would skip
                    consecutive = 0
                    log(f"    rejected: {note}")
                else:
                    run.failed += 1
                    consecutive += 0 if note.startswith("curate:") else 1   # one bad file is not a dead connection
                    log(f"    {status}: {note}")
                    if consecutive >= args.max_failures:
                        run.stopped = f"{consecutive} failures in a row, last: {note}"
                        log(f"    STOP: {run.stopped}")
                        break

                # pace: 2.5-5 s per tab, a longer breather every 100 / 1000 tabs
                time.sleep(random.uniform(*args.gap))
                if i % 1000 == 0:
                    pause = random.uniform(*args.pause_1000)
                    log(f"    long pause {pause:.0f} s (after {i} tabs)")
                    time.sleep(pause)
                elif i % 100 == 0:
                    pause = random.uniform(*args.pause_100)
                    log(f"    pause {pause:.0f} s (after {i} tabs)")
                    time.sleep(pause)
        except KeyboardInterrupt:
            log("interrupted")
        finally:
            log(f"done: {run.ok} ok, {run.blocked} blocked, {run.failed} failed, {run.skipped} skipped, "
                f"{time.time() - run.started:.0f}s")
            write_progress(run, 0, len(todo), None, time.time() - run.started, finished=True)
            ctx.close()
    return 0 if not run.stopped else 3


if __name__ == "__main__":
    raise SystemExit(main())
