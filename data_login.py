#!/usr/bin/env python
"""Open the downloader's own browser profile so you can sign in to Ultimate Guitar by hand, once.

    python data_login.py                 # opens a window, waits until you are signed in, then closes

data_dl.py downloads with a persistent Chromium profile (data_encoded/profile). When that profile holds a
logged-in session, `python data_dl.py --no-login` needs no password and no UG_PASSWORD in the
environment - which is the point of this script: sign in yourself, in your own window, and nothing has to
hand your credentials to a script or a log.

It waits for up to --minutes, checking every few seconds whether the 'Log in' control is gone, and exits
0 as soon as it is. Closing the window yourself also ends it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_dl as dl  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=15.0, help="how long to wait for the sign-in (default 15)")
    ap.add_argument("--profile", default=str(dl.PROFILE))
    a = ap.parse_args(argv)

    from playwright.sync_api import sync_playwright

    profile = Path(a.profile)
    profile.mkdir(parents=True, exist_ok=True)
    dl.log(f"profile: {profile}")
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            str(profile), headless=False, accept_downloads=True,
            viewport={"width": 1360, "height": 900},
            args=[f"--user-agent={dl.UA}", "--disable-blink-features=AutomationControlled"],
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto("https://www.ultimate-guitar.com/", wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_timeout(3000)

        if not dl.login_controls(page) and page.query_selector("#login-form-password") is None:
            dl.log("already signed in (the profile kept its cookies) - nothing to do")
            ctx.close()
            return 0

        dl.log("SIGN IN IN THE WINDOW THAT JUST OPENED. Waiting...")
        deadline = time.time() + a.minutes * 60
        while time.time() < deadline:
            try:
                if not ctx.pages:
                    dl.log("window closed")
                    return 1
                page = ctx.pages[0]
                if not dl.login_controls(page) and page.query_selector("#login-form-password") is None:
                    dl.log("signed in - the profile now holds the session")
                    page.wait_for_timeout(2000)      # let the cookies be written
                    ctx.close()
                    return 0
            except Exception:
                pass                                  # mid-navigation, or the page is being replaced
            time.sleep(3)
        dl.log(f"not signed in after {a.minutes:g} minutes")
        ctx.close()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
