# UGDownloader (jabbey1) — summary

Cloned to `UGDownloader/` (commit 59abf4a, last commit 2024-12-23, app version 2.6). Read only; nothing from it was run.
Third-party code with **no LICENSE file** (nothing in setup.cfg, pyproject.toml or setup.py either), so it is
"all rights reserved" by default.

## What it is
A Windows desktop app (customtkinter GUI, packaged with PyInstaller) that drives a real browser with Selenium.
You enter your Ultimate Guitar username, password and an artist; it logs in, opens the artist's tab lists and
downloads **every user-created tab** of the chosen type, one after the other.

| Feature | Detail |
|---|---|
| File types | Guitar Pro (.gp3/.gp4/.gp5/.gpx/.gp), Power Tab, text (tab, chords, bass, ukulele); "Official" tabs are skipped |
| Search by | Artist, a UG user (everything that user created), or "My Tabs" (your saved tabs) |
| Other | "Check Artist Tab Count" (online vs. local count), a saved list of artists to download later, Chrome or Firefox, headless mode, optional "I don't care about cookies" extension |
| File names | `Artist - Song (V1, 4.8@6).gpx` = version, rating @ number of votes (read from each tab page) |
| Stack | selenium 4.16, webdriver_manager, seleniumbase, customtkinter, requests, psutil |

## How it works (from the code)
1. `DriverSetup.py` starts Chrome/Firefox with the download folder preset (images and stylesheets switched off).
2. `Utils.login` clicks the site's login button and types user and password into the form (a CAPTCHA defeats it).
3. `DLoader.search_for_artist` opens `ultimate-guitar.com/search.php?search_type=bands&value=<artist>` and
   clicks the exact artist name (exact spelling and capitalization needed).
4. `DLoader.link_handler` clicks the type filter (e.g. "Guitar Pro"), walks every page of the list and collects tab URLs.
5. `Tab.read_info` opens each tab page to read version, rating, tuning, author, last edit.
6. `DLoader.download_tab` opens each tab and clicks the Download button with JavaScript; if the tab is "blocked"
   it opens `tabs.ultimate-guitar.com/tab/download?id=<id>` directly.
7. Files are renamed afterwards; progress is shown in the GUI.

## Things worth knowing
- **No rating filter.** The rating only goes into the file name. For your 4.4+ rule you would have to change it
  (or keep using our shortlist tool and download by hand).
- **No rate limiting.** The only pauses are 0.5 s after a download and a few 1–2 s waits when logging in. Its README says
  it "moves quicker than a human could".
- **It uses the pages our tool deliberately avoids.** Ultimate Guitar's robots.txt disallows `/search.php` and
  `/tab/download` for automated access; this tool uses both, and logs in with your account. That is the
  reason I did not build an automatic downloader (see the `ultimate_guitar` tool in this repo). Using it risks
  your account.
- **Password stored in plain text:** the GUI's "save" button writes `username password` to
  `_UGDownloaderFiles/userinfo.txt`.
- **Fragile:** the page elements are found by obfuscated CSS class names (e.g. `rPQkl yDkT4 ...`) kept in
  `_UGDownloaderFiles/config.ini`; they break whenever the site is redeployed and have to be edited by hand.
- **Windows only:** `DriverSetup.py` imports `CREATE_NO_WINDOW` from `subprocess`, which exists only on Windows,
  so it will not even start on this Linux machine without edits.
- **Binary blobs:** two browser-extension files (`extension_3_4_6_0.crx`, `extension_3_5_0_0.crx`, ~430 KB each)
  are bundled; I did not inspect them.
- **Auto-update check:** `Utils.check_update` calls the GitHub releases API of the project.

## Using it for RiffGen
Not recommended as is: it downloads everything (no rating filter), ignores the site's robots.txt, and the
files would still need `sort`-style renaming and the GP5 export. The only idea worth borrowing is its
file-name convention (`(V1, 4.8@6)`), which our `sort` step could read.
