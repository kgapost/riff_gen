# Album assignment tool — design and handoff

Assign any of the ~61k corpus tabs to one of my nine albums as inspiration, alongside my own songs,
through a web UI a bandmate can also use. Agreed 2026-11-07/08 on the work PC (`iti-1261`); the server
is to be built on the home PC, which is why this document exists in the repo rather than in a session.

## Decisions already made — do not reopen

- **The nine albums are final**: `cs, caffeine, waves, vc1, vc2, potha, tme, maphorion, omni`, each a
  `data/__<album>/` folder holding 10–20 of my own tabs and 50–100 inspiration tabs.
- **The folders on disk are the only truth.** No sidecar labelling scheme. Copying a file into an album
  folder is how a tab is assigned, which is why the content-digest dedup was removed.
- `__<album>/own/` = mine, the model may copy it. `__<album>/insp/` = inspiration, the model may learn
  from it but must not reproduce it.
- **Inspiration copies do not keep their genre label.** Asked and answered: no.
- **No rejection or dismiss list in the UI.** I would rather re-see candidates as my taste for each
  album firms up.
- Apply by **copying** files, not hardlinking.
- **My bandmate has no Claude account.** An Artifact therefore cannot hold the shared state: both the
  `db` and `artifact` capabilities need a signed-in account with write rights, and a public-link visitor
  gets `null`. The UI must be self-hosted and reachable over a tunnel.
- One shared state file, saved on demand, restored on load, **last write wins**. No concurrent access —
  we coordinate by talking. Assignments are applied to the folders once, at the end.

## Already implemented and committed (3b755d0)

Label scheme and vocabulary in `riff.py`: `__<album>/own|insp/` → `src:own` / `src:insp`; the label
vocabulary built from the `ALBUMS` / `GENRES` / `SOURCES` constants so it never changes as folders fill;
`genre:other` fallback; digest dedup removed with `--max-versions` scoped per top-level folder; exact
copies marked `dup` and kept out of the fixed eval gaps; guards that stop `prepare` on an album folder
not in `ALBUMS` and on holdout songs outside `__<album>/own/holdout/`.

## What the home PC needs — the index, not the tabs

`data/` is gitignored (correctly), so a fresh clone has none of it. But the server only reads the
**encoded index**, never the `.gp5` files:

| file | size | what it is for |
|---|---|---|
| `data/encoded/meta.json` | 12 M | 56,220 song records: `file` (→ artist/title for search), `labels`, `tempo`, `nb`, `bar0`, `tokens` |
| `data/encoded/bar_off.npy` | 48 M | bar offsets — bar lengths, and indexing into `tokens.bin` |
| `data/encoded/bar_hash.npy` | 48 M | bar identity — finds a song's most-repeated bar (its riff) |
| `data/encoded/bar_pc.npy` | 12 M | 12-bin chroma per bar — fingerprint glyph and similarity |
| `data/encoded/bar_on.npy` | 12 M | onset pattern per bar — fingerprint glyph and similarity |
| `data/encoded/bar_gn.npy` | 12 M | guitar-note count per bar — density and "riffiness" facets |
| `data/encoded/bar_low.npy` | 6 M | lowest string — tuning facet |
| `data/encoded/bar_sec.npy` | 6 M | section marks |
| `data/encoded/stats.json` | 4 K | corpus totals |
| `data/encoded/tokens.bin` | 797 M | the notes themselves — only needed to render a notation snippet |

~156 MB without `tokens.bin`, ~950 MB with it. Run `scripts/export_index.sh <dest>` on the work PC
(`--no-snippets` to skip `tokens.bin`) and move `<dest>` to the home PC as `data/encoded/`.

**The tab files stay on the work PC.** The UI writes assignments as corpus-relative paths; the copying
happens where the tabs are. So the loop is: assign at home → commit `configs/albums/assignments.json`
→ pull on the work PC → `riff.py albums apply` there.

Server dependencies are light: `numpy` and a web framework. Not `torch`, not `PyGuitarPro` (nothing
parses a `.gp5`; the index is already encoded).

## The server to build

A small Python app serving a static page, exposed through a Cloudflare tunnel with a shared token in the
URL. **It must not be reachable without the token** — it goes on the public internet.

State: `configs/albums/assignments.json`, written atomically (temp file + rename) with a timestamped
backup on each save, so last-write-wins never silently destroys an afternoon's work.

```json
{"version": 1, "savedAt": "...", "savedBy": "kostas",
 "albums": {"cs": {"own": ["<corpus-relative path>"], "insp": ["..."]}}}
```

The UI cannot browse 61k tabs, and any tab may go to any album, so it is search-driven, not
browse-driven:

- **Search first** — substring over 56k `Artist - Title` strings, server-side, instant.
- **Facets with live counts** — genre, tuning, tempo band, bars, density. Return counts plus one page of
  results; never a 61k list.
- **More-like-this** — nearest neighbours by chroma + onset + tuning + tempo, seeded from any tab. This
  is how 10 tabs I know become the 75 I need.
- **Contact sheet** — virtualised card grid, each card a fingerprint glyph from `bar_pc`/`bar_on` plus a
  snippet of the song's most-repeated bar.
- **One-key assignment** — `1…9` album, `O` own, `I` inspiration. A tab may go to several albums.
- **Album panel** — the nine albums with own/insp counts against the 10–20 / 50–100 targets.

## Still to build after the UI

- `riff.py albums apply` — copy assigned files into `__<album>/own|insp/`, with `--dry-run` and
  `--prune`. Runs on the work PC.
- **Per-source plagiarism tiers** — `src:own` report-only, `src:insp` relaxed, rest of the corpus strict;
  which tier applies is decided per album by the folders. `configs/plagiarism.yaml` holds the thresholds
  but **nothing in riff.py implements any of it yet** — the only existing guard is the per-song bar
  repeat check behind `--allow-copy`. See README §15.
- `riff.py refinetune` — retrain on my own tabs as they evolve. Possible only because the label
  vocabulary is frozen, so `--init` keeps accepting older checkpoints.

## Watch out

- A grid search was running on the work PC (`runs/grid2`, ~36 runs). **Do not re-prepare
  `data/encoded_grid`** while it runs — it would change the gaps and void the result table.
- `riff.py` and the encoded corpora are version-coupled; check for a busy peer session before long jobs.
