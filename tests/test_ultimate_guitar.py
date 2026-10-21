"""The Ultimate Guitar helper, offline: parsing an artist page, the shortlist, sorting downloads."""

import html
import json

from riffgen.tools import ultimate_guitar as ug


def _page(tabs):
    data = {"store": {"page": {"data": {"other_tabs": tabs, "album_tabs": [], "pagination": {"pages": []}}}}}
    return f'<div class="js-store" data-content="{html.escape(json.dumps(data))}"></div>'


def _tab(i, song, version, rating, votes, kind="Pro", access="public"):
    return {"id": i, "song_name": song, "artist_name": "Band", "type": kind, "version": version, "rating": rating,
            "votes": votes, "tab_access_type": access, "tuning": "Drop D",
            "tab_url": f"https://tabs.ultimate-guitar.com/tab/band/x-guitar-pro-{i}"}


def test_shortlist_keeps_free_community_tabs_above_rating():
    tabs = [_tab(1, "Riff", 1, 4.5, 10), _tab(2, "Riff", 2, 4.8, 3), _tab(3, "Other", 1, 3.9, 50),
            _tab(4, "Paid", 1, 4.9, 99, kind="TabPro"), _tab(5, "Official", 1, 4.9, 99, kind="official")]
    found = ug.community_gp_tabs(ug.page_data(_page(tabs)))
    assert [t["id"] for t in found] == [1, 2, 3]
    assert [r["id"] for r in ug.shortlist(found, 4.0, 1, False)] == [1, 2]
    assert [r["id"] for r in ug.shortlist(found, 4.0, 1, True)] == [2]


def test_sort_names_versions_as_song_family(tmp_path):
    rows = ug.shortlist(ug.community_gp_tabs(ug.page_data(_page([_tab(1, "Riff", 1, 4.5, 10),
                                                                   _tab(2, "Riff", 2, 4.8, 3)]))), 4.0, 1, False)
    ug.write_outputs(rows, tmp_path, "test")
    dl = tmp_path / "dl"
    dl.mkdir()
    (dl / "Band - Riff (ver 2).gp5").write_bytes(b"x")
    ug.main(["sort", "--shortlist", str(tmp_path / "shortlist.json"), "--downloads", str(dl),
             "--dest", str(tmp_path / "album")])
    assert (tmp_path / "album" / "Band" / "Band - Riff__v2.gp5").is_file()
    status = {r["id"]: r["status"] for r in json.loads((tmp_path / "shortlist.json").read_text())}
    assert status == {1: "to download", 2: "sorted"}
