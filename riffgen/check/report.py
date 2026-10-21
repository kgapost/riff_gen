"""check_report.md / check_report.csv writers (generation and the folder check)."""

from __future__ import annotations

import csv
from pathlib import Path

FIELDS = ["file", "plagiarism", "level", "source", "source_bars", "cand_bars", "run_bars", "coverage",
          "rhythm_like", "playability", "errors", "warnings", "fixed_file", "notes"]


def write_reports(folder: str | Path, rows: list[dict], title: str, issues: dict[str, list] | None = None) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "check_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    lines = [f"# {title}", "", "Plagiarism: GREEN ok, YELLOW close (kept, flagged), RED copy (removed during generation). "
             "SELF-QUOTE = the match is one of your own songs. A technical check, not a legal opinion.", "",
             "| file | plagiarism | source | playability | errors | warnings | notes |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        src = f"{r.get('source', '')} bars {r.get('source_bars', '')} ({r.get('level', '')})" if r.get("source") else ""
        fixed = f" -> {r['fixed_file']}" if r.get("fixed_file") else ""
        lines.append(f"| {r['file']} | {r.get('plagiarism', '')} | {src} | {r.get('playability', '')} | "
                     f"{r.get('errors', '')} | {r.get('warnings', '')} | {r.get('notes', '')}{fixed} |")
    if issues:
        lines += ["", "## Playability details", ""]
        for name, lst in issues.items():
            if not lst:
                continue
            lines.append(f"### {name}")
            for i in lst[:40]:
                lines.append(f"- {i.severity} bar {i.bar} beat {i.beat:.2f} (guitar {i.track}): {i.msg}")
            if len(lst) > 40:
                lines.append(f"- ... {len(lst) - 40} more")
            lines.append("")
    (folder / "check_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
