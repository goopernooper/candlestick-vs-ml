"""Assemble the paper sections into one document.

Usage
-----
    python paper/build.py

Concatenates paper/00_*.md ... paper/10_*.md in order into paper/draft.md.
Each section keeps its own "References for this section" list while drafting;
here those lists are removed and merged into a single alphabetical reference
list at the end, with duplicates dropped. Figure paths stay relative to paper/.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF_HEADING = re.compile(r"^### References for (this section|these sections)\s*$", re.M)


def split_references(text: str) -> tuple[str, list[str]]:
    """Return the section body without its reference list, and the list's entries."""
    match = REF_HEADING.search(text)
    if not match:
        return text.rstrip() + "\n", []
    body, refs = text[:match.start()], text[match.end():]
    entries = [line[2:].strip() for line in refs.splitlines() if line.startswith("- ")]
    return body.rstrip() + "\n", entries


def sort_key(entry: str) -> str:
    plain = re.sub(r"[*_]", "", entry).lower()
    return re.sub(r"[^a-z0-9 ]", "", plain)


def merge(entries: list[str]) -> list[str]:
    """Drop duplicates (same authors and year, ignoring formatting) and sort."""
    seen, kept = set(), []
    for entry in entries:
        key = sort_key(entry)[:60]
        if key not in seen:
            seen.add(key)
            kept.append(entry)
    return sorted(kept, key=sort_key)


def build(folder: Path = HERE) -> str:
    sections = sorted(p for p in folder.glob("[0-9][0-9]_*.md"))
    bodies, refs = [], []
    for path in sections:
        body, entries = split_references(path.read_text())
        bodies.append(body)
        refs.extend(entries)
    reference_list = "## References\n\n" + "\n".join(f"- {r}" for r in merge(refs)) + "\n"
    return "\n".join(bodies) + "\n" + reference_list


def main() -> int:
    text = build()
    out = HERE / "draft.md"
    out.write_text(text)
    words = len(re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text).split())
    print(f"Wrote paper/draft.md ({words:,} words)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
