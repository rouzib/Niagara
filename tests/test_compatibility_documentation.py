"""Check the rendered statuses in the Caskade compatibility guide."""

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs/source/user_guide/caskade_compatibility.md"
STYLES = ROOT / "docs/source/_static/custom.css"
STATUSES = {
    "compat-works": "Works",
    "compat-caution": "Needs care",
    "compat-fails": "Does not work",
    "compat-unverified": "Not tested",
}


class TableReader(HTMLParser):
    """Collect cell text and status spans from rendered table rows."""

    def __init__(self):
        super().__init__()
        self.rows = []
        self.row = None
        self.cell = None
        self.status = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in {"td", "th"} and self.row is not None:
            self.cell = {"text": "", "statuses": []}
        elif tag == "span" and self.cell is not None:
            classes = dict(attrs).get("class", "").split()
            if "compat-status" in classes:
                self.status = {"classes": classes, "text": ""}
                self.cell["statuses"].append(self.status)

    def handle_data(self, data):
        if self.cell is not None:
            self.cell["text"] += data
        if self.status is not None:
            self.status["text"] += data

    def handle_endtag(self, tag):
        if tag == "span":
            self.status = None
        elif tag in {"td", "th"} and self.cell is not None:
            self.row.append(self.cell)
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def test_rendered_compatibility_rows_have_one_status():
    """Render the actual Markdown tables through MyST and inspect their cells."""
    pytest.importorskip("myst_parser")
    from docutils.core import publish_parts
    from myst_parser.parsers.docutils_ import Parser

    source = GUIDE.read_text(encoding="utf-8")
    lines = source.splitlines()
    tables = []
    for index, line in enumerate(lines):
        if line.startswith("| Guide feature"):
            table = []
            for table_line in lines[index:]:
                if not table_line.startswith("|"):
                    break
                table.append(table_line)
            tables.append("\n".join(table))
    assert len(tables) == 2, "Expected the Advanced and Hierarchical guide tables"

    reader = TableReader()
    for table in tables:
        rendered = publish_parts(table, parser=Parser(), writer_name="html5")["fragment"]
        reader.feed(rendered)

    cases = 0
    for row in reader.rows:
        assert len(row) == 3, "Expected three cells per compatibility row"
        if row[1]["text"].strip() == "Status":
            continue
        name = row[0]["text"].strip()
        statuses = row[1]["statuses"]
        assert len(statuses) == 1, f"{name}: expected one status, got {len(statuses)}"
        status = statuses[0]
        classes = set(status["classes"]) & STATUSES.keys()
        assert len(classes) == 1, f"{name}: expected one recognized status class"
        expected = STATUSES[classes.pop()]
        assert status["text"].strip() == expected, f"{name}: incorrect status label"
        assert row[1]["text"].strip() == expected, f"{name}: extra text in status cell"
        cases += 1
    assert cases, "No rendered compatibility cases were found"


def test_compatibility_status_colors_are_distinct_in_both_themes():
    css = STYLES.read_text(encoding="utf-8")
    for prefix in ("", 'html[data-theme="dark"] '):
        colors = []
        backgrounds = []
        for name in STATUSES:
            selector = re.escape(prefix + "." + name)
            match = re.search(r"(?m)^" + selector + r"\s*\{([^{}]+)\}", css)
            assert match, f"Missing {prefix}{name} styles"
            declarations = dict(
                (key.strip(), value.strip().lower())
                for key, value in re.findall(r"([\w-]+)\s*:\s*([^;]+);", match.group(1))
            )
            colors.append(declarations["color"])
            backgrounds.append(declarations["background"])
        assert len(set(colors)) == len(STATUSES)
        assert len(set(backgrounds)) == len(STATUSES)
