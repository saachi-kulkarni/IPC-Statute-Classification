"""Regression tests for the shared citation parser (src/common.py)."""
import pytest

from common import cited_sections, first_citation_word


@pytest.mark.parametrize("text, expected", [
    ("convicted under Section302 read with Section 34 of the IPC", {"302", "34"}),
    ("u/s 147, 148, 149 and 307 IPC", {"147", "148", "149", "307"}),
    ("offence punishable under 498-A IPC", {"498A"}),
    ("Sections 420/468/471 I.P.C.", {"420", "468", "471"}),          # no spurious '471I'
    ("converted from Section 302 I.P.C to Section 304, Part II", {"302", "304"}),
    ("under section 120-B", {"120B"}),
    ("Sections 323, 506 IPC", {"323", "506"}),                       # every number in a list
    ("sections of the Act were complied with", set()),
])
def test_cited_sections(text, expected):
    assert cited_sections(text) == expected


def test_first_citation_word():
    text = "The accused was arrested. Later he was charged under Sections 323, 506 IPC."
    assert first_citation_word(text, {"506"}) == 9      # 'Sections' is the 10th word (0-based 9)
    assert first_citation_word(text, {"302"}) is None
