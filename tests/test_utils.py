import os
os.environ.setdefault("OPENROUTER_API_KEY", "test")

from broadening_search import model1 as m

def test_priority_score():
    assert m.priority_score("Journal of Quantitative Spectroscopy and Radiative Transfer") == 0
    assert m.priority_score("Journal of Molecular Spectroscopy") == 1
    assert m.priority_score("Journal of Chemical Physics") == 2
    assert m.priority_score("Some Other Journal") == 999

def test_is_physics_journal():
    assert m.is_physics_journal("Physical Review X")
    assert m.is_physics_journal("Physical Review A")
    assert not m.is_physics_journal("Random Non-Physics Journal")

def test_strip_and_highlight():
    txt = "<p>Voigt line shape</p>"
    assert "Voigt line shape" in m.strip_tags(txt)
    html = m.highlight_keywords("pressure broadening and linewidth", ["linewidth"])
    assert "<mark" in html and "linewidth" in html

def test_dedup_key_and_deduplicate():
    a = {"Title":"A Study", "Authors":"Ada Lovelace", "Year":2020, "Journal":"JCP", "DOI":""}
    b = {"Title":"A Study", "Authors":"Ada Lovelace", "Year":2020, "Journal":"JCP", "DOI":""}
    c = {"Title":"Different", "Authors":"Alan Turing", "Year":2020, "Journal":"JCP", "DOI":"10.1000/xyz"}
    out = m.deduplicate_papers([a,b,c])
    assert len(out) == 2
