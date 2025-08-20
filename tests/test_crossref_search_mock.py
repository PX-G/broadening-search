import os
os.environ.setdefault("OPENROUTER_API_KEY", "test")

from broadening_search import model1 as m

def test_search_crossref_multi_uses_filters(monkeypatch):
    # Mock low-level Crossref call to avoid network
    def fake_call(params):
        # emulate two items; one irrelevant by soft-neg, one relevant
        if "type:journal-article" not in params.get("filter",""):
            raise AssertionError("missing type filter")
        return [
            {
                "title":["Gamma ray statistics in collider"],  # should be filtered by SOFT_NEG
                "author":[{"given":"A","family":"X"}],
                "issued":{"date-parts":[[2021]]},
                "container-title":["Random Physics"],
                "DOI":"10.1000/neg",
                "abstract":"Study of gamma ray in collider.",
                "created":{"date-parts":[[2021]]},
            },
            {
                "title":["Voigt linewidths in CO2"],  # should remain
                "author":[{"given":"B","family":"Y"}],
                "issued":{"date-parts":[[2022]]},
                "container-title":["Journal of Chemical Physics"],
                "DOI":"10.1000/pos",
                "abstract":"Lorentz/Voigt profiles and HWHM.",
                "created":{"date-parts":[[2022]]},
            }
        ]
    monkeypatch.setattr(m, "_crossref_call", fake_call)

    out = m.search_crossref_multi(keywords=["linewidth"], species=["CO2"], perturbers=["N2"], max_results=5, min_year=2000, max_year=2025)
    titles = [r["Title"] for r in out]
    assert any("Voigt linewidths" in t for t in titles)
    assert all("Gamma ray" not in t for t in titles)
