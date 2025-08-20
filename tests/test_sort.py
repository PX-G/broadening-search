import os
os.environ.setdefault("OPENROUTER_API_KEY", "test")

from broadening_search import model1 as m

def test_sort_results_stable_priority_then_year_then_title():
    rows = [
        {"Title":"zeta", "Journal":"Random", "Year":2021},
        {"Title":"alpha", "Journal":"Journal of Chemical Physics", "Year":2019},  # prio=2
        {"Title":"beta", "Journal":"Journal of Molecular Spectroscopy", "Year":2015},  # prio=1
        {"Title":"gamma", "Journal":"Journal of Quantitative Spectroscopy and Radiative Transfer", "Year":2010},  # prio=0
    ]
    out = m.sort_results_stable(rows, year_order="desc", prioritize=True)
    # Expected order by priority (0 -> 1 -> 2) then year desc for ties:
    assert out[0]["Journal"].startswith("Journal of Quantitative")
    assert out[1]["Journal"].startswith("Journal of Molecular")
    assert out[2]["Journal"].startswith("Journal of Chemical")
