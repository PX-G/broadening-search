import os
os.environ.setdefault("OPENROUTER_API_KEY", "test")

from broadening_search import model1 as m

def test_export_csv_and_bibtex(tmp_path):
    results = [
        {"Title":"Linewidths in CO2", "Authors":"Ada Lovelace", "Year":2021, "Journal":"JQSRT", "DOI":"10.1000/aa"},
        {"Title":"Pressure broadening of NO", "Authors":"Alan Turing", "Year":2020, "Journal":"JMS", "DOI":""},
    ]
    csv_path = tmp_path / "out.csv"
    bib_path = tmp_path / "out.bib"
    p1 = m.export_results_csv(results, str(csv_path))
    p2 = m.export_results_bibtex(results, str(bib_path))
    assert os.path.exists(p1) and os.path.getsize(p1) > 0
    assert os.path.exists(p2) and os.path.getsize(p2) > 0
