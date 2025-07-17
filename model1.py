import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

import requests
import gradio as gr
import csv
import io
import re
from collections import Counter
from datetime import datetime

# ====== CONFIG =======
OPENROUTER_API_KEY = "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"
LLM_MODEL = "deepseek/deepseek-chat-v3-0324:free"
PAGE_SIZE = 20

PHYSICS_JOURNALS = [
    "Physical Review", "Journal of Chemical Physics", "Journal of Molecular Structure", "Journal of Quantitative Spectroscopy and Radiative Transfer",
    "Journal of Molecular Spectroscopy", "Astrophysical Journal", "Astronomy & Astrophysics", "Molecular Physics",
    "Chemical Physics Letters", "Journal of Physical Chemistry", "Journal of Geophysical Research", "Icarus",
    "Planetary and Space Science", "Optics Express", "Applied Optics", "Spectrochimica Acta Part A",
    "Journal of the Optical Society of America", "Chemical Physics",
    "Atmospheric Chemistry and Physics", "Journal of Atmospheric Sciences", "Physics Letters A",
    "Physical Chemistry Chemical Physics", "Review of Scientific Instruments", "Journal of Applied Physics",
    "Journal of Physics B", "Journal of Physics D", "Journal of Physics: Condensed Matter", "Physics of Fluids",
    "Physics of Plasmas", "Physics Today", "Nature Physics", "Science", "Physics Reports", "Physics Letters B",
    "Nuclear Physics", "Physical Review Letters", "Physical Review A", "Physical Review B", "Physical Review C",
    "Physical Review D", "Physical Review E", "Physical Review Applied", "Physical Review X", "Physical Review Fluids",
    "Physical Review Materials", "Physical Review Physics Education Research", "Physical Review Research",
    "Physical Review Accelerators and Beams", "Physical Review Special Topics - Accelerators and Beams"
]

PARAM_KEYWORDS = [
    "gamma", "γ", "pressure broadening coefficient", "temperature exponent", "n=", "γ=",
    "measured", "measurement", "experimental", "value", "fit", "uncertainty",
    "cm-1 atm-1", "reported", "determined", "parameters", "obtained", "tabulated", "table",
    "J =", "K =", "data", "coefficient", "result", "parameter", "shift", "pressure", "rotational",
    "collision", "dependence", "determination", "half-width", "linewidth", "spectra", "temperature", 
    "comparison", "collisional", "intensity", "narrowing", "vibration", "overtone", "band", "transition", 
    "calculation","absorption", "theory", "data"
]

def is_physics_journal(journal_name):
    if not journal_name:
        return False
    journal_lower = journal_name.lower()
    return any(physics_journal.lower() in journal_lower for physics_journal in PHYSICS_JOURNALS)

def highlight_keywords(text, keywords):
    if not text: return ""
    for kw in keywords:
        if kw.strip():
            pattern = r'\b' + re.escape(kw) + r'\b'
            text = re.sub(pattern, lambda m: f'<mark style="background:#ffe066;border-radius:3px;padding:1px 3px">{m.group(0)}</mark>', text, flags=re.IGNORECASE)
    return text

def get_key(entry):
    doi = (entry.get('DOI', '') or '').strip().lower()
    if doi:
        return "doi:" + doi
    title = (entry.get('Title', '') or '').strip().lower()
    authors = (entry.get('Authors', '') or '').strip().lower()
    short = "".join(authors.split()[:2])
    return "title:" + title + "|auth:" + short

def deduplicate_papers(papers):
    seen = set()
    unique = []
    for p in papers:
        key = get_key(p)
        if key not in seen:
            unique.append(p)
            seen.add(key)
    return unique

def search_crossref_multi(keywords, species, perturbers, max_results=500, min_year=2000, max_year=2025, physics_journals_only=False):
    all_results = []
    # For recall, combine all search terms into one big query
    query_terms = list(filter(None, keywords + species + perturbers))
    # At least use parameter keywords if user does not input
    if not query_terms:
        query_terms = PARAM_KEYWORDS
    query_str = " AND ".join(query_terms)
    params = {
        "query.bibliographic": query_str,
        "rows": max_results,
        "sort": "issued",
        "order": "desc"
    }
    try:
        resp = requests.get("https://api.crossref.org/works", params=params, timeout=60)
        items = resp.json().get("message", {}).get("items", [])
        for item in items:
            title = (item.get("title") or [""])[0] or ""
            authors = ", ".join(f"{p.get('given','')} {p.get('family','')}".strip() for p in item.get("author", []))
            pub = item.get("published-print", item.get("published-online", {}))
            year = pub.get("date-parts", [[None]])[0][0]
            try: year_val = int(year)
            except: year_val = None
            if min_year and year_val and year_val < min_year:
                continue
            if max_year and year_val and year_val > max_year:
                continue
            journal = (item.get("container-title") or [""])[0]
            if physics_journals_only and journal and not is_physics_journal(journal):
                continue
            doi = item.get("DOI", "")
            abstract = item.get("abstract", "")
            all_results.append({
                "Title": title,
                "Authors": authors,
                "Year": year,
                "Journal": journal,
                "DOI": doi,
                "Abstract": abstract
            })
    except Exception as e:
        print("CrossRef search error:", e)
    all_results = deduplicate_papers(all_results)
    return all_results

def llm_filter(entries, keywords, species, perturbers):
    filtered = []
    for entry in entries:
        # 宽松召回+高容错 prompt
        content = f"""Title: {entry['Title']}\nAbstract: {entry.get('Abstract','')}\nJournal: {entry.get('Journal','')}\n"""
        prompt = (
            "You are an expert in molecular spectroscopy. "
            "Does the following paper discuss, even possibly, any experimental data, measurements, or physical parameters "
            "related to molecular spectral line broadening (pressure broadening) or related quantities (e.g. gamma, n, shift, linewidth, cross section, etc)? "
            "If YES, even if it is uncertain or ambiguous, please answer 'Relevant: ...' with a brief summary. "
            "If the information is insufficient to decide, please ERR ON THE SIDE OF INCLUDING IT and answer as 'Relevant: ...'. "
            "Only answer 'Irrelevant' if you are absolutely sure the paper has NO connection to experimental/parameter data for spectral line broadening. "
            "----\n"
            f"{content}"
            "----\n"
            "Your answer should be:\n"
            "\"Relevant: \" + a one-sentence summary, or \"Irrelevant\". Only output this phrase."
        )
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "HTTP-Referer": "https://chat.openai.com",
                    "X-Title": "broadening-llm"
                },
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 128,
                    "temperature": 0.2,
                },
                timeout=30
            )
            ans = resp.json()["choices"][0]["message"]["content"]
            # 只要不是严格“irrelevant”都保留
            if ans.strip().lower().startswith("relevant"):
                entry["LLM"] = ans.replace("\n","")
                filtered.append(entry)
            elif "not enough" in ans.lower() or "cannot judge" in ans.lower() or "unsure" in ans.lower():
                entry["LLM"] = ans.replace("\n","")
                filtered.append(entry)
        except Exception as e:
            entry["LLM"] = f"LLM error: {str(e)}"
            # 筛选出错也保留
            filtered.append(entry)
    return filtered

def format_results_html(results, page=1, keywords=[]):
    total = len(results)
    start = (page-1)*PAGE_SIZE
    end = min(start+PAGE_SIZE, total)
    shown = results[start:end]
    html = """
    <style>
    .table-bib {border-collapse:collapse;width:99%;font-size:15px;}
    .table-bib th {background:#F5F5FA;padding:6px;}
    .table-bib td {padding:6px;}
    .table-bib tr:nth-child(even){background:#f9f9f9;}
    .table-bib tr:hover{background:#eef3ff;}
    .physics-journal {background-color: #e6f7ff !important;}
    </style>
    """
    html += '<div style="overflow-x:auto;max-width:100vw;">'
    html += '<table class="table-bib" border="1">'
    html += '<thead><tr>'
    for col in ["#", "Title", "Authors", "Year", "Journal", "DOI"]:
        html += f'<th>{col}</th>'
    html += '</tr></thead><tbody>'
    for i, r in enumerate(shown, start=start+1):
        journal_class = "physics-journal" if is_physics_journal(r["Journal"]) else ""
        html += f'<tr class="{journal_class}">'
        html += f'<td style="text-align:center;">{i}</td>'
        html += f'<td>{highlight_keywords(r["Title"], keywords)}</td>'
        author_short = (r["Authors"][:35] + ("…" if len(r["Authors"])>35 else "")) if r["Authors"] else ""
        html += f'<td title="{r["Authors"]}" style="max-width:150px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{author_short}</td>'
        html += f'<td style="text-align:center;">{r["Year"] or ""}</td>'
        html += f'<td>{r["Journal"]}</td>'
        doi_display = r["DOI"] or ""
        if doi_display:
            html += f'''<td><a href="https://doi.org/{doi_display}" target="_blank" style="color:#2d79c7;text-decoration:underline" onclick="navigator.clipboard.writeText('{doi_display}')">{doi_display}</a></td>'''
        else:
            html += "<td></td>"
        html += '</tr>'
    html += '</tbody></table>'
    html += f'<div style="padding:6px;text-align:right;color:#666">Total <b>{total}</b> entries, page <b>{page}</b> / <b>{(total-1)//PAGE_SIZE+1}</b></div>'
    html += f'<div style="padding:6px;background:#e6f7ff;border-radius:4px;margin-top:8px">Blue background: Physics journal</div>'
    html += '</div>'
    return html

def export_results_csv(results):
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=["Title", "Authors", "Year", "Journal", "DOI"])
    writer.writeheader()
    for r in results:
        writer.writerow({k: r.get(k, "") for k in ["Title", "Authors", "Year", "Journal", "DOI"]})
    return csv_buffer.getvalue().encode("utf-8")

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Spectral Line Broadening Literature Search (deepseek/deepseek-chat-v3-0324:free)
    <div style='color:#444;font-size:15px;margin-bottom:10px'>
    Use keywords, active species, perturbers and physics journal filter to maximize recall for pressure broadening parameter literature. Click DOI to copy. Download CSV for batch processing.
    </div>
    """)

    keywords_input = gr.Textbox(label="Search keywords (comma separated)", value="", placeholder="pressure broadening, gamma, shift, ...")
    species_input = gr.Textbox(label="Active Species (comma separated)", value="", placeholder="CO2, NO, H2O ...")
    perturber_input = gr.Textbox(label="Perturbers (comma separated)", value="", placeholder="N2, O2, He ...")
    max_results_input = gr.Number(label="Max results", value=500, precision=0)
    min_year_input = gr.Number(label="Earliest year", value=2000, precision=0)
    max_year_input = gr.Number(label="Latest year", value=2025, precision=0)
    journal_filter_checkbox = gr.Checkbox(label="Only physics journals", value=True)
    search_btn = gr.Button("Search", scale=1)
    prev_btn = gr.Button("Previous page", scale=1)
    next_btn = gr.Button("Next page", scale=1)
    csv_btn = gr.Button("Export as CSV", scale=2)
    csv_output = gr.File(label="", visible=False)
    result_html = gr.HTML()
    results_state = gr.State([])
    page_state = gr.State(1)
    log_output = gr.Textbox(label="Log", value="", lines=6)

    def do_search(kw, sp, pe, mx, miny, maxy, journal_filter):
        log_msgs = ["Starting CrossRef search..."]
        mx = int(mx) if mx else 500
        miny = int(miny) if miny else 2000
        maxy = int(maxy) if maxy else 2025
        keywords = [k.strip() for k in kw.split(",") if k.strip()]
        species = [s.strip() for s in sp.split(",") if s.strip()]
        perturber = [p.strip() for p in pe.split(",") if p.strip()]
        all_results = search_crossref_multi(keywords, species, perturber, max_results=mx, min_year=miny, max_year=maxy, physics_journals_only=journal_filter)
        log_msgs.append(f"Fetched {len(all_results)} records...")
        log_msgs.append("Applying LLM filter for experimental/parameter papers ...")
        filtered = llm_filter(all_results, keywords, species, perturber)
        log_msgs.append(f"After LLM filter: {len(filtered)} remain ...")
        html = format_results_html(filtered, 1, keywords + species + perturber)
        log_msgs.append("Done.")
        return html, filtered, 1, "\n".join(log_msgs)

    def turn_page(results, page, direction):
        page = page + direction
        total = len(results)
        maxpage = max(1, (total-1)//PAGE_SIZE+1)
        if page < 1: page = 1
        if page > maxpage: page = maxpage
        html = format_results_html(results, page)
        return html, page

    def export_csv(results):
        tempname = "broadening_results.csv"
        with open(tempname, "wb") as f:
            f.write(export_results_csv(results))
        return tempname

    search_btn.click(
        fn=do_search,
        inputs=[keywords_input, species_input, perturber_input, max_results_input, min_year_input, max_year_input, journal_filter_checkbox],
        outputs=[result_html, results_state, page_state, log_output]
    )

    prev_btn.click(
        lambda res, p: turn_page(res, p, -1),
        inputs=[results_state, page_state],
        outputs=[result_html, page_state]
    )
    next_btn.click(
        lambda res, p: turn_page(res, p, 1),
        inputs=[results_state, page_state],
        outputs=[result_html, page_state]
    )
    csv_btn.click(
        fn=export_csv,
        inputs=[results_state],
        outputs=[csv_output]
    )

if __name__ == "__main__":
    demo.launch()