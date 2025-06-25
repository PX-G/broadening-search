import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

import bibtexparser
import requests
import gradio as gr
import csv
import io
import re
from collections import Counter
from datetime import datetime

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

def is_chemical_formula(text):
    return re.match(r'^[A-Z][A-Z0-9_]*$', text) is not None

def highlight_keywords(text, keywords):
    if not text: return ""
    all_kws = list(set([k.lower() for k in keywords] + [k.lower() for k in PARAM_KEYWORDS]))

    def chemical_replacer(match):
        word = match.group(0)
        return f'<mark style="background:#ff9999;border-radius:3px;padding:1px 3px">{word}</mark>'

    def normal_replacer(match):
        word = match.group(0)
        return f'<mark style="background:#ffe066;border-radius:3px;padding:1px 3px">{word}</mark>'

    chemical_formulas = [kw for kw in all_kws if is_chemical_formula(kw)]
    for cf in chemical_formulas:
        pattern = r'\b' + re.escape(cf) + r'\b'
        text = re.sub(pattern, chemical_replacer, text)

    normal_keywords = [kw for kw in all_kws if not is_chemical_formula(kw)]
    for kw in normal_keywords:
        if kw.strip():
            pattern = r'\b' + re.escape(kw) + r'\b'
            text = re.sub(pattern, normal_replacer, text, flags=re.IGNORECASE)

    return text

def has_param_info(entry, param_keywords=PARAM_KEYWORDS):
    content = " ".join(str(entry.get(k, "")) for k in ['title','abstract','keywords','note'])
    return any(re.search(r'\b' + re.escape(pk) + r'\b', content, re.IGNORECASE) for pk in param_keywords)

def bib_match_entries(bibstr, keywords=[], species=[], perturber=[], min_year=None, max_year=None, journal_filter=False):
    bib_database = bibtexparser.loads(bibstr)
    matches = []
    species = [s.upper() if is_chemical_formula(s) else s for s in species]
    for entry in bib_database.entries:
        content = " ".join(str(entry.get(k, "")).lower() for k in ['title','abstract','keywords','note'])
        content_original = " ".join(str(entry.get(k, "")) for k in ['title','abstract','keywords','note'])

        # Keyword match
        if keywords:
            kw_found = False
            for kw in keywords:
                pattern = r'\b' + re.escape(kw.lower()) + r'\b'
                if re.search(pattern, content):
                    kw_found = True
                    break
            if not kw_found:
                continue
        # Species match
        if species:
            species_found = False
            for s in species:
                if is_chemical_formula(s):
                    pattern = r'\b' + re.escape(s) + r'\b'
                    if re.search(pattern, content_original):
                        species_found = True
                        break
                else:
                    pattern = r'\b' + re.escape(s.lower()) + r'\b'
                    if re.search(pattern, content):
                        species_found = True
                        break
            if not species_found:
                continue
        # Perturber match
        if perturber:
            perturber_found = False
            for p in perturber:
                pattern = r'\b' + re.escape(p.lower()) + r'\b'
                if re.search(pattern, content):
                    perturber_found = True
                    break
            if not perturber_found:
                continue
        year = entry.get('year','')
        try:
            year_val = int(year)
        except:
            year_val = None
        if min_year and year_val and year_val < min_year:
            continue
        if max_year and year_val and year_val > max_year:
            continue
        journal = entry.get('journal','') or entry.get('booktitle','')
        if journal_filter and journal and not is_physics_journal(journal):
            continue
        if not has_param_info(entry):
            continue
        matches.append({
            "Title": entry.get('title',''),
            "Authors": entry.get('author',''),
            "Year": year,
            "Journal": journal,
            "DOI": entry.get('doi',''),
            "Source": "BIB",
        })
    return matches

def has_param_info_crossref(item, param_keywords=PARAM_KEYWORDS):
    title = (item.get("title") or [""])[0]
    abstract = item.get("abstract", "")
    text = (title + " " + abstract).lower()
    return any(re.search(r'\b' + re.escape(pk.lower()) + r'\b', text) for pk in param_keywords)

def search_crossref(query, max_results=1000, min_year=None, max_year=None, journal_filter=False, journal_priority_list=None):
    papers = []
    batch = 1000
    offset = 0
    while len(papers) < max_results:
        rows = min(batch, max_results - len(papers))
        try:
            resp = requests.get(
                "https://api.crossref.org/works",
                params={
                    "query.bibliographic": query,
                    "rows": rows,
                    "offset": offset,
                    "sort": "issued",
                    "order": "desc"
                },
                timeout=30
            )
            items = resp.json().get("message", {}).get("items", [])
            for item in items:
                if not has_param_info_crossref(item):
                    continue
                title = (item.get("title") or [""])[0] or ""
                abstract = item.get("abstract", "") or ""
                authors = ", ".join(f"{p.get('given','')} {p.get('family','')}".strip() for p in item.get("author", []))
                pub = item.get("published-print", item.get("published-online", {}))
                year = pub.get("date-parts", [[None]])[0][0]
                try:
                    year_val = int(year)
                except:
                    year_val = None
                if min_year and year_val and year_val < min_year:
                    continue
                if max_year and year_val and year_val > max_year:
                    continue
                journal = (item.get("container-title") or [""])[0]
                if journal_filter and journal and not is_physics_journal(journal):
                    continue
                papers.append({
                    "Title": title,
                    "Authors": authors,
                    "Year": year,
                    "Journal": journal,
                    "DOI": item.get("DOI", ""),
                    "Source": "CrossRef"
                })
            if not items or len(items) < rows:
                break
            offset += rows
        except Exception as e:
            print("CrossRef error:", e)
            break
    # Sort results by journal priority if provided
    if journal_priority_list:
        papers = sort_by_journal_priority(papers, journal_priority_list)
    return papers

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

def bibfile_to_str(bibfile):
    if bibfile is None:
        return None
    try:
        if hasattr(bibfile, "read"):
            bibfile.seek(0)
            data = bibfile.read()
            if isinstance(data, bytes):
                return data.decode("utf-8")
            else:
                return data
        elif hasattr(bibfile, "value"):
            return bibfile.value
        elif isinstance(bibfile, bytes):
            return bibfile.decode("utf-8")
        elif isinstance(bibfile, str):
            return bibfile
        else:
            return None
    except Exception as e:
        print("Failed to read bib file:", e)
        return None

def extract_keywords_and_journal_priority(bibstr, topn_keywords=15):
    bib_database = bibtexparser.loads(bibstr)
    title_words = []
    journals = []
    stopwords = set([
        "the", "of", "in", "for", "and", "to", "on", "a", "with", "as", "by", "at",
        "an", "from", "study", "line", "lines", "broadening", "parameters", "rotational",
        "pressure", "dependence", "determination", "measurement", "coefficients",
        "spectroscopy", "molecular", "collision", "quantum"
    ])
    for entry in bib_database.entries:
        title = entry.get('title', '')
        words = re.findall(r'\b[a-zA-Z0-9]{3,}\b', title.lower())
        words = [w for w in words if w not in stopwords]
        title_words.extend(words)
        journal = entry.get('journal', '') or entry.get('booktitle', '')
        if journal:
            journals.append(journal.strip())
    keyword_freq = Counter(title_words)
    journal_freq = Counter(journals)
    top_keywords = [w for w, _ in keyword_freq.most_common(topn_keywords)]
    journal_priority = [j for j, _ in journal_freq.most_common()]
    return top_keywords, journal_priority

def sort_by_journal_priority(results, journal_priority_list):
    journal2rank = {j.lower(): i for i, j in enumerate(journal_priority_list)}
    def get_rank(r):
        j = r.get("Journal", "").lower()
        return journal2rank.get(j, len(journal_priority_list))
    return sorted(results, key=get_rank)

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
    for col in ["#", "Title", "Authors", "Year", "Journal", "DOI", "Source"]:
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
        html += f'<td style="text-align:center;">{r["Source"]}</td>'
        html += '</tr>'
    html += '</tbody></table>'
    html += f'<div style="padding:6px;text-align:right;color:#666">Total <b>{total}</b> entries, page <b>{page}</b> / <b>{(total-1)//PAGE_SIZE+1}</b></div>'
    html += f'<div style="padding:6px;background:#e6f7ff;border-radius:4px;margin-top:8px">Blue background: Physics journal</div>'
    html += '</div>'
    return html

def export_results_csv(results):
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=["Title", "Authors", "Year", "Journal", "DOI", "Source"])
    writer.writeheader()
    for r in results:
        writer.writerow(r)
    return csv_buffer.getvalue().encode("utf-8")

CUR_YEAR = datetime.now().year
DEFAULT_MIN_YEAR = CUR_YEAR - 9
DEFAULT_MAX_YEAR = CUR_YEAR

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""
    <h2 style='color:#385b95;font-weight:700'>Molecular Pressure Broadening Parameters Literature Screening (auto-high-frequency keywords & journal ranking)</h2>
    <div style='color:#777;font-size:15px;margin-bottom:10px'>
    By default, only the latest 10 years are searched. After uploading a bib file, recommended keywords and journal frequency are auto-extracted for sorting. Pagination, smart keyword highlighting. Export CSV for batch processing. Click DOI to copy.<br>
    <b style='color:#e85c00'>Only papers with experimental/parameter data are shown. Duplicates removed.</b>
    <div style='margin-top:8px;color:#d35400'>
    <b>Active Species (e.g., NO, CO2) will be treated as chemical formulas (case sensitive) for precise matching</b>
    </div>
    <div style='margin-top:8px;background:#e6f7ff;padding:8px;border-radius:4px'>
    <b>Results are sorted by the most frequent journals in your library. Physics journals are highlighted in blue.</b>
    </div>
    </div>
    """)

    bib_input = gr.File(label="Upload local BibTeX file (recommended)", file_types=[".bib"])
    keywords_input = gr.Textbox(label="Search keywords (comma separated)", value="", scale=3)
    species_input = gr.Textbox(label="Active Species (comma separated)", value="NO", scale=2)
    perturber_input = gr.Textbox(label="Perturbers (comma separated)", value="N2", scale=2)
    with gr.Row():
        min_year_input = gr.Number(label="Earliest year (optional)", value=DEFAULT_MIN_YEAR, precision=0, scale=1)
        max_year_input = gr.Number(label="Latest year (optional)", value=DEFAULT_MAX_YEAR, precision=0, scale=1)
        max_results_input = gr.Number(label="Max results (≤5000 suggested)", value=1000, precision=0, scale=2)
    journal_filter_checkbox = gr.Checkbox(label="Only physics journals", value=True)
    search_btn = gr.Button("Search", scale=1)
    with gr.Row():
        prev_btn = gr.Button("Previous page", scale=1)
        next_btn = gr.Button("Next page", scale=1)
        csv_btn = gr.Button("Export as CSV", scale=2)
        csv_output = gr.File(label="", visible=False)
    result_html = gr.HTML()
    results_state = gr.State([])
    page_state = gr.State(1)
    query_kw_state = gr.State([])

    def do_search(bibfile, kw, sp, pe, miny, maxy, mx, journal_filter):
        # 1. 解析本地bib文件
        bibstr = bibfile_to_str(bibfile)
        bib_results = []
        keywords = [k.strip() for k in kw.split(",") if k.strip()]
        species = [s.strip() for s in sp.split(",") if s.strip()]
        perturber = [p.strip() for p in pe.split(",") if p.strip()]
        try:
            miny = int(miny) if miny else DEFAULT_MIN_YEAR
            maxy = int(maxy) if maxy else DEFAULT_MAX_YEAR
        except: miny, maxy = DEFAULT_MIN_YEAR, DEFAULT_MAX_YEAR
        try:
            mx = int(mx) if mx else 1000
        except: mx = 1000
        # --- 本地bib优先 ---
        if bibstr:
            bib_results = bib_match_entries(bibstr, keywords, species, perturber, miny, maxy, journal_filter)
            if bib_results:
                bib_results = deduplicate_papers(bib_results)
                html = format_results_html(bib_results, 1, keywords + species + perturber)
                return html, bib_results, 1, keywords + species + perturber
        # --- 仅无本地结果时，才查CrossRef ---
        query_str = " AND ".join(keywords + species + perturber)
        cr_results = search_crossref(query_str, mx, miny, maxy, journal_filter)
        # 只保留title/abstract含有“broadening”/“pressure”/“collision”/“rotational”等的论文
        FILTER_TERMS = ['broadening', 'pressure', 'collision', 'rotational', 'linewidth', 'absorption', 'shift']
        def is_relevant(item):
            txt = (item['Title'] + " " + item.get('Journal',"")).lower()
            return any(t in txt for t in FILTER_TERMS)
        cr_results = [r for r in cr_results if is_relevant(r)]
        cr_results = deduplicate_papers(cr_results)
        html = format_results_html(cr_results, 1, keywords + species + perturber)
        return html, cr_results, 1, keywords + species + perturber

    def turn_page(results, page, keywords, direction):
        page = page + direction
        total = len(results)
        maxpage = max(1, (total-1)//PAGE_SIZE+1)
        if page < 1: page = 1
        if page > maxpage: page = maxpage
        html = format_results_html(results, page, keywords)
        return html, page

    def export_csv(results):
        tempname = "physics_journal_results.csv"
        with open(tempname, "wb") as f:
            f.write(export_results_csv(results))
        return tempname

    search_btn.click(
        fn=do_search,
        inputs=[bib_input, keywords_input, species_input, perturber_input, min_year_input, max_year_input, max_results_input, journal_filter_checkbox],
        outputs=[result_html, results_state, page_state, query_kw_state]
    )

    prev_btn.click(
        lambda res, p, kw: turn_page(res, p, kw, -1),
        inputs=[results_state, page_state, query_kw_state],
        outputs=[result_html, page_state]
    )
    next_btn.click(
        lambda res, p, kw: turn_page(res, p, kw, 1),
        inputs=[results_state, page_state, query_kw_state],
        outputs=[result_html, page_state]
    )
    csv_btn.click(
        fn=export_csv,
        inputs=[results_state],
        outputs=[csv_output]
    )

if __name__ == "__main__":
    demo.launch()