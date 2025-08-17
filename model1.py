# broadening_search_app.py
# Spectral line broadening literature search (Crossref + heuristic + batch LLM)
# ─────────────────────────────────────────────────────────────────────────────

import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

import requests
import gradio as gr
import csv
import io
import re
import json
import string
from typing import List, Dict, Tuple

# ====== CONFIG =======
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    raise RuntimeError("Missing OPENROUTER_API_KEY environment variable.")

LLM_MODEL = "deepseek/deepseek-chat-v3-0324:free"
PAGE_SIZE = 20

# —— 物理期刊白名单（精确名/前缀）——
PHYSICS_JOURNALS = [
    "Physical Review", "Physical Review Letters",
    "Physical Review A", "Physical Review B", "Physical Review C", "Physical Review D",
    "Physical Review E", "Physical Review X", "Physical Review Applied",
    "Physical Review Fluids", "Physical Review Materials",
    "Journal of Chemical Physics",
    "Journal of Molecular Spectroscopy",
    "Journal of Quantitative Spectroscopy and Radiative Transfer",
    "Molecular Physics",
    "Review of Scientific Instruments",
    "Journal of Physics B",
    "Journal of Physics D",
    "Journal of Physics: Condensed Matter",
    "Optics Express",
    "Applied Optics",
    "Journal of the Optical Society of America",
    "Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy"
]

# —— 启发式关键词（含老论文表述）——
PARAM_HEURISTICS = [
    # 常用
    "pressure broadening", "pressure-broadened", "collisional broadening",
    "line broadening", "linewidth", "halfwidth", "half-width",
    "voigt", "line shape", "lineshape", "broadening coefficient",
    "gamma", "pressure shift", "line shift", "temperature exponent",
    # 老论文/常见缩写
    "lorentz", "lorentzian", "hwhm", "fwhm",
    "air-broadened", "self-broadened", "n2-broadened", "o2-broadened",
    "line-mixing", "foreign broadening"
]

# 默认检索词（UI 可覆盖）
DEFAULT_QUERY_TERMS = [
    "pressure broadening", "line broadening", "linewidth", "half-width", "halfwidth",
    "Voigt", "broadening coefficient", "gamma", "pressure shift", "temperature exponent", "line shape"
]

# ─────────────────────────────
# 规范化/工具函数
# ─────────────────────────────

def _norm_journal(name: str) -> str:
    if not name:
        return ""
    s = name.lower()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

_PHYSICS_SET = {_norm_journal(j) for j in PHYSICS_JOURNALS}
_PHYSICS_PREFIXES = {_norm_journal(p) for p in ["Physical Review"]}

def is_physics_journal(journal_name: str) -> bool:
    n = _norm_journal(journal_name)
    if not n:
        return False
    return (n in _PHYSICS_SET) or any(n.startswith(p) for p in _PHYSICS_PREFIXES)

def strip_tags(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)  # 去 JATS/HTML
    s = re.sub(r"\s+", " ", s).strip()
    return s

def get_key(entry: Dict) -> str:
    doi = (entry.get('DOI', '') or '').strip().lower()
    if doi:
        return "doi:" + doi
    title = (entry.get('Title','') or '').lower()
    title = title.translate(str.maketrans('', '', string.punctuation))
    title = re.sub(r"\s+", " ", title).strip()
    first_author = (entry.get('Authors','') or '').split(",")[0].lower().strip()
    return f"title:{title}|fa:{first_author}"

def deduplicate_papers(papers: List[Dict]) -> List[Dict]:
    seen = set()
    unique = []
    for p in papers:
        key = get_key(p)
        if key not in seen:
            unique.append(p)
            seen.add(key)
    return unique

def highlight_keywords(text: str, keywords: List[str]) -> str:
    if not text:
        return ""
    uniq = sorted({kw.strip() for kw in keywords if kw.strip()}, key=len, reverse=True)
    for kw in uniq:
        pat = r'(?<![\w])' + re.escape(kw) + r'(?![\w])'
        text = re.sub(
            pat,
            lambda m: f'<mark style="background:#ffe066;border-radius:3px;padding:1px 3px">{m.group(0)}</mark>',
            text,
            flags=re.IGNORECASE
        )
    return text

# ─────────────────────────────
# Crossref 多策略检索（并集 + 去重 + 软性反噪）
# ─────────────────────────────

def search_crossref_multi(keywords, species, perturbers,
                          max_results=500, min_year=2000, max_year=2025,
                          physics_journals_only=False, order="desc") -> List[Dict]:
    """
    Multi-strategy query:
    - 多组同义词 & 多个 query 字段（query / query.title / query.bibliographic）
    - 并集去重；可选期刊白名单；软性过滤掉 CIA-only / 高能物理等噪声
    """
    def _call(params):
        headers = {"User-Agent": "broadening-search/1.3 (mailto:your_email@ucl.ac.uk)"}
        try:
            r = requests.get("https://api.crossref.org/works", params=params, headers=headers, timeout=60)
            r.raise_for_status()
            return r.json().get("message", {}).get("items", [])
        except Exception:
            return []

    # 用户词/默认词
    user_terms = [t.strip() for t in (keywords or []) if t.strip()]
    base_terms = user_terms or DEFAULT_QUERY_TERMS

    sp = " ".join(species or [])
    pe = " ".join(perturbers or [])
    specie_block = " ".join([sp, pe]).strip()

    broaden_sets = [
        "pressure broadening",
        "collisional broadening",
        "line broadening",
        "linewidth half-width halfwidth",
        "Voigt lineshape line shape",
        "broadening coefficient gamma",
        "pressure shift line shift",
        "temperature exponent n",
        # 老论文里的描述
        "Lorentz Lorentzian HWHM FWHM"
    ]

    query_strings = []
    if base_terms:
        query_strings.append(" ".join(base_terms + ([specie_block] if specie_block else [])))
    for s in broaden_sets:
        qs = " ".join([s, specie_block]).strip() if specie_block else s
        query_strings.append(qs)
    query_strings = list(dict.fromkeys([q for q in query_strings if q]))

    strategies = [
        ("query", None),
        ("query.title", None),
        ("query.bibliographic", None),
    ]

    base_filter = f"from-pub-date:{min_year}-01-01,until-pub-date:{max_year}-12-31,type:journal-article"

    pool = []
    for q in query_strings:
        for field, _ in strategies:
            params = {
                field: q,
                "filter": base_filter,
                "rows": max_results,
                "select": "title,author,issued,container-title,DOI,abstract",
                "sort": "issued",
                "order": order
            }
            items = _call(params)
            for item in items:
                title = (item.get("title") or [""])[0] or ""
                authors = ", ".join(f"{p.get('given','')} {p.get('family','')}".strip() for p in item.get("author", []))
                year = None
                issued = item.get("issued", {})
                if "date-parts" in issued and issued["date-parts"] and issued["date-parts"][0]:
                    year = issued["date-parts"][0][0]
                journal = (item.get("container-title") or [""])[0]
                doi = item.get("DOI", "")
                abstract = strip_tags(item.get("abstract") or "")
                pool.append({
                    "Title": title, "Authors": authors, "Year": year, "Journal": journal, "DOI": doi, "Abstract": abstract
                })

    pool = deduplicate_papers(pool)

    if physics_journals_only:
        pool = [r for r in pool if is_physics_journal(r.get("Journal"))]

    # 软性负样过滤：CIA-only & 高能/粒子/等离子等；若出现真正展宽关键词则放行
    soft_neg = [
        "collision-induced absorption", " cia ",
        "gamma ray", "neutrino", "proton", "hadron", "quark", "collider",
        "heavy-ion", "brownian", "quantum chromodynamics", "phase transition in lattice",
        "optical clock", "frequency comb", "microresonator", "laser cavity linewidth"
    ]
    must_pos = [
        "broadening", "linewidth", "half-width", "halfwidth",
        "voigt", "broadening coefficient", "gamma", "line shift",
        "lorentz", "lorentzian", "hwhm", "fwhm"
    ]

    cleaned = []
    for r in pool:
        s = (r.get("Title","") + " " + r.get("Abstract","")).lower()
        if any(neg in s for neg in soft_neg) and not any(pos in s for pos in must_pos):
            continue
        cleaned.append(r)

    return cleaned

# ─────────────────────────────
# 启发式排序 + 批量 LLM 过滤（Recall-first）
# ─────────────────────────────

def coarse_filter(entries: List[Dict], top_k: int | None = None) -> Tuple[List[Dict], int]:
    """
    只排序不裁剪；top_k=None 表示保留全部。
    你也可以传入一个上限（比如 300）来做轻度裁剪。
    """
    scored = []
    for e in entries:
        s = (e.get("Title","") + " " + (e.get("Abstract","") or "")).lower()
        hits = sum(1 for w in PARAM_HEURISTICS if w in s)
        if e.get("Abstract"):
            hits += 0.5
        scored.append((hits, e))
    scored.sort(key=lambda x: x[0], reverse=True)
    if top_k is None:
        top_k = len(scored)
    keep = [e for _, e in scored][:min(top_k, len(scored))]
    return keep, len(keep)

def llm_filter(entries: List[Dict],
               keywords: List[str],
               species: List[str],
               perturbers: List[str],
               batch_size: int = 30,
               heur_cap: int | None = None) -> Tuple[List[Dict], Dict]:
    kept: List[Dict] = []
    meta = {"coarse": 0, "batches": 0}
    if not entries:
        return kept, meta

    cand, coarse_n = coarse_filter(entries, top_k=heur_cap)  # None=不过滤
    meta["coarse"] = coarse_n

    def build_payload(chunk: List[Dict]) -> List[Dict]:
        records = []
        for idx, e in enumerate(chunk):
            t = (e.get("Title") or "")[:300]
            a = (e.get("Abstract") or "").replace("\n", " ")
            a = a[:1200]
            j = (e.get("Journal") or "")[:100]
            records.append({"i": idx, "title": t, "abstract": a, "journal": j})
        return records

    def call_llm(records: List[Dict]) -> List[Dict]:
        sys_prompt = (
            "You are an expert in molecular/atomic spectroscopy (gas-phase). "
            "For each paper, decide if it likely reports EXPERIMENTAL line-shape or pressure-broadening "
            "parameters for molecular/atomic transitions (e.g., Lorentz/Voigt width HWHM/FWHM, gamma, "
            "temperature exponent n, pressure/line shift) typically in IR/visible/microwave spectra. "
            "EXCLUDE particle/nuclear/high-energy physics (gamma rays, neutrinos, proton collisions, colliders), "
            "solid-state or laser-cavity linewidths without gas-phase collisions, generic optics (optical clocks), "
            "and unrelated Brownian/transport statistics. "
            "Be inclusive only if it plausibly contains measured broadening parameters. "
            "Return a JSON array with objects: {i:<index>, relevant:true/false, reason:<short>}."
        )
        user_prompt = "PAPERS:\n" + "\n".join(
            [f"[{r['i']}] Title: {r['title']}\nAbstract: {r['abstract']}\nJournal: {r['journal']}" for r in records]
        )
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "X-Title": "broadening-llm-batch"
                },
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": 700,
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"}
                },
                timeout=60
            )
            content = resp.json()["choices"][0]["message"]["content"]
            try:
                data = json.loads(content)
            except Exception:
                m = re.search(r"\[.*\]", content, re.S)
                data = json.loads(m.group(0)) if m else []

            if isinstance(data, dict) and "results" in data:
                arr = data["results"]
            elif isinstance(data, list):
                arr = data
            else:
                arr = data.get("data", data.get("papers", [])) if isinstance(data, dict) else []
            return arr
        except Exception as e:
            # 出错保守：全部纳入
            return [{"i": r["i"], "relevant": True, "reason": f"fallback: {e}"} for r in records]

    for start in range(0, len(cand), batch_size):
        chunk = cand[start:start+batch_size]
        payload = build_payload(chunk)
        meta["batches"] += 1
        results = call_llm(payload) or []
        for r in results:
            try:
                i = int(r.get("i", -1))
            except:
                i = -1
            if 0 <= i < len(chunk):
                original = cand[start + i]
                if r.get("relevant", True):
                    original["LLM"] = f"Relevant: {r.get('reason','')}"
                    kept.append(original)
        if not results:
            for original in chunk:
                original["LLM"] = "Relevant: fallback empty"
                kept.append(original)

    kept = deduplicate_papers(kept)
    return kept, meta

# ─────────────────────────────
# 展示 & 导出
# ─────────────────────────────

def format_results_html(results: List[Dict], page: int = 1, keywords: List[str] = []) -> str:
    total = len(results)
    start = (page - 1) * PAGE_SIZE
    end = min(start + PAGE_SIZE, total)
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
        journal_class = "physics-journal" if is_physics_journal(r.get("Journal")) else ""
        html += f'<tr class="{journal_class}">'
        html += f'<td style="text-align:center;">{i}</td>'
        html += f'<td>{highlight_keywords(r.get("Title",""), keywords)}</td>'

        authors_full = r.get("Authors","") or ""
        author_short = (authors_full[:35] + ("…" if len(authors_full) > 35 else ""))
        html += f'<td title="{authors_full}" style="max-width:150px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{author_short}</td>'

        html += f'<td style="text-align:center;">{r.get("Year","") or ""}</td>'
        html += f'<td>{r.get("Journal","") or ""}</td>'

        doi_display = r.get("DOI","") or ""
        if doi_display:
            html += (f'''<td><a href="https://doi.org/{doi_display}" target="_blank" '''
                     f'''style="color:#2d79c7;text-decoration:underline" '''
                     f'''onclick="navigator.clipboard.writeText('{doi_display}')">{doi_display}</a></td>''')
        else:
            html += "<td></td>"

        html += '</tr>'

    pages = max(1, (total - 1) // PAGE_SIZE + 1)
    html += '</tbody></table>'
    html += f'<div style="padding:6px;text-align:right;color:#666">Total <b>{total}</b> entries, page <b>{page}</b> / <b>{pages}</b></div>'
    html += f'<div style="padding:6px;background:#e6f7ff;border-radius:4px;margin-top:8px">Blue background: Physics journal</div>'
    html += '</div>'
    return html

def export_results_csv(results: List[Dict]) -> bytes:
    csv_buffer = io.StringIO()
    fieldnames = ["Title", "Authors", "Year", "Journal", "DOI", "LLM"]
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow({k: r.get(k, "") for k in fieldnames})
    return csv_buffer.getvalue().encode("utf-8")

# ─────────────────────────────
# Gradio UI
# ─────────────────────────────

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""
    # Spectral Line Broadening Literature Search ({LLM_MODEL})
    <div style='color:#444;font-size:15px;margin-bottom:10px'>
    Use keywords, active species, perturbers and physics journal filter to maximize recall for pressure broadening parameter literature.
    Click DOI to copy. Download CSV for batch processing.
    </div>
    """)

    keywords_input = gr.Textbox(
        label="Search keywords (comma separated)",
        value="pressure broadening, linewidth, halfwidth, Voigt, gamma, shift, Lorentz, HWHM, FWHM",
        placeholder="pressure broadening, gamma, shift, ..."
    )
    species_input = gr.Textbox(label="Active Species (comma separated)", value="", placeholder="CO2, NO, H2O ...")
    perturber_input = gr.Textbox(label="Perturbers (comma separated)", value="", placeholder="N2, O2, He, H2, air ...")
    max_results_input = gr.Number(label="Max results", value=500, precision=0)
    min_year_input = gr.Number(label="Earliest year", value=2000, precision=0)
    max_year_input = gr.Number(label="Latest year", value=2025, precision=0)
    journal_filter_checkbox = gr.Checkbox(label="Only physics journals", value=True)

    # 新增：启发式上限（0=不过滤）与按年排序
    heur_cap_input = gr.Number(label="Heuristic cap (0 = keep all)", value=0, precision=0)
    order_dropdown = gr.Dropdown(
        label="Sort by year",
        choices=["DESC (newest first)", "ASC (oldest first)"],
        value="DESC (newest first)"
    )

    with gr.Row():
        search_btn = gr.Button("Search", scale=1)
        prev_btn = gr.Button("Previous page", scale=1)
        next_btn = gr.Button("Next page", scale=1)
        csv_btn = gr.Button("Export as CSV", scale=2)

    csv_output = gr.File(label="", visible=False)
    result_html = gr.HTML()
    results_state = gr.State([])
    page_state = gr.State(1)
    kw_all_state = gr.State([])
    log_output = gr.Textbox(label="Log", value="", lines=8)

    def do_search(kw, sp, pe, mx, miny, maxy, journal_filter, heur_cap, order_choice):
        log_msgs = []
        mx = int(mx) if mx else 500
        miny = int(miny) if miny else 2000
        maxy = int(maxy) if maxy else 2025
        cap = int(heur_cap) if heur_cap else 0  # 0=不过滤
        order = "desc" if "DESC" in (order_choice or "").upper() else "asc"

        keywords = [k.strip() for k in (kw or "").split(",") if k.strip()]
        species = [s.strip() for s in (sp or "").split(",") if s.strip()]
        perturber = [p.strip() for p in (pe or "").split(",") if p.strip()]
        if not perturber:
            perturber = ["N2", "O2", "He", "H2", "air"]

        log_msgs.append("Starting CrossRef search (expanded multi-strategy)...")
        all_results = search_crossref_multi(
            keywords, species, perturber,
            max_results=mx, min_year=miny, max_year=maxy,
            physics_journals_only=journal_filter, order=order
        )
        log_msgs.append(f"Fetched (after merge/dedup/soft-neg): {len(all_results)} records")

        log_msgs.append("Applying heuristic + batch LLM filter ...")
        filtered, meta = llm_filter(all_results, keywords, species, perturber, heur_cap=(None if cap==0 else cap))
        log_msgs.append(f"After heuristic: {meta.get('coarse', 0)} candidates; LLM kept: {len(filtered)} (batches={meta.get('batches',0)})")

        # 再按年份做稳定排序（以免多路合并后顺序乱）
        if order == "asc":
            filtered = sorted(filtered, key=lambda r: (r.get("Year") is None, r.get("Year")))
        else:
            filtered = sorted(filtered, key=lambda r: (r.get("Year") is None, -int(r.get("Year") or 0)))

        kw_all = keywords + species + perturber
        html = format_results_html(filtered, 1, kw_all)
        log_msgs.append("Done.")
        return html, filtered, 1, kw_all, "\n".join(log_msgs)

    def turn_page(results, page, kw_all, direction):
        page = page + direction
        total = len(results)
        maxpage = max(1, (total - 1) // PAGE_SIZE + 1)
        page = min(max(page, 1), maxpage)
        html = format_results_html(results, page, kw_all)
        return html, page

    def export_csv(results):
        tempname = "broadening_results.csv"
        with open(tempname, "wb") as f:
            f.write(export_results_csv(results))
        return tempname

    search_btn.click(
        fn=do_search,
        inputs=[
            keywords_input, species_input, perturber_input,
            max_results_input, min_year_input, max_year_input,
            journal_filter_checkbox, heur_cap_input, order_dropdown
        ],
        outputs=[result_html, results_state, page_state, kw_all_state, log_output]
    )
    prev_btn.click(
        lambda res, p, kwa: turn_page(res, p, kwa, -1),
        inputs=[results_state, page_state, kw_all_state],
        outputs=[result_html, page_state]
    )
    next_btn.click(
        lambda res, p, kwa: turn_page(res, p, kwa, 1),
        inputs=[results_state, page_state, kw_all_state],
        outputs=[result_html, page_state]
    )
    csv_btn.click(
        fn=export_csv,
        inputs=[results_state],
        outputs=[csv_output]
    )

if __name__ == "__main__":
    demo.launch()
