"""
Spectral Line Broadening Literature Search
------------------------------------------
Crossref multi-strategy retrieval + spectroscopy-aware heuristics + batch LLM triage,
with stable sorting (Journal priority → Year → Title), optional species/perturber
hard filter, and CSV/BibTeX export.

Run:
  1) pip install gradio requests certifi
  2) export/set OPENROUTER_API_KEY="sk-or-xxxx"
  3) python model1.py
"""

from __future__ import annotations

import os
import re
import csv
import json
import string
from typing import List, Dict, Tuple, Any, Optional

import certifi
import gradio as gr
import requests
from requests.adapters import HTTPAdapter, Retry

# Ensure CA bundle when requests is used on some Windows/Python builds
os.environ["SSL_CERT_FILE"] = certifi.where()

# =============================================================================
# Configuration
# =============================================================================

VERSION = "1.8"
PAGE_SIZE = 20
DEFAULT_MAX_RESULTS = 500
DEFAULT_YEAR_MIN = 2000
DEFAULT_YEAR_MAX = 2025

# Networking
HTTP_TIMEOUT = 60
RETRY_TOTAL = 3
RETRY_BACKOFF = 0.6
RETRY_STATUS = [429, 500, 502, 503, 504]

# LLM
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "Missing OPENROUTER_API_KEY. Set env var, e.g.\n"
        "  Windows PowerShell:  $env:OPENROUTER_API_KEY='sk-or-xxxx'\n"
        "  macOS/Linux:         export OPENROUTER_API_KEY='sk-or-xxxx'"
    )
LLM_MODEL = "moonshotai/kimi-k2:free"
LLM_BATCH_SIZE = 30
MAX_ABSTRACT_CHARS = 1200
MAX_TITLE_CHARS = 300
MAX_JOURNAL_CHARS = 100

# Heuristics
PARAM_HEURISTICS = [
    "pressure broadening", "pressure-broadened", "collisional broadening",
    "line broadening", "linewidth", "halfwidth", "half-width",
    "voigt", "line shape", "lineshape", "broadening coefficient",
    "gamma", "pressure shift", "line shift", "temperature exponent",
    "lorentz", "lorentzian", "hwhm", "fwhm",
    "air-broadened", "self-broadened", "n2-broadened", "o2-broadened",
    "line-mixing", "foreign broadening"
]

DEFAULT_QUERY_TERMS = [
    "pressure broadening", "line broadening", "linewidth", "half-width", "halfwidth",
    "Voigt", "broadening coefficient", "gamma", "pressure shift", "temperature exponent", "line shape"
]

# Physics journals (exact names/prefix)
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
    "Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy",
]

# Journal priority (JQSRT → JMS → JCP)
JOURNAL_PRIORITY = [
    "Journal of Quantitative Spectroscopy and Radiative Transfer",
    "Journal of Molecular Spectroscopy",
    "Journal of Chemical Physics",
]

# Soft-negative and must-positive terms for denoising
SOFT_NEG = [
    "collision-induced absorption", " cia ",
    "gamma ray", "neutrino", "proton", "hadron", "quark", "collider",
    "heavy-ion", "brownian", "quantum chromodynamics", "phase transition in lattice",
    "optical clock", "frequency comb", "microresonator", "laser cavity linewidth",
]
MUST_POS = [
    "broadening", "linewidth", "half-width", "halfwidth",
    "voigt", "broadening coefficient", "gamma", "line shift",
    "lorentz", "lorentzian", "hwhm", "fwhm",
]

# =============================================================================
# Session (retry/backoff)
# =============================================================================

def build_session() -> requests.Session:
    """Create a Session with retry/backoff and a descriptive User-Agent."""
    s = requests.Session()
    retries = Retry(
        total=RETRY_TOTAL,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=RETRY_STATUS,
        allowed_methods=frozenset(["GET", "POST"]),
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": f"broadening-search/{VERSION} (research use)"})
    return s

SESSION = build_session()

# =============================================================================
# Utilities
# =============================================================================

def _norm_journal(name: str) -> str:
    """Normalize journal names for matching."""
    if not name:
        return ""
    s = name.lower().replace("&", "and")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

_PHYSICS_SET = {_norm_journal(j) for j in PHYSICS_JOURNALS}
_PHYSICS_PREFIXES = {_norm_journal("Physical Review")}
_PRIORITY_MAP = {_norm_journal(j): i for i, j in enumerate(JOURNAL_PRIORITY)}

def is_physics_journal(journal_name: str) -> bool:
    """True if the journal is a physics journal (exact or prefix match)."""
    n = _norm_journal(journal_name)
    return bool(n) and ((n in _PHYSICS_SET) or any(n.startswith(p) for p in _PHYSICS_PREFIXES))

def priority_score(journal_name: str) -> int:
    """Return 0/1/2 for JQSRT/JMS/JCP, else 999 (lower is better)."""
    return _PRIORITY_MAP.get(_norm_journal(journal_name), 999)

def strip_tags(s: Optional[str]) -> str:
    """Remove XML/HTML tags and condense whitespace."""
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def get_key(entry: Dict[str, Any]) -> str:
    """Dedup key: DOI if present; else title(no punct)+first author."""
    doi = (entry.get("DOI") or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    title = (entry.get("Title") or "").lower()
    title = title.translate(str.maketrans("", "", string.punctuation))
    title = re.sub(r"\s+", " ", title).strip()
    first_author = (entry.get("Authors") or "").split(",")[0].lower().strip()
    return f"title:{title}|fa:{first_author}"

def deduplicate_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stable deduplication preserving first occurrence."""
    seen: set[str] = set()
    unique: List[Dict[str, Any]] = []
    for p in papers:
        key = get_key(p)
        if key not in seen:
            unique.append(p)
            seen.add(key)
    return unique

def highlight_keywords(text: str, keywords: List[str]) -> str:
    """Highlight keywords with <mark> tags in a safe/insensitive way."""
    if not text:
        return ""
    uniq = sorted({kw.strip() for kw in keywords if kw.strip()}, key=len, reverse=True)
    for kw in uniq:
        pat = r"(?<![\w])" + re.escape(kw) + r"(?![\w])"
        text = re.sub(
            pat,
            lambda m: f'<mark style="background:#ffe066;border-radius:3px;padding:1px 3px">{m.group(0)}</mark>',
            text,
            flags=re.IGNORECASE,
        )
    return text

# =============================================================================
# Stable sort: priority → year → title
# =============================================================================

def _safe_year(y: Any) -> Optional[int]:
    try:
        return int(y)
    except Exception:
        return None

def _norm_title(t: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (t or "").strip().lower())

def sort_results_stable(results: List[Dict[str, Any]], year_order: str = "desc", prioritize: bool = True) -> List[Dict[str, Any]]:
    """
    Stable sort key: (priority, year_key, title)
      - priority: 0/1/2 for JQSRT/JMS/JCP if prioritize, else 999
      - year_key: None always last; descending or ascending
      - title: normalized as final tie-breaker for consistent pagination
    """
    desc = year_order.lower().startswith("d")

    def key_fn(r: Dict[str, Any]):
        prio = priority_score(r.get("Journal")) if prioritize else 999
        y = _safe_year(r.get("Year"))
        year_key = (y is None, -(y or 0)) if desc else (y is None, (y or 10**9))
        return (prio, year_key, _norm_title(r.get("Title")))

    return sorted(results, key=key_fn)

# =============================================================================
# Crossref search (union of multi strategies → dedup → soft denoise)
# =============================================================================

def _crossref_call(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        r = SESSION.get("https://api.crossref.org/works", params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        msg = r.json().get("message", {})
        return msg.get("items", []) or []
    except Exception:
        return []

def search_crossref_multi(
    keywords: List[str],
    species: List[str],
    perturbers: List[str],
    *,
    max_results: int = DEFAULT_MAX_RESULTS,
    min_year: int = DEFAULT_YEAR_MIN,
    max_year: int = DEFAULT_YEAR_MAX,
    physics_journals_only: bool = False,
    order: str = "desc",
) -> List[Dict[str, Any]]:
    """
    Build multiple query strings across Crossref fields and union the results.
    Soft-denoise removes obvious non-broadening uses (e.g., gamma rays, optical clocks).
    """
    # Validate years
    if min_year > max_year:
        min_year, max_year = max_year, min_year

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
        "Lorentz Lorentzian HWHM FWHM",
    ]

    # Build query strings
    query_strings: List[str] = []
    if base_terms:
        query_strings.append(" ".join(base_terms + ([specie_block] if specie_block else [])))
    for s in broaden_sets:
        query_strings.append((" ".join([s, specie_block]).strip()) if specie_block else s)
    query_strings = list(dict.fromkeys([q for q in query_strings if q]))

    strategies = ["query", "query.title", "query.bibliographic"]
    base_filter = f"from-pub-date:{min_year}-01-01,until-pub-date:{max_year}-12-31,type:journal-article"

    pool: List[Dict[str, Any]] = []
    for q in query_strings:
        for field in strategies:
            params = {
                field: q,
                "filter": base_filter,
                "rows": max_results,
                "select": "title,author,issued,container-title,DOI,abstract,created",
                "sort": "issued",
                "order": order,
            }
            for item in _crossref_call(params):
                title = (item.get("title") or [""])[0] or ""
                authors = ", ".join(f"{p.get('given','')} {p.get('family','')}".strip() for p in item.get("author", []))
                # Year from 'issued' (fallback to 'created')
                year: Optional[int] = None
                issued = item.get("issued", {})
                if "date-parts" in issued and issued["date-parts"] and issued["date-parts"][0]:
                    year = issued["date-parts"][0][0]
                if year is None:
                    created = item.get("created", {})
                    try:
                        year = created.get("date-parts", [[None]])[0][0]
                    except Exception:
                        year = None

                journal = (item.get("container-title") or [""])[0]
                doi = item.get("DOI", "")
                abstract = strip_tags(item.get("abstract"))

                pool.append(
                    {"Title": title, "Authors": authors, "Year": year, "Journal": journal, "DOI": doi, "Abstract": abstract}
                )

    pool = deduplicate_papers(pool)

    if physics_journals_only:
        pool = [r for r in pool if is_physics_journal(r.get("Journal"))]

    # Soft denoise
    cleaned: List[Dict[str, Any]] = []
    for r in pool:
        s = (r.get("Title", "") + " " + r.get("Abstract", "")).lower()
        if any(neg in s for neg in SOFT_NEG) and not any(pos in s for pos in MUST_POS):
            continue
        cleaned.append(r)

    return cleaned

# =============================================================================
# Heuristic rank + batch LLM triage
# =============================================================================

def coarse_filter(entries: List[Dict[str, Any]], top_k: Optional[int] = None) -> Tuple[List[Dict[str, Any]], int]:
    """Score by heuristic keyword presence; optionally keep top_k."""
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for e in entries:
        s = (e.get("Title", "") + " " + (e.get("Abstract") or "")).lower()
        hits = sum(1 for w in PARAM_HEURISTICS if w in s)
        if e.get("Abstract"):
            hits += 0.5
        scored.append((hits, e))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_k = len(scored) if top_k is None else max(0, int(top_k))
    keep = [e for _, e in scored[: min(top_k, len(scored))]]
    return keep, len(keep)

def _build_llm_payload(chunk: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prepare a compact payload for model consumption."""
    records: List[Dict[str, Any]] = []
    for idx, e in enumerate(chunk):
        t = (e.get("Title") or "")[:MAX_TITLE_CHARS]
        a = (e.get("Abstract") or "").replace("\n", " ")[:MAX_ABSTRACT_CHARS]
        j = (e.get("Journal") or "")[:MAX_JOURNAL_CHARS]
        records.append({"i": idx, "title": t, "abstract": a, "journal": j})
    return records

def _call_llm(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Call OpenRouter; parse JSON safely; on failure mark all as relevant (recall-first)."""
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
        resp = SESSION.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "X-Title": "broadening-llm-batch"},
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                "max_tokens": 700,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
            timeout=HTTP_TIMEOUT,
        )
        data_obj: Any
        content = resp.json()["choices"][0]["message"]["content"]
        try:
            data_obj = json.loads(content)
        except Exception:
            m = re.search(r"\[.*\]", content, re.S)
            data_obj = json.loads(m.group(0)) if m else []

        if isinstance(data_obj, dict) and "results" in data_obj:
            return list(data_obj["results"])
        if isinstance(data_obj, list):
            return data_obj
        if isinstance(data_obj, dict):
            return list(data_obj.get("data", data_obj.get("papers", [])) or [])
        return []
    except Exception as e:
        # recall-first fallback
        return [{"i": r["i"], "relevant": True, "reason": f"fallback: {e}"} for r in records]

def llm_filter(
    entries: List[Dict[str, Any]],
    keywords: List[str],
    species: List[str],
    perturbers: List[str],
    *,
    batch_size: int = LLM_BATCH_SIZE,
    heur_cap: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Batch LLM screening with a coarse heuristic pre-rank."""
    kept: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {"coarse": 0, "batches": 0}
    if not entries:
        return kept, meta

    cand, coarse_n = coarse_filter(entries, top_k=heur_cap)
    meta["coarse"] = coarse_n

    for start in range(0, len(cand), batch_size):
        chunk = cand[start : start + batch_size]
        payload = _build_llm_payload(chunk)
        meta["batches"] += 1
        results = _call_llm(payload) or []
        # Map back
        for r in results:
            try:
                i = int(r.get("i", -1))
            except Exception:
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

# =============================================================================
# Export helpers (CSV / BibTeX)
# =============================================================================

def export_results_csv(results: List[Dict[str, Any]], path: str = "broadening_results.csv") -> str:
    """Write minimal CSV (Title, Authors, Year, Journal, DOI). Return file path."""
    fields = ["Title", "Authors", "Year", "Journal", "DOI"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fields}
            writer.writerow(row)
    return path

def _bibtex_escape(s: str) -> str:
    """Lightweight escaping for common BibTeX special chars."""
    if not s:
        return ""
    repl = {
        "\\": "\\textbackslash{}",
        "{": "\\{",
        "}": "\\}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s

def _authors_to_bibtex(auth_str: str) -> str:
    """
    Convert "Given Family, Given2 Family2" → "Family, Given and Family2, Given2".
    This is heuristic but works for Crossref's 'given' + 'family' flattening we do.
    """
    if not auth_str:
        return ""
    parts = [a.strip() for a in auth_str.split(",") if a.strip()]
    norm = []
    for p in parts:
        toks = p.split()
        if len(toks) == 1:
            norm.append(toks[0])
        else:
            family = toks[-1]
            given = " ".join(toks[:-1])
            norm.append(f"{family}, {given}")
    return " and ".join(norm)

def _citekey_from_entry(r: Dict[str, Any]) -> str:
    """Build a simple citekey: <firstAuthorFamily><year><firstWordOfTitle>."""
    title = (r.get("Title") or "").strip()
    authors = (r.get("Authors") or "").strip()
    year = str(r.get("Year") or "").strip()
    first_word = re.sub(r"[^A-Za-z0-9]+", "", title.split()[0]) if title else "paper"
    first_author = "anon"
    if authors:
        first = authors.split(",")[0].strip()
        toks = first.split()
        first_author = (toks[-1] if toks else first).lower()
    ck = f"{first_author}{year}{first_word}"
    return re.sub(r"[^A-Za-z0-9]+", "", ck)

def export_results_bibtex(results: List[Dict[str, Any]], path: str = "broadening_results.bib") -> str:
    """Write a minimal @article BibTeX file. Return file path."""
    lines: List[str] = []
    for r in results:
        key = _citekey_from_entry(r)
        title = _bibtex_escape(r.get("Title") or "")
        journal = _bibtex_escape(r.get("Journal") or "")
        year = r.get("Year")
        doi = (r.get("DOI") or "").strip()
        authors_bib = _bibtex_escape(_authors_to_bibtex(r.get("Authors") or ""))

        lines.append(f"@article{{{key},")
        if title:   lines.append(f"  title   = {{{title}}},")
        if authors_bib: lines.append(f"  author  = {{{authors_bib}}},")
        if journal: lines.append(f"  journal = {{{journal}}},")
        if year:    lines.append(f"  year    = {{{year}}},")
        if doi:
            lines.append(f"  doi     = {{{doi}}},")
            lines.append(f"  url     = {{https://doi.org/{doi}}},")
        lines.append("}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path

# =============================================================================
# Rendering
# =============================================================================

def format_results_html(results: List[Dict[str, Any]], page: int = 1, keywords: List[str] = []) -> str:
    """Render current page as an HTML table with light highlighting and physics-journal tint."""
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

    for i, r in enumerate(shown, start=start + 1):
        journal_class = "physics-journal" if is_physics_journal(r.get("Journal")) else ""
        html += f'<tr class="{journal_class}">'
        html += f'<td style="text-align:center;">{i}</td>'
        html += f'<td>{highlight_keywords(r.get("Title",""), keywords)}</td>'

        authors_full = r.get("Authors", "") or ""
        author_short = (authors_full[:35] + ("…" if len(authors_full) > 35 else ""))
        html += (
            f'<td title="{authors_full}" '
            f'style="max-width:150px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{author_short}</td>'
        )

        html += f'<td style="text-align:center;">{r.get("Year") or ""}</td>'
        html += f'<td>{r.get("Journal") or ""}</td>'

        doi_display = r.get("DOI") or ""
        if doi_display:
            html += (
                f'<td><a href="https://doi.org/{doi_display}" target="_blank" '
                f'style="color:#2d79c7;text-decoration:underline" '
                f'onclick="navigator.clipboard.writeText(\'{doi_display}\')">{doi_display}</a></td>'
            )
        else:
            html += "<td></td>"

        html += "</tr>"

    pages = max(1, (total - 1) // PAGE_SIZE + 1)
    html += "</tbody></table>"
    html += f'<div style="padding:6px;text-align:right;color:#666">Total <b>{total}</b> entries, page <b>{page}</b> / <b>{pages}</b></div>'
    html += '<div style="padding:6px;background:#e6f7ff;border-radius:4px;margin-top:8px">Blue background: Physics journal</div>'
    html += "</div>"
    return html

# =============================================================================
# Gradio UI
# =============================================================================

def create_demo() -> gr.Blocks:
    """Build the Gradio interface."""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"""
        # Spectral Line Broadening Literature Search ({LLM_MODEL})
        <div style='color:#444;font-size:15px;margin-bottom:10px'>
        Use keywords, active species, perturbers and physics journal filter to maximize recall for pressure broadening parameter literature.
        Click DOI to copy.
        </div>
        """)

        keywords_input = gr.Textbox(
            label="Search keywords (comma separated)",
            value="pressure broadening, linewidth, halfwidth, Voigt, gamma, shift, Lorentz, HWHM, FWHM",
            placeholder="pressure broadening, gamma, shift, ..."
        )
        species_input = gr.Textbox(label="Active Species (comma separated)", value="", placeholder="CO2, NO, H2O ...")
        perturber_input = gr.Textbox(label="Perturbers (comma separated)", value="", placeholder="N2, O2, He, H2, air ...")
        max_results_input = gr.Number(label="Max results", value=DEFAULT_MAX_RESULTS, precision=0)
        min_year_input = gr.Number(label="Earliest year", value=DEFAULT_YEAR_MIN, precision=0)
        max_year_input = gr.Number(label="Latest year", value=DEFAULT_YEAR_MAX, precision=0)
        journal_filter_checkbox = gr.Checkbox(label="Only physics journals", value=True)

        require_species_chk = gr.Checkbox(label="Require species/perturber in title/abstract", value=False)
        heur_cap_input = gr.Number(label="Heuristic cap (0 = keep all)", value=0, precision=0)
        order_dropdown = gr.Dropdown(
            label="Sort by year",
            choices=["DESC (newest first)", "ASC (oldest first)"],
            value="DESC (newest first)",
        )
        prioritize_chk = gr.Checkbox(label="Prioritize journals (JQSRT → JMS → JCP first)", value=True)

        with gr.Row():
            search_btn = gr.Button("Search", scale=1)
            prev_btn = gr.Button("Previous page", scale=1)
            next_btn = gr.Button("Next page", scale=1)

        # New: export buttons
        with gr.Row():
            csv_btn = gr.Button("Export CSV", scale=1)
            bib_btn = gr.Button("Export BibTeX", scale=1)

        result_html = gr.HTML()
        results_state = gr.State([])
        page_state = gr.State(1)
        kw_all_state = gr.State([])
        log_output = gr.Textbox(label="Log", value="", lines=8)
        csv_file = gr.File(label="", visible=False)
        bib_file = gr.File(label="", visible=False)

        def do_search(kw, sp, pe, mx, miny, maxy, journal_filter, require_species, heur_cap, order_choice, prioritize):
            log_msgs: List[str] = []
            mx = int(mx) if mx else DEFAULT_MAX_RESULTS
            miny = int(miny) if miny else DEFAULT_YEAR_MIN
            maxy = int(maxy) if maxy else DEFAULT_YEAR_MAX
            cap = int(heur_cap) if heur_cap else 0  # 0 = keep all
            order = "desc" if "DESC" in (order_choice or "").upper() else "asc"

            keywords = [k.strip() for k in (kw or "").split(",") if k.strip()]
            species = [s.strip() for s in (sp or "").split(",") if s.strip()]
            perturber = [p.strip() for p in (pe or "").split(",") if p.strip()] or ["N2", "O2", "He", "H2", "air"]

            log_msgs.append("Starting CrossRef search (expanded multi-strategy)...")
            all_results = search_crossref_multi(
                keywords, species, perturber,
                max_results=mx, min_year=miny, max_year=maxy,
                physics_journals_only=journal_filter, order=order
            )
            log_msgs.append(f"Fetched (after merge/dedup/soft-neg): {len(all_results)} records")

            # Hard filter (require species/perturber tokens in title/abstract)
            if require_species and (species or perturber):
                toks = [t.lower() for t in (species + perturber)]
                def has_tok(r: Dict[str, Any]) -> bool:
                    s = (r.get("Title","") + " " + (r.get("Abstract","") or "")).lower()
                    return any(t in s for t in toks)
                before = len(all_results)
                all_results = [r for r in all_results if has_tok(r)]
                log_msgs.append(f"Require species/perturber ON: {before} → {len(all_results)}")

            log_msgs.append("Applying heuristic + batch LLM filter ...")
            filtered, meta = llm_filter(all_results, keywords, species, perturber, heur_cap=(None if cap == 0 else cap))
            log_msgs.append(f"After heuristic: {meta.get('coarse', 0)} candidates; LLM kept: {len(filtered)} (batches={meta.get('batches', 0)})")

            # Stable sort: Priority → Year → Title
            filtered = sort_results_stable(filtered, year_order=order, prioritize=prioritize)

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

        def do_export_csv(results: List[Dict[str, Any]]):
            return export_results_csv(results)

        def do_export_bib(results: List[Dict[str, Any]]):
            return export_results_bibtex(results)

        search_btn.click(
            fn=do_search,
            inputs=[
                keywords_input, species_input, perturber_input,
                max_results_input, min_year_input, max_year_input,
                journal_filter_checkbox, require_species_chk, heur_cap_input,
                order_dropdown, prioritize_chk
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
        csv_btn.click(fn=do_export_csv, inputs=[results_state], outputs=[csv_file])
        bib_btn.click(fn=do_export_bib, inputs=[results_state], outputs=[bib_file])

    return demo

# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    demo = create_demo()
    demo.launch()
