import os
import certifi
import requests
import arxiv
import torch
from collections import Counter
import re

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# SSL 证书配置
os.environ["SSL_CERT_FILE"] = certifi.where()
import gradio as gr
# 本地摘要模型
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)

# 动态关键词池
keywords_pool = {
    "phenomenon": [
        "line broadening", "pressure broadening", "collisional broadening",
        "line shape", "line width", "spectral line shapes", "broadening coefficients"
    ],
    "dependency": [
        "rotational dependence", "J-dependence", "rotational effects",
        "rotational quantum number", "rovibrational", "rotationally resolved"
    ],
    "molecules": [
        "water", "H2O", "carbon monoxide", "CO", "carbon dioxide", "CO2",
        "methane", "CH4", "ammonia", "NH3"
    ],
    "context": [
        "spectroscopy", "quantum chemistry", "atmospheric physics", "astrophysics"
    ]
}

def build_concept_query(concept_terms):
    return "(" + " OR ".join([f'"{term}"' if ' ' in term else term for term in concept_terms]) + ")"

def build_full_query(pool_dict):
    query_parts = []
    for category, terms in pool_dict.items():
        if terms:
            query_parts.append(build_concept_query(terms))
    return " AND ".join(query_parts)

def summarize_and_filter(abstract: str) -> str:
    if not abstract: return ""
    text = abstract.lower()
    filter_terms = []
    for terms in keywords_pool.values():
        filter_terms.extend([t.lower() for t in terms])
    if not any(kw in text for kw in filter_terms):
        return ""
    try:
        return summarizer(abstract, max_length=60, min_length=20, do_sample=False)[0]["summary_text"]
    except Exception:
        return abstract[:120] + "..."

def extract_high_freq_terms(papers, exclude_terms, top_n=5):
    words = []
    for p in papers:
        text = (p.get('Title', '') + ' ' + p.get('Summary', '')).lower()
        tokens = re.findall(r'\b\w+\b', text)
        words.extend(tokens)
    counter = Counter(words)
    for term in exclude_terms:
        counter.pop(term.lower(), None)
    return [word for word, count in counter.most_common(top_n)]

def update_keywords_pool(papers, pool_dict):
    exclude_terms = []
    for terms in pool_dict.values():
        exclude_terms.extend([t.lower() for t in terms])
    new_terms = extract_high_freq_terms(papers, exclude_terms, top_n=3)
    for term in new_terms:
        if 'rotational' in term or 'j' in term:
            if term not in pool_dict['dependency']:
                pool_dict['dependency'].append(term)
        elif 'broadening' in term or 'line' in term:
            if term not in pool_dict['phenomenon']:
                pool_dict['phenomenon'].append(term)
        elif 'water' in term or 'co' in term or 'ch4' in term or 'nh3' in term or 'h2o' in term:
            if term not in pool_dict['molecules']:
                pool_dict['molecules'].append(term)
        else:
            if term not in pool_dict['context']:
                pool_dict['context'].append(term)

def search_and_render_html(species_str, perturber_str, qnum_str, max_results, auto_expand=False):
    # 用户输入动态加入分子/缓冲气/量子数
    temp_keywords_pool = {k: v.copy() for k, v in keywords_pool.items()}
    if species_str.strip(): temp_keywords_pool['molecules'].append(species_str.strip())
    if perturber_str.strip(): temp_keywords_pool['molecules'].append(perturber_str.strip())
    if qnum_str.strip(): temp_keywords_pool['dependency'].append(qnum_str.strip())
    full_query = build_full_query(temp_keywords_pool)
    rows = []

    # arXiv 检索
    for paper in arxiv.Search(query=full_query, sort_by=arxiv.SortCriterion.SubmittedDate, max_results=max_results).results():
        summary = summarize_and_filter(paper.summary)
        rows.append({
            "Title": paper.title.replace("\n", " "),
            "Authors": ", ".join(a.name for a in paper.authors),
            "Year": paper.published.year,
            "Journal": paper.journal_ref or "",
            "DOI": paper.doi or paper.entry_id.split("/")[-1],
            "Source": "arXiv",
            "Summary": summary
        })

    # CrossRef 检索
    try:
        resp = requests.get(
            "https://api.crossref.org/works",
            params={"query.bibliographic": full_query, "rows": max_results},
            timeout=30  # 超时增大
        )
        items = resp.json().get("message", {}).get("items", [])
    except Exception as e:
        print("CrossRef request error:", e)
        items = []

    for item in items:
        authors = ", ".join(
            f"{p.get('given','')} {p.get('family','')}".strip()
            for p in item.get("author", [])
        )
        pub = item.get("published-print", item.get("published-online", {}))
        year = pub.get("date-parts", [[None]])[0][0]
        rows.append({
            "Title": (item.get("title") or [""])[0],
            "Authors": authors,
            "Year": year,
            "Journal": (item.get("container-title") or [""])[0],
            "DOI": item.get("DOI", ""),
            "Source": "CrossRef",
            "Summary": ""
        })

    species_list = [s.strip().lower() for s in species_str.split(',') if s.strip()]
    perturbs_list = [p.strip().lower() for p in perturber_str.split(',') if p.strip()]
    qnums_list = [q.strip().lower() for q in qnum_str.split(',') if q.strip()]

    def keep_row(r):
        text = (r['Title'] + ' ' + r['Summary']).lower()
        return (
            (not species_list or any(s in text for s in species_list)) and
            (not perturbs_list or any(p in text for p in perturbs_list)) and
            (not qnums_list or any(q in text for q in qnums_list))
        )
    filtered = [r for r in rows if keep_row(r)]

    # 动态高频扩展
    if auto_expand:
        update_keywords_pool(filtered, keywords_pool)

    # 渲染 HTML
    html = '<div style="overflow-x:auto;">'
    html += '<table border="1" cellpadding="5" cellspacing="0" style="border-collapse:collapse;width:100%;">'
    html += '<thead style="background:#f0f0f0;">'
    for col in ["Title", "Authors", "Year", "Journal", "DOI", "Source", "Summary"]:
        html += f'<th>{col}</th>'
    html += '</thead><tbody>'
    for r in filtered:
        auth_list = r["Authors"].split(', ')
        short_auth = ', '.join(auth_list[:2]) + ('...' if len(auth_list) > 2 else '')
        html += '<tr>'
        html += f'<td>{r["Title"]}</td>'
        html += f'<td title="{r["Authors"]}">{short_auth}</td>'
        html += f'<td style="text-align:center;">{r["Year"] or ""}</td>'
        html += f'<td>{r["Journal"]}</td>'
        html += f'<td><a href="https://doi.org/{r["DOI"]}" target="_blank">{r["DOI"]}</a></td>'
        html += f'<td style="text-align:center;">{r["Source"]}</td>'
        html += f'<td>{r["Summary"]}</td>'
        html += '</tr>'
    html += '</tbody></table></div>'
    html += "<pre>当前关键词池: " + str(keywords_pool) + "</pre>"
    return html

with gr.Blocks() as demo:
    gr.Markdown("## Dynamic Literature Search with Auto-expanding Keywords (arXiv + CrossRef)")
    gr.Markdown("仅 arXiv 和 CrossRef，动态关键词池，自动高频扩展。检索后可选自动扩展并展示当前关键词池。")
    with gr.Row():
        species_input = gr.Textbox(label="Active Species", placeholder="e.g. NO, CO, HCN")
        perturber_input = gr.Textbox(label="Perturbers", placeholder="e.g. N2, CO2, Ar")
        qnum_input = gr.Textbox(label="Quantum Numbers", placeholder="e.g. J', J\", Ka, Kc")
    with gr.Row():
        max_results_input = gr.Slider(label="Max Results", minimum=1, maximum=50, step=1, value=20)
        auto_expand = gr.Checkbox(label="自动扩展高频关键词", value=True)
        search_btn = gr.Button("Search & Expand")
    result_html = gr.HTML()

    search_btn.click(
        fn=search_and_render_html,
        inputs=[species_input, perturber_input, qnum_input, max_results_input, auto_expand],
        outputs=[result_html]
    )

if __name__ == "__main__":
    demo.launch()
