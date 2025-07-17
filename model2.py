import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

import requests
import gradio as gr
import bibtexparser
import csv
import io
import re
from collections import Counter
from datetime import datetime

OPENROUTER_API_KEY = "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"
LLM_MODEL = "moonshotai/kimi-k2:free"
PAGE_SIZE = 20

PARAM_KEYWORDS = [
    "gamma", "γ", "pressure broadening coefficient", "temperature exponent", "n=", "γ=",
    "measured", "measurement", "experimental", "value", "fit", "uncertainty",
    "cm-1 atm-1", "reported", "determined", "parameters", "obtained", "tabulated", "table",
    "J =", "K =", "data", "coefficient", "result", "parameter", "shift", "pressure", "rotational",
    "collision", "dependence", "determination", "half-width", "linewidth", "spectra", "temperature", 
    "comparison", "collisional", "intensity", "narrowing", "vibration", "overtone", "band", "transition", 
    "calculation","absorption", "theory", "data"
]

def highlight_keywords(text, keywords):
    if not text: return ""
    all_kws = list(set([k.lower() for k in keywords] + [k.lower() for k in PARAM_KEYWORDS]))
    for kw in all_kws:
        if kw and len(kw) > 2:
            text = re.sub(f'({re.escape(kw)})', r'<mark style="background:#ffe066;">\1</mark>', text, flags=re.I)
    return text

def extract_keywords_from_bib(bibstr, topn=12):
    # 提取 title/keywords 字词出现频率最高的n个词，去除常见词
    stopwords = set([
        "the", "of", "in", "for", "and", "to", "on", "a", "with", "as", "by", "at",
        "an", "from", "study", "line", "lines", "broadening", "parameters", "rotational",
        "pressure", "dependence", "determination", "measurement", "coefficients",
        "spectroscopy", "molecular", "collision", "quantum", "effect", "effects"
    ])
    bib_database = bibtexparser.loads(bibstr)
    word_counter = Counter()
    for entry in bib_database.entries:
        txt = (entry.get('title','') + " " + entry.get('keywords','')).lower()
        words = re.findall(r'\b[a-zA-Z0-9]{3,}\b', txt)
        word_counter.update([w for w in words if w not in stopwords])
    # 最常见n个词，逗号分隔
    return ", ".join([w for w,_ in word_counter.most_common(topn)])

def search_crossref(query, max_results=50, min_year=None, max_year=None):
    results = []
    offset = 0
    batch = 50
    while len(results) < max_results:
        rows = min(batch, max_results - len(results))
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
                timeout=20
            )
            items = resp.json().get("message", {}).get("items", [])
            for item in items:
                title = (item.get("title") or [""])[0]
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
                results.append({
                    "Title": title,
                    "Authors": authors,
                    "Year": year,
                    "Journal": journal,
                    "DOI": item.get("DOI", ""),
                    "Abstract": item.get("abstract", "")
                })
            if not items or len(items) < rows:
                break
            offset += rows
        except Exception as e:
            print("CrossRef error:", e)
            break
    return results

def llm_filter(entries, keywords):
    filtered = []
    for entry in entries:
        content = f"""Title: {entry['Title']}\nAbstract: {entry.get('Abstract','')}\nJournal: {entry.get('Journal','')}\n"""
        prompt = f"""
你是一位分子光谱领域专家。现在请你宽松筛选：判断下面这篇文献是否有可能包含与分子光谱线展宽（broadening）、pressure broadening、collisional broadening、linewidth、gamma、n、shift、rotational dependence、温度参数、collisional data、物理参数、表格、实验测量等相关的实验数据或数值参数。只要有任何可能涉及实验测量、物理参数、谱线参数、表格、tabulated data、broadening/shift/n/gamma/rotational/linewidth/absorption/温度等物理量、哪怕不完整，都保留。不要太严格，不是评论/综述/工程/医疗/药学/生物类的，都宁愿多保留一点。
---
{content}
---
请只输出一行：
- 如果有可能相关，请输出："相关: " + 一句话总结参数或内容
- 如果完全不相关（如仅为评论、综述、工程方法、无关领域），才输出："无关"
- 如果无法判断，请默认输出“相关”，不要误删。
"""
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
                    "temperature": 0.1,
                },
                timeout=25
            )
            ans = resp.json()["choices"][0]["message"]["content"]
            # 只要不是明确“无关”都保留（终极宽松模式）
            if not ans.strip().startswith("无关"):
                entry["LLM"] = ans.replace("\n","")
                filtered.append(entry)
        except Exception as e:
            entry["LLM"] = f"筛选出错：{str(e)}"
            # 出错也保留
            filtered.append(entry)
    return filtered

def format_results_table(entries, page=1, keywords=[]):
    total = len(entries)
    start = (page-1)*PAGE_SIZE
    end = min(start+PAGE_SIZE, total)
    shown = entries[start:end]
    html = """
    <style>
    .custom-table {border-collapse:collapse;width:100%;font-size:17px;}
    .custom-table th {background:#F5F5FA;padding:8px;}
    .custom-table td {padding:8px;}
    .custom-table tr:nth-child(even){background:#e6f3fa;}
    .custom-table tr:hover{background:#d6ecff;}
    </style>
    <div style='overflow-x:auto;max-width:100vw;'>
    <table class='custom-table' border='1'>
    <thead>
      <tr>
        <th>#</th>
        <th>Title</th>
        <th>Authors</th>
        <th>Year</th>
        <th>Journal</th>
        <th>DOI</th>
      </tr>
    </thead>
    <tbody>
    """
    for i, r in enumerate(shown, start=start+1):
        html += "<tr>"
        html += f"<td style='text-align:center'>{i}</td>"
        html += f"<td>{highlight_keywords(r['Title'], keywords)}</td>"
        auth_short = (r['Authors'][:35] + ("…" if len(r['Authors'])>35 else "")) if r['Authors'] else ""
        html += f"<td title='{r['Authors']}'>{auth_short}</td>"
        html += f"<td style='text-align:center'>{r['Year'] or ''}</td>"
        html += f"<td>{r['Journal']}</td>"
        doi_display = r["DOI"] or ""
        if doi_display:
            html += f'''<td><a href="https://doi.org/{doi_display}" target="_blank" style="color:#2d79c7;text-decoration:underline" onclick="navigator.clipboard.writeText('{doi_display}')">{doi_display}</a></td>'''
        else:
            html += "<td></td>"
        html += "</tr>"
    html += "</tbody></table></div>"
    html += f"<div style='padding:8px;color:#666;text-align:right;'>Total <b>{total}</b> entries, page <b>{page}</b> / <b>{(total-1)//PAGE_SIZE+1}</b></div>"
    return html

def export_results_csv(entries):
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=["Title", "Authors", "Year", "Journal", "DOI"])
    writer.writeheader()
    for r in entries:
        writer.writerow({
            "Title": r.get("Title",""),
            "Authors": r.get("Authors",""),
            "Year": r.get("Year",""),
            "Journal": r.get("Journal",""),
            "DOI": r.get("DOI","")
        })
    return csv_buffer.getvalue().encode("utf-8")

CUR_YEAR = datetime.now().year
DEFAULT_MIN_YEAR = CUR_YEAR - 9
DEFAULT_MAX_YEAR = CUR_YEAR

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h2>📚 Bib上传➔自动关键词提取➔CrossRef搜索➔LLM智能筛选有实验参数文献（DeepSeek V3 Chat，可导出）</h2>")
    bib_file = gr.File(label="上传BibTeX文件（.bib）", file_types=[".bib"])
    kw_input = gr.Textbox(label="Search keywords (comma separated)", value="pressure broadening", scale=2)
    species_input = gr.Textbox(label="Active Species (comma separated)", value="", scale=1)
    perturber_input = gr.Textbox(label="Perturbers (comma separated)", value="", scale=1)
    min_year = gr.Number(label="Earliest year", value=2016)
    max_year = gr.Number(label="Latest year", value=2025)
    max_results = gr.Number(label="Max results", value=500)
    with gr.Row():
        search_btn = gr.Button("Search")
        prev_btn = gr.Button("Previous page")
        next_btn = gr.Button("Next page")
        csv_btn = gr.Button("Export as CSV")
        csv_output = gr.File(label="", visible=False)
    result_html = gr.HTML()
    results_state = gr.State([])
    page_state = gr.State(1)
    query_kw_state = gr.State([])

    def auto_extract_keywords(bibfile):
        # 上传bib自动提取高频关键词并自动填入输入框
        if bibfile is None:
            return gr.update(value="pressure broadening")
        try:
            if hasattr(bibfile, "read"):
                bibfile.seek(0)
                data = bibfile.read()
                if isinstance(data, bytes):
                    bibstr = data.decode("utf-8")
                else:
                    bibstr = data
            elif isinstance(bibfile, bytes):
                bibstr = bibfile.decode("utf-8")
            elif isinstance(bibfile, str):
                bibstr = bibfile
            else:
                return gr.update(value="pressure broadening")
            kw = extract_keywords_from_bib(bibstr)
            return gr.update(value=kw if kw else "pressure broadening")
        except Exception as e:
            return gr.update(value="pressure broadening")

    def do_search(kw, sp, pe, miny, maxy, mx):
        log_msgs = []
        keywords = [k.strip() for k in kw.split(",") if k.strip()]
        species = [s.strip() for s in sp.split(",") if s.strip()]
        perturber = [p.strip() for p in pe.split(",") if p.strip()]
        miny = int(miny) if miny else DEFAULT_MIN_YEAR
        maxy = int(maxy) if maxy else DEFAULT_MAX_YEAR
        mx = int(mx) if mx else 20
        query_str = " ".join(keywords + species + perturber)
        log_msgs.append("开始 CrossRef 搜索 ...")
        results = search_crossref(query_str, mx, miny, maxy)
        log_msgs.append(f"Fetched {len(results)} records...")
        if results:
            log_msgs.append("调用LLM智能筛选，仅保留含有实验参数文献 ...")
            results = llm_filter(results, keywords+species+perturber)
            log_msgs.append(f"LLM筛选后剩余 {len(results)} 篇 ...")
        log_msgs.append("全部完成。")
        html = format_results_table(results, 1, keywords + species + perturber)
        return html, results, 1, keywords + species + perturber, "\n".join(log_msgs)

    def turn_page(results, page, keywords, direction):
        page = page + direction
        total = len(results)
        maxpage = max(1, (total-1)//PAGE_SIZE+1)
        if page < 1: page = 1
        if page > maxpage: page = maxpage
        html = format_results_table(results, page, keywords)
        return html, page

    def export_csv(results):
        tempname = "crossref_llm_results.csv"
        with open(tempname, "wb") as f:
            f.write(export_results_csv(results))
        return tempname

    # 绑定 bib 上传自动关键词
    bib_file.change(fn=auto_extract_keywords, inputs=[bib_file], outputs=[kw_input])

    search_btn.click(
        fn=do_search,
        inputs=[kw_input, species_input, perturber_input, min_year, max_year, max_results],
        outputs=[result_html, results_state, page_state, query_kw_state, gr.Textbox(label="进度/日志")]
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
