import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

import bibtexparser
import requests
import gradio as gr
import csv
import io
import re

# 每页显示多少条
PAGE_SIZE = 20

def highlight_keywords(text, keywords):
    if not text:
        return ""
    result = text
    for kw in sorted(keywords, key=lambda x: -len(x)):
        if kw.strip():
            result = re.sub(f'({re.escape(kw)})', r'<mark style="background:yellow">\1</mark>', result, flags=re.IGNORECASE)
    return result

def bib_match_entries(bibstr, keywords=[], species=[], perturber=[], min_year=None, max_year=None):
    bib_database = bibtexparser.loads(bibstr)
    matches = []
    for entry in bib_database.entries:
        content = " ".join(str(entry.get(k, "")).lower() for k in ['title','abstract','keywords','note'])
        if keywords and not any(kw.lower() in content for kw in keywords):
            continue
        if species and not any(s.lower() in content for s in species):
            continue
        if perturber and not any(p.lower() in content for p in perturber):
            continue
        year = entry.get('year','')
        try:
            year_val = int(year)
        except:
            year_val = None
        # 年份过滤
        if min_year and year_val and year_val < min_year:
            continue
        if max_year and year_val and year_val > max_year:
            continue
        matches.append({
            "Title": entry.get('title',''),
            "Authors": entry.get('author',''),
            "Year": year,
            "Journal": entry.get('journal','') or entry.get('booktitle',''),
            "DOI": entry.get('doi',''),
            "Source": "BIB",
        })
    return matches

def search_crossref(query, max_results=1000, min_year=None, max_year=None):
    papers = []
    batch = 1000
    offset = 0
    while len(papers) < max_results:
        rows = min(batch, max_results - len(papers))
        try:
            resp = requests.get(
                "https://api.crossref.org/works",
                params={"query.bibliographic": query, "rows": rows, "offset": offset},
                timeout=30
            )
            items = resp.json().get("message", {}).get("items", [])
            for item in items:
                authors = ", ".join(f"{p.get('given','')} {p.get('family','')}".strip() for p in item.get("author", []))
                pub = item.get("published-print", item.get("published-online", {}))
                year = pub.get("date-parts", [[None]])[0][0]
                try:
                    year_val = int(year)
                except:
                    year_val = None
                # 年份过滤
                if min_year and year_val and year_val < min_year:
                    continue
                if max_year and year_val and year_val > max_year:
                    continue
                papers.append({
                    "Title": (item.get("title") or [""])[0],
                    "Authors": authors,
                    "Year": year,
                    "Journal": (item.get("container-title") or [""])[0],
                    "DOI": item.get("DOI", ""),
                    "Source": "CrossRef"
                })
            if not items or len(items) < rows:
                break
            offset += rows
        except Exception as e:
            print("CrossRef error:", e)
            break
    return papers

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
        print("读取 bib 文件失败:", e)
        return None

def format_results_html(results, page=1, keywords=[]):
    # 分页
    total = len(results)
    start = (page-1)*PAGE_SIZE
    end = min(start+PAGE_SIZE, total)
    shown = results[start:end]
    html = '<div style="overflow-x:auto;max-width:100vw;">'
    html += '<table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse;width:99%;font-size:15px;">'
    html += '<thead style="background:#F5F5FA;">'
    for col in ["#", "Title", "Authors", "Year", "Journal", "DOI", "Source"]:
        html += f'<th>{col}</th>'
    html += '</thead><tbody>'
    for i, r in enumerate(shown, start=start+1):
        html += '<tr>'
        html += f'<td style="text-align:center;">{i}</td>'
        html += f'<td>{highlight_keywords(r["Title"], keywords)}</td>'
        html += f'<td title="{r["Authors"]}">{(r["Authors"][:35] + ("…" if len(r["Authors"])>35 else "")) if r["Authors"] else ""}</td>'
        html += f'<td style="text-align:center;">{r["Year"] or ""}</td>'
        html += f'<td>{r["Journal"]}</td>'
        html += f'<td><a href="https://doi.org/{r["DOI"]}" target="_blank" style="color:blue;" title="复制">{r["DOI"]}</a></td>'
        html += f'<td style="text-align:center;">{r["Source"]}</td>'
        html += '</tr>'
    html += '</tbody></table>'
    html += f'<div style="padding:6px;text-align:right;color:#666">共 <b>{total}</b> 条，当前第 <b>{page}</b> 页 / 共 <b>{(total-1)//PAGE_SIZE+1}</b> 页</div>'
    html += '</div>'
    return html

def export_results_csv(results):
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=["Title", "Authors", "Year", "Journal", "DOI", "Source"])
    writer.writeheader()
    for r in results:
        writer.writerow(r)
    return csv_buffer.getvalue().encode("utf-8")

# gr.State 变量存全部结果和页码
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    <h2 style='color:#444;font-weight:700'>分子压力展宽文献筛查（本地bib优先，CrossRef自动补充）</h2>
    <div style='color:#777;font-size:15px;margin-bottom:10px'>
    支持关键词+物种+缓冲气+年份过滤，结果分页显示，可导出CSV。
    </div>
    """)
    with gr.Row():
        bib_input = gr.File(label="上传本地 BibTeX 文件（可选）", file_types=[".bib"])
    with gr.Row():
        keywords_input = gr.Textbox(label="检索关键词 (逗号分隔)", value="broadening, measurement, gamma, uncertainty", scale=3)
        species_input = gr.Textbox(label="Active Species/分子 (逗号分隔)", value="NO", scale=2)
        perturber_input = gr.Textbox(label="Perturbers/缓冲气 (逗号分隔)", value="N2", scale=2)
    with gr.Row():
        min_year_input = gr.Number(label="最早年份 (可选)", value=2015, precision=0, scale=1)
        max_year_input = gr.Number(label="最晚年份 (可选)", value=2025, precision=0, scale=1)
        max_results_input = gr.Number(label="最大检索数(建议≤5000)", value=1000, precision=0, scale=2)
    search_btn = gr.Button("查找文献", scale=1)
    with gr.Row():
        prev_btn = gr.Button("上一页", scale=1)
        next_btn = gr.Button("下一页", scale=1)
        csv_btn = gr.Button("导出为CSV", scale=2)
        csv_output = gr.File(label="", visible=False)
    result_html = gr.HTML()
    results_state = gr.State([])
    page_state = gr.State(1)
    query_kw_state = gr.State([])

    def do_search(bibfile, kw, sp, pe, miny, maxy, mx):
        try:
            miny = int(miny) if miny else None
            maxy = int(maxy) if maxy else None
        except: miny, maxy = None, None
        try:
            mx = int(mx) if mx else 1000
        except: mx = 1000
        keywords = [k.strip() for k in kw.split(",") if k.strip()]
        species = [s.strip() for s in sp.split(",") if s.strip()]
        perturber = [p.strip() for p in pe.split(",") if p.strip()]
        bibstr = bibfile_to_str(bibfile)
        results = []
        if bibstr:
            results = bib_match_entries(bibstr, keywords, species, perturber, miny, maxy)
        if not results:
            query_str = " AND ".join(keywords + species + perturber)
            results = search_crossref(query_str, mx, miny, maxy)
        html = format_results_html(results, 1, keywords+species+perturber)
        return html, results, 1, keywords+species+perturber

    def turn_page(results, page, keywords, direction):
        page = page + direction
        total = len(results)
        maxpage = max(1, (total-1)//PAGE_SIZE+1)
        if page < 1: page = 1
        if page > maxpage: page = maxpage
        html = format_results_html(results, page, keywords)
        return html, page

    def export_csv(results):
        tempname = "results.csv"
        with open(tempname, "wb") as f:
            f.write(export_results_csv(results))
        return tempname

    search_btn.click(
        fn=do_search,
        inputs=[bib_input, keywords_input, species_input, perturber_input, min_year_input, max_year_input, max_results_input],
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


