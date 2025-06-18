import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

import bibtexparser
import requests
import gradio as gr
import csv
import io
import re
from datetime import datetime

PAGE_SIZE = 20

# 定义物理相关期刊列表
PHYSICS_JOURNALS = [
    "Physical Review", "Journal of Chemical Physics", "Journal of Quantitative Spectroscopy and Radiative Transfer",
    "Journal of Molecular Spectroscopy", "Astrophysical Journal", "Astronomy & Astrophysics", "Molecular Physics",
    "Chemical Physics Letters", "Journal of Physical Chemistry", "Journal of Geophysical Research", "Icarus",
    "Planetary and Space Science", "Optics Express", "Applied Optics", "Spectrochimica Acta Part A",
    "Journal of the Optical Society of America", "Journal of Molecular Structure", "Chemical Physics",
    "Atmospheric Chemistry and Physics", "Journal of Atmospheric Sciences", "Physics Letters A",
    "Journal of Chemical Physics", "Physical Chemistry Chemical Physics", "Journal of Physical Chemistry A",
    "Journal of Physical Chemistry B", "Review of Scientific Instruments", "Journal of Applied Physics",
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
    "J =", "K =", "data", "coefficient", "result", "parameter"
]

# 新增：期刊过滤函数
def is_physics_journal(journal_name):
    """检查期刊是否属于物理相关期刊"""
    if not journal_name:
        return False
    journal_lower = journal_name.lower()
    return any(physics_journal.lower() in journal_lower for physics_journal in PHYSICS_JOURNALS)

# 化学式边界检测函数
def is_chemical_formula(text):
    """检测文本是否是化学式（大写字母开头，包含数字或特殊符号）"""
    return re.match(r'^[A-Z][A-Z0-9_]*$', text) is not None

def highlight_keywords(text, keywords):
    if not text: return ""
    all_kws = list(set([k.lower() for k in keywords] + [k.lower() for k in PARAM_KEYWORDS]))
    
    # 区分化学式和普通关键词的标记方式
    def chemical_replacer(match):
        word = match.group(0)
        return f'<mark style="background:#ff9999;border-radius:3px;padding:1px 3px">{word}</mark>'
    
    def normal_replacer(match):
        word = match.group(0)
        return f'<mark style="background:#ffe066;border-radius:3px;padding:1px 3px">{word}</mark>'
    
    # 先处理化学式（大写）
    chemical_formulas = [kw for kw in all_kws if is_chemical_formula(kw)]
    for cf in chemical_formulas:
        pattern = r'\b' + re.escape(cf) + r'\b'
        text = re.sub(pattern, chemical_replacer, text)
    
    # 处理普通关键词
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
    
    # 预处理物种：将化学式转为大写
    species = [s.upper() if is_chemical_formula(s) else s for s in species]
    
    for entry in bib_database.entries:
        content = " ".join(str(entry.get(k, "")).lower() for k in ['title','abstract','keywords','note'])
        content_original = " ".join(str(entry.get(k, "")) for k in ['title','abstract','keywords','note'])
        
        # 关键词匹配（使用边界检测）
        if keywords:
            kw_found = False
            for kw in keywords:
                pattern = r'\b' + re.escape(kw.lower()) + r'\b'
                if re.search(pattern, content):
                    kw_found = True
                    break
            if not kw_found:
                continue
        
        # 物种匹配（化学式区分大小写）
        if species:
            species_found = False
            for s in species:
                # 化学式使用原始大小写匹配
                if is_chemical_formula(s):
                    pattern = r'\b' + re.escape(s) + r'\b'
                    if re.search(pattern, content_original):
                        species_found = True
                        break
                # 普通物种名称使用小写匹配
                else:
                    pattern = r'\b' + re.escape(s.lower()) + r'\b'
                    if re.search(pattern, content):
                        species_found = True
                        break
            if not species_found:
                continue
        
        # 缓冲气匹配（使用边界检测）
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
        
        # 期刊过滤
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

def search_crossref(query, max_results=1000, min_year=None, max_year=None, journal_filter=False):
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
                content = (title + " " + abstract).lower()
                content_original = title + " " + abstract
                
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
                
                # 期刊过滤
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
    return papers

# 彻底去重：DOI优先，无DOI则用标题+作者简拼哈希
def get_key(entry):
    doi = (entry.get('DOI', '') or '').strip().lower()
    if doi:
        return "doi:" + doi
    title = (entry.get('Title', '') or '').strip().lower()
    authors = (entry.get('Authors', '') or '').strip().lower()
    short = "".join(authors.split()[:2])  # 取作者前两个词
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
        print("读取 bib 文件失败:", e)
        return None

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
    html += f'<div style="padding:6px;text-align:right;color:#666">共 <b>{total}</b> 条，当前第 <b>{page}</b> 页 / 共 <b>{(total-1)//PAGE_SIZE+1}</b> 页</div>'
    html += f'<div style="padding:6px;background:#e6f7ff;border-radius:4px;margin-top:8px">蓝色背景：物理相关期刊</div>'
    html += '</div>'
    return html

def export_results_csv(results):
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=["Title", "Authors", "Year", "Journal", "DOI", "Source"])
    writer.writeheader()
    for r in results:
        writer.writerow(r)
    return csv_buffer.getvalue().encode("utf-8")

# 默认年份为近10年
CUR_YEAR = datetime.now().year
DEFAULT_MIN_YEAR = CUR_YEAR - 9
DEFAULT_MAX_YEAR = CUR_YEAR

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""
    <h2 style='color:#385b95;font-weight:700'>分子压力展宽参数文献筛查（物理期刊限定）</h2>
    <div style='color:#777;font-size:15px;margin-bottom:10px'>
    默认只查最新10年。可用多关键词、多分子、多缓冲气及年份过滤，结果自动去重、分页美观展示，参数关键词智能高亮。导出CSV便于批量处理。DOI可点开也可点击复制。<br>
    <b style='color:#e85c00'>只显示包含实验/参数数据的文献，重复文献自动去除。</b>
    <div style='margin-top:8px;color:#d35400'>
    <b>注意：Active Species (如NO, CO2) 会自动识别为化学式，匹配时将区分大小写并作为独立实体处理</b>
    </div>
    <div style='margin-top:8px;background:#e6f7ff;padding:8px;border-radius:4px'>
    <b>期刊限定：</b> 结果仅包含物理相关期刊（{len(PHYSICS_JOURNALS)}种），如Physical Review, J. Chem. Phys., JQSRT等
    </div>
    </div>
    """)
    with gr.Row():
        bib_input = gr.File(label="上传本地 BibTeX 文件（可选）", file_types=[".bib"])
    with gr.Row():
        keywords_input = gr.Textbox(label="检索关键词 (逗号分隔)", value="broadening, measurement, gamma, uncertainty", scale=3)
        species_input = gr.Textbox(label="Active Species/分子 (逗号分隔)", value="NO", scale=2)
        perturber_input = gr.Textbox(label="Perturbers/缓冲气 (逗号分隔)", value="N2", scale=2)
    with gr.Row():
        min_year_input = gr.Number(label="最早年份 (可选)", value=DEFAULT_MIN_YEAR, precision=0, scale=1)
        max_year_input = gr.Number(label="最晚年份 (可选)", value=DEFAULT_MAX_YEAR, precision=0, scale=1)
        max_results_input = gr.Number(label="最大检索数(建议≤5000)", value=1000, precision=0, scale=2)
    journal_filter_checkbox = gr.Checkbox(label="仅限物理相关期刊", value=True)
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

    def do_search(bibfile, kw, sp, pe, miny, maxy, mx, journal_filter):
        try:
            miny = int(miny) if miny else DEFAULT_MIN_YEAR
            maxy = int(maxy) if maxy else DEFAULT_MAX_YEAR
        except: miny, maxy = DEFAULT_MIN_YEAR, DEFAULT_MAX_YEAR
        try:
            mx = int(mx) if mx else 1000
        except: mx = 1000
        keywords = [k.strip() for k in kw.split(",") if k.strip()]
        species = [s.strip() for s in sp.split(",") if s.strip()]
        perturber = [p.strip() for p in pe.split(",") if p.strip()]
        bibstr = bibfile_to_str(bibfile)
        results = []
        if bibstr:
            results = bib_match_entries(bibstr, keywords, species, perturber, miny, maxy, journal_filter)
        if not results:
            # 构建查询时区分化学式和其他关键词
            query_parts = []
            for s in species:
                if is_chemical_formula(s):
                    query_parts.append(f'"{s}"')  # 化学式加引号确保精确匹配
                else:
                    query_parts.append(s)
            query_parts.extend(keywords)
            query_parts.extend(perturber)
            query_str = " AND ".join(query_parts)
            results = search_crossref(query_str, mx, miny, maxy, journal_filter)
        results = deduplicate_papers(results)
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