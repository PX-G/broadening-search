import os
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]
import re
import io
import glob
import gradio as gr
import pandas as pd
import pdfplumber
from tqdm import tqdm
from transformers import pipeline

# ----------------------------
# 1. 可自定义关键词池（按需调整）
# ----------------------------
KEYWORDS = [
    "broadening", "linewidth", "pressure broadening", "line shape", "half-width",
    "shift", "uncertainty", "CO2", "N2", "H2O", "O2","NO","methane", "ammonia",
    "transition", "quantum number", "J", "K", "temperature", "gamma", "n", "delta"
]

# ----------------------------
# 2. AI参数抽取器（可替换成微调模型）
# ----------------------------
def extract_table_from_text(text):
    # 提取所有参数行（支持多行，支持科学计数法）
    pattern = re.compile(
        r"(?:J[\'′]?\s*=?\s*\d+).{0,40}?gamma\s*[=:=]?\s*([-+]?\d+\.\d+[eE]?[-+]?\d*)"
        r".{0,40}?n\s*[=:=]?\s*([-+]?\d+\.\d+[eE]?[-+]?\d*)"
        r".{0,40}?delta\s*[=:=]?\s*([-+]?\d+\.\d+[eE]?[-+]?\d*)"
        r".{0,40}?T\s*[=:=]?\s*([-+]?\d+\.\d+[eE]?[-+]?\d*)", re.I)
    out = []
    for match in pattern.finditer(text):
        groups = match.groups()
        if groups:
            out.append(groups)
    return out

# 可选：用LLM增强（更智能抽取，可替换）
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device="cpu")

def smart_extract(text):
    # 先粗筛（减少长文本负担）
    table = extract_table_from_text(text)
    if len(table) > 0:
        return table
    # 若无结构数据，尝试AI抽取
    summary = summarizer(text[:2048], max_length=300, min_length=30, do_sample=False)[0]['summary_text']
    # 再用正则粗抽（可进一步用LLM问答等多轮增强）
    return extract_table_from_text(summary)

# ----------------------------
# 3. PDF批量参数表AI抽取（核心）
# ----------------------------
def batch_extract(pdf_files):
    # 支持文件夹
    if isinstance(pdf_files, str) and os.path.isdir(pdf_files):
        pdf_list = glob.glob(os.path.join(pdf_files, "*.pdf"))
    elif isinstance(pdf_files, list):
        pdf_list = pdf_files
    else:
        pdf_list = [pdf_files]
    results = []
    for pdf_file in tqdm(pdf_list):
        try:
            with pdfplumber.open(pdf_file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
            # 筛查是否为相关文献
            if not any(kw.lower() in text.lower() for kw in KEYWORDS):
                continue
            param_list = smart_extract(text)
            for item in param_list:
                results.append({
                    "File": os.path.basename(pdf_file),
                    "gamma": item[0] if len(item) > 0 else "",
                    "n": item[1] if len(item) > 1 else "",
                    "delta": item[2] if len(item) > 2 else "",
                    "Temperature": item[3] if len(item) > 3 else ""
                })
        except Exception as e:
            results.append({"File": os.path.basename(pdf_file), "gamma": "ERROR: "+str(e), "n": "", "delta": "", "Temperature": ""})
    return pd.DataFrame(results)

# ----------------------------
# 4. Gradio前端（可批量导入/导出/预览/分页/高亮）
# ----------------------------
def gradio_extract(pdf_files):
    df = batch_extract(pdf_files)
    if df.empty:
        return "未发现参数表数据", None
    return gr.Dataframe(df, label="批量AI参数表"), df.to_csv(index=False).encode("utf-8")

with gr.Blocks(title="一键批量查全+AI参数表抽取") as demo:
    gr.Markdown("## 一键查全 + 批量AI参数表抽取 + 网页导出  \n支持PDF批量导入/参数表AI提取/CSV导出（个人科研自动化）")
    pdf_input = gr.File(label="上传待抽取参数表的PDF文献（支持多个，或上传文件夹）", file_count="multiple", type="filepath")
    btn = gr.Button("批量抽取参数表（查全+AI智能）")
    out_data = gr.Dataframe(label="参数表批量预览")
    out_csv = gr.File(label="一键下载CSV")
    btn.click(fn=gradio_extract, inputs=pdf_input, outputs=[out_data, out_csv])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7869, share=False)
