"""
Master Compiler for Tata Power GET AI/ML Study Handbook PDF
------------------------------------------------------------
Assembles all 25 parts into a unified, high-resolution, publication-quality PDF.
"""

import os
import subprocess
import mistune

from handbook_src.part1_foundations import CONTENT as PART1
from handbook_src.part2_python_numpy_pandas import CONTENT as PART2
from handbook_src.part3_nlp_tfidf import CONTENT as PART3
from handbook_src.part4_ml_models_eval import CONTENT as PART4
from handbook_src.part5_pipeline_code import CONTENT as PART5
from handbook_src.part6_implementation_debug_limits import CONTENT as PART6
from handbook_src.part7_100_questions import CONTENT as PART7
from handbook_src.part8_cross_exam_tata_power_resume_cheat import CONTENT as PART8


def build_master_handbook():
    print("[*] Assembling all 25 parts of the Master Handbook...")
    
    css_styles = """
    @page {
        size: A4;
        margin: 18mm 15mm 18mm 15mm;
        @bottom-right {
            content: counter(page);
            font-size: 8pt;
            color: #64748B;
        }
        @bottom-left {
            content: "Tata Power GET AI/ML Study Handbook | Movie Sentiment Analysis";
            font-size: 8pt;
            color: #64748B;
        }
    }
    
    body {
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', 'Helvetica Neue', Arial, sans-serif;
        color: #1E293B;
        line-height: 1.55;
        font-size: 10pt;
        background-color: #FFFFFF;
    }
    
    .cover-page {
        page-break-after: always;
        padding-top: 35mm;
        text-align: center;
    }
    
    .cover-badge {
        display: inline-block;
        background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
        color: #3730A3;
        font-size: 10.5pt;
        font-weight: 700;
        padding: 6px 18px;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 20px;
        border: 1px solid #C7D2FE;
    }
    
    .cover-title {
        font-size: 26pt;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
        margin-bottom: 10px;
        letter-spacing: -0.02em;
    }
    
    .cover-subtitle {
        font-size: 13pt;
        color: #475569;
        margin-bottom: 30px;
        font-weight: 400;
    }
    
    .cover-divider {
        width: 120px;
        height: 4px;
        background: linear-gradient(90deg, #2563EB, #4F46E5);
        margin: 0 auto 30px auto;
        border-radius: 2px;
    }
    
    .meta-box {
        display: inline-block;
        text-align: left;
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 26px;
        margin-top: 15px;
        font-size: 10pt;
        line-height: 1.8;
    }
    
    .toc-grid {
        display: flex;
        justify-content: space-between;
        margin-top: 20px;
    }
    
    .toc-grid div {
        width: 48%;
    }
    
    .toc-grid ul {
        list-style-type: none;
        padding-left: 0;
    }
    
    .toc-grid li {
        margin-bottom: 8pt;
        font-size: 9.5pt;
        border-bottom: 1px dashed #E2E8F0;
        padding-bottom: 4pt;
    }
    
    h1 {
        font-size: 17pt;
        font-weight: 700;
        color: #0F172A;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 6px;
        margin-top: 24pt;
        margin-bottom: 12pt;
        page-break-after: avoid;
    }
    
    .page-break-before {
        page-break-before: always;
    }
    
    h2 {
        font-size: 13.5pt;
        font-weight: 700;
        color: #1E3A8A;
        margin-top: 18pt;
        margin-bottom: 8pt;
        page-break-after: avoid;
    }
    
    h3 {
        font-size: 11.5pt;
        font-weight: 700;
        color: #334155;
        margin-top: 14pt;
        margin-bottom: 6pt;
        page-break-after: avoid;
    }
    
    h4 {
        font-size: 10.5pt;
        font-weight: 700;
        color: #1E293B;
        margin-top: 12pt;
        margin-bottom: 5pt;
        page-break-after: avoid;
    }
    
    p, li {
        color: #334155;
        margin-bottom: 6pt;
    }
    
    ul, ol {
        padding-left: 20px;
        margin-bottom: 10pt;
    }
    
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 12pt 0;
        font-size: 9pt;
        page-break-inside: avoid;
    }
    
    th {
        background-color: #0F172A;
        color: #FFFFFF;
        font-weight: 600;
        text-align: left;
        padding: 7px 9px;
        border: 1px solid #0F172A;
    }
    
    td {
        padding: 6px 9px;
        border: 1px solid #CBD5E1;
        color: #334155;
    }
    
    tr:nth-child(even) {
        background-color: #F8FAFC;
    }
    
    pre {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 8px 12px;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 8.5pt;
        line-height: 1.4;
        overflow-x: auto;
        margin: 8pt 0;
        page-break-inside: avoid;
        color: #0F172A;
    }
    
    code {
        font-family: 'Consolas', 'Courier New', monospace;
        background-color: #F1F5F9;
        color: #0F172A;
        padding: 1.5px 4px;
        border-radius: 4px;
        font-size: 8.5pt;
    }
    
    pre code {
        background-color: transparent;
        padding: 0;
        color: inherit;
    }
    
    blockquote {
        border-left: 4px solid #3B82F6;
        background-color: #EFF6FF;
        margin: 10pt 0;
        padding: 8px 12px;
        border-radius: 0 6px 6px 0;
        color: #1E40AF;
        font-size: 9.5pt;
    }
    
    .checkpoint-box {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-left: 5px solid #6366F1;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 16pt 0;
        page-break-inside: avoid;
    }
    
    .checkpoint-box h3 {
        color: #4338CA;
        margin-top: 0;
        margin-bottom: 6pt;
    }
    
    .answers-toggle {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 8px 12px;
        margin-top: 8pt;
        font-size: 9pt;
    }
    
    .dialogue-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 14pt;
        page-break-inside: avoid;
    }
    
    .dialogue-box h4 {
        color: #1E3A8A;
        margin-top: 0;
        margin-bottom: 6pt;
    }
    
    .resume-box {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 14pt 0;
        font-size: 9.5pt;
    }
    
    .checklist p {
        margin-bottom: 6pt;
        font-size: 9.5pt;
    }
    
    hr {
        border: none;
        border-top: 1px solid #E2E8F0;
        margin: 16pt 0;
    }
    
    a {
        color: #2563EB;
        text-decoration: none;
    }
    """

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Tata Power GET AI/ML Master Study Handbook - Movie Review Sentiment Analysis</title>
<style>
{css_styles}
</style>
</head>
<body>

{PART1}
{PART2}
{PART3}
{PART4}
{PART5}
{PART6}
{PART7}
{PART8}

</body>
</html>
"""

    html_file = 'TATA_POWER_GET_AIML_MASTER_HANDBOOK.html'
    pdf_file = 'TATA_POWER_GET_AIML_MASTER_HANDBOOK.pdf'
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(full_html)
        
    print(f"[+] Saved complete 25-part HTML to {html_file}")
    
    edge_path = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
    curr_dir = os.path.abspath('.')
    abs_html = os.path.join(curr_dir, html_file)
    abs_pdf = os.path.join(curr_dir, pdf_file)
    
    print("[*] Invoking Microsoft Edge Headless to render Master PDF...")
    cmd = [
        edge_path,
        '--headless',
        '--disable-gpu',
        '--no-pdf-header-footer',
        f'--print-to-pdf={abs_pdf}',
        abs_html
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and os.path.exists(abs_pdf):
        file_size_mb = os.path.getsize(abs_pdf) / (1024 * 1024)
        print(f"[+] SUCCESS! Master PDF Generated: {abs_pdf}")
        print(f"[+] File Size: {file_size_mb:.2f} MB")
    else:
        print(f"[!] PDF generation failed. Error code: {res.returncode}")
        print(res.stderr)


if __name__ == '__main__':
    build_master_handbook()
