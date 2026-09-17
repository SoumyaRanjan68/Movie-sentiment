"""
PDF Generation Script for Movie Review Sentiment Analysis Project
-----------------------------------------------------------------
Converts the comprehensive documentation and interview handbook into a
beautifully styled, publication-ready PDF document using Microsoft Edge Headless.
"""

import os
import subprocess
import mistune

def build_pdf():
    print("[*] Reading source markdown documents...")
    
    with open('INTERVIEW_NOTES.md', 'r', encoding='utf-8') as f:
        notes_md = f.read()
        
    with open('README.md', 'r', encoding='utf-8') as f:
        readme_md = f.read()

    # Convert markdown to HTML using mistune
    markdown_parser = mistune.create_markdown(plugins=['table', 'strikethrough'])
    notes_html = markdown_parser(notes_md)
    readme_html = markdown_parser(readme_md)

    # HTML Document with executive styling and print CSS
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Movie Review Sentiment Analysis - Interview Master Handbook</title>
<style>
    @page {{
        size: A4;
        margin: 20mm 16mm 20mm 16mm;
        @bottom-right {{
            content: counter(page);
        }}
    }}
    
    body {{
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
        color: #1E293B;
        line-height: 1.6;
        font-size: 10.5pt;
        background-color: #FFFFFF;
    }}
    
    /* Cover Page */
    .cover-page {{
        page-break-after: always;
        padding-top: 45mm;
        text-align: center;
    }}
    
    .cover-badge {{
        display: inline-block;
        background-color: #EEF2FF;
        color: #3730A3;
        font-size: 11pt;
        font-weight: 700;
        padding: 6px 16px;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 20px;
        border: 1px solid #C7D2FE;
    }}
    
    .cover-title {{
        font-size: 28pt;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
        margin-bottom: 12px;
        letter-spacing: -0.02em;
    }}
    
    .cover-subtitle {{
        font-size: 14pt;
        color: #475569;
        margin-bottom: 35px;
        font-weight: 400;
    }}
    
    .cover-divider {{
        width: 100px;
        height: 4px;
        background: linear-gradient(90deg, #2563EB, #4F46E5);
        margin: 0 auto 35px auto;
        border-radius: 2px;
    }}
    
    .cover-meta {{
        margin-top: 40px;
        font-size: 11pt;
        color: #334155;
        line-height: 1.8;
    }}
    
    .meta-box {{
        display: inline-block;
        text-align: left;
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 20px 30px;
        margin-top: 20px;
    }}
    
    /* Headings */
    h1 {{
        font-size: 18pt;
        font-weight: 700;
        color: #0F172A;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 8px;
        margin-top: 28pt;
        margin-bottom: 14pt;
        page-break-after: avoid;
    }}
    
    .page-break-before {{
        page-break-before: always;
    }}
    
    h2 {{
        font-size: 14pt;
        font-weight: 700;
        color: #1E3A8A;
        margin-top: 20pt;
        margin-bottom: 10pt;
        page-break-after: avoid;
    }}
    
    h3 {{
        font-size: 12pt;
        font-weight: 700;
        color: #334155;
        margin-top: 16pt;
        margin-bottom: 8pt;
        page-break-after: avoid;
    }}
    
    h4 {{
        font-size: 11pt;
        font-weight: 700;
        color: #1E293B;
        margin-top: 14pt;
        margin-bottom: 6pt;
        page-break-after: avoid;
    }}
    
    p, li {{
        color: #334155;
        margin-bottom: 8pt;
    }}
    
    ul, ol {{
        padding-left: 22px;
        margin-bottom: 12pt;
    }}
    
    /* Tables */
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 14pt 0;
        font-size: 9.5pt;
        page-break-inside: avoid;
    }}
    
    th {{
        background-color: #0F172A;
        color: #FFFFFF;
        font-weight: 600;
        text-align: left;
        padding: 8px 10px;
        border: 1px solid #0F172A;
    }}
    
    td {{
        padding: 7px 10px;
        border: 1px solid #CBD5E1;
        color: #334155;
    }}
    
    tr:nth-child(even) {{
        background-color: #F8FAFC;
    }}
    
    /* Code blocks */
    pre {{
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 10px 14px;
        font-family: 'Consolas', 'Courier New', Courier, monospace;
        font-size: 8.5pt;
        line-height: 1.45;
        overflow-x: auto;
        margin: 10pt 0;
        page-break-inside: avoid;
        color: #0F172A;
    }}
    
    code {{
        font-family: 'Consolas', 'Courier New', Courier, monospace;
        background-color: #F1F5F9;
        color: #0F172A;
        padding: 2px 5px;
        border-radius: 4px;
        font-size: 9pt;
    }}
    
    pre code {{
        background-color: transparent;
        padding: 0;
        color: inherit;
    }}
    
    /* Blockquotes / Callouts */
    blockquote {{
        border-left: 4px solid #3B82F6;
        background-color: #EFF6FF;
        margin: 12pt 0;
        padding: 10px 14px;
        border-radius: 0 6px 6px 0;
        color: #1E40AF;
        font-size: 10pt;
    }}
    
    /* Horizontal Rule */
    hr {{
        border: none;
        border-top: 1px solid #E2E8F0;
        margin: 20pt 0;
    }}
    
    /* Links */
    a {{
        color: #2563EB;
        text-decoration: none;
    }}
    
    .kpi-row {{
        display: flex;
        justify-content: space-between;
        margin: 15px 0;
    }}
</style>
</head>
<body>

<!-- COVER PAGE -->
<div class="cover-page">
    <div class="cover-badge">Tata Power AI/ML Interview Preparation</div>
    <div class="cover-title">MOVIE REVIEW SENTIMENT ANALYSIS</div>
    <div class="cover-subtitle">Complete Project Architecture, Theoretical Compendium & Interview Master Handbook</div>
    <div class="cover-divider"></div>
    
    <div class="meta-box">
        <div><strong>Role Target:</strong> Graduate Engineer Trainee (GET) &ndash; AI/ML</div>
        <div><strong>Focus Area:</strong> Natural Language Processing (NLP) & Classical Machine Learning</div>
        <div><strong>Benchmark Dataset:</strong> 50,000 IMDB Large Movie Review Dataset (Balanced)</div>
        <div><strong>Winning Architecture:</strong> TF-IDF (Unigram + Bigram) + Logistic Regression</div>
        <div><strong>Model Generalization:</strong> Accuracy: 90.23% &nbsp;|&nbsp; F1-Score: 90.38% &nbsp;|&nbsp; Latency: &lt; 2.5 ms</div>
        <div><strong>GitHub Repository:</strong> <a href="https://github.com/SoumyaRanjan68/Movie-sentiment">github.com/SoumyaRanjan68/Movie-sentiment</a></div>
    </div>
</div>

<!-- SECTION 1: EXECUTIVE PROJECT OVERVIEW -->
<div class="page-break-before">
    <h1>Project Architecture & System Specifications</h1>
    {readme_html}
</div>

<!-- SECTION 2: INTERVIEW NOTES & MASTER HANDBOOK -->
<div class="page-break-before">
    {notes_html}
</div>

</body>
</html>
"""

    html_file = 'handbook.html'
    pdf_file = 'INTERVIEW_PREPARATION_HANDBOOK.pdf'
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(full_html)
        
    print(f"[+] Wrote styled HTML document to {html_file}")
    
    edge_path = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
    curr_dir = os.path.abspath('.')
    abs_html = os.path.join(curr_dir, html_file)
    abs_pdf = os.path.join(curr_dir, pdf_file)
    
    print("[*] Invoking Microsoft Edge Headless to render PDF...")
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
        print(f"[+] SUCCESS! Generated PDF: {abs_pdf}")
        print(f"[+] Total PDF File Size: {file_size_mb:.2f} MB")
    else:
        print(f"[!] Error generating PDF. Code: {res.returncode}")
        print(res.stderr)

if __name__ == '__main__':
    build_pdf()
