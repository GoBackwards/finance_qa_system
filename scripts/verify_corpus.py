"""Check corpus completeness, fiscal-period evidence, text quality and hashes.

This records review flags; it does not alter the source PDFs.
"""
import collections
import hashlib
import json
import re
from pathlib import Path
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    'Semiconductors': 'NVDA AMD INTC QCOM AVGO TXN MU ADI',
    'Software and Cloud Services': 'MSFT ORCL CRM ADBE INTU NOW ADSK CDNS',
    'Retail and Consumer Brands': 'AMZN WMT COST TGT HD LOW NKE SBUX',
    'Food, Beverages, and Household Products': 'KO PEP PG CL KMB MDLZ GIS KHC',
    'Industrial and Machinery Companies': 'CAT DE CMI ITW EMR PH ROK DOV',
}
ALIASES = dict(zip('NVDA AMD INTC QCOM AVGO TXN MU ADI MSFT ORCL CRM ADBE INTU NOW ADSK CDNS AMZN WMT COST TGT HD LOW NKE SBUX KO PEP PG CL KMB MDLZ GIS KHC CAT DE CMI ITW EMR PH ROK DOV'.split(),
    ['NVIDIA','Advanced Micro Devices','Intel','Qualcomm','Broadcom','Texas Instruments','Micron','Analog Devices','Microsoft','Oracle','Salesforce','Adobe','Intuit','ServiceNow','Autodesk','Cadence','Amazon','Walmart','Costco','Target','Home Depot','Lowe','Nike','Starbucks','Coca-Cola','PepsiCo','Procter','Colgate','Kimberly','Mondelez','General Mills','Kraft Heinz','Caterpillar','Deere','Cummins','Illinois Tool Works','Emerson','Parker','Rockwell','Dover']))
COMMON = set('the and of to in for we our is as that with was were from by on are a an or at this its company million year'.split())


def audit(path, ticker, year):
    with pdfium.PdfDocument(str(path)) as doc:
        texts = []
        for page in doc:
            tp = page.get_textpage()
            try:
                texts.append(tp.get_text_range())
            finally:
                tp.close()
                page.close()
    flat = re.sub(r'\s+', ' ', '\n'.join(texts))
    normalized_name = re.sub(r'[^a-z]', '', ALIASES[ticker].lower())
    identity = normalized_name in re.sub(r'[^a-z]', '', flat[:200000].lower())
    period_evidence = []
    flagged = []
    prose_pages = 0
    for index, txt in enumerate(texts, 1):
        normalized = re.sub(r'\s+', ' ', txt)
        pattern = r'(?:for\s*the\s*)?(?:fiscal\s*)?year\s*ended\s*([A-Za-z]+\s*\d{1,2},?\s*(20\d{2}))'
        for match in re.finditer(pattern, normalized, re.I):
            if len(period_evidence) < 16:
                period_evidence.append({'pdf_page': index, 'text': match.group(0), 'year': int(match.group(2))})
        words = re.findall(r'[a-z]{2,}', txt.lower())
        # Noncharacters/control characters can conceal an entire PDF text layer.
        broken_chars = sum(c in '\ufffd\ufffe\uffff' or (ord(c) < 32 and c not in '\n\r\t\f') for c in txt)
        bad_encoding = broken_chars / max(1, len(txt)) > .01
        if len(words) >= 100:
            prose_pages += 1
            fraction = sum(word in COMMON for word in words) / len(words)
            if fraction < 0.06 or bad_encoding:
                flagged.append(index)
        elif bad_encoding:
            flagged.append(index)
    # Retail years can end in Jan/Feb of the following calendar year.
    permitted_end_years = {year, year + 1} if ticker in {'TGT','HD','LOW','CDNS'} else {year}
    strong = [e for e in period_evidence if re.match(r'for\s*the\s*fiscal', e['text'], re.I)]
    selected = strong[0] if strong else period_evidence[0] if period_evidence else None
    period_ok = selected is not None and selected['year'] in permitted_end_years
    quality_flag = len(flagged) > max(8, prose_pages * .25)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'ticker':ticker,'fiscal_year':year,'relative_path':path.relative_to(ROOT).as_posix(),
            'pages':len(texts),'size_bytes':path.stat().st_size,'sha256':digest,
            'company_identity_found':identity,'fiscal_period_check':'passed' if period_ok else 'review',
            'selected_fiscal_period':selected,'fiscal_period_evidence':period_evidence,
            'text_quality_check':'review' if quality_flag else 'passed',
            'low_common_word_pages':flagged,'prose_pages':prose_pages,
            'text_characters':len(flat)}


def main():
    rows, missing = [], []
    for sector,tickers in GROUPS.items():
        for ticker in tickers.split():
            for year in range(2021, 2026):
                path = ROOT/'data'/'raw'/ticker/f'{ticker}_FY{year}_annual_report.pdf'
                if not path.exists():
                    missing.append(path.relative_to(ROOT).as_posix())
                    continue
                row = audit(path,ticker,year)
                row['sector'] = sector
                rows.append(row)
                if not row['company_identity_found'] or row['fiscal_period_check']=='review' or row['text_quality_check']=='review':
                    print('REVIEW',ticker,year,'identity',row['company_identity_found'],'period',row['selected_fiscal_period'],'quality',row['text_quality_check'],flush=True)
    hashes=collections.defaultdict(list)
    for row in rows:
        hashes[row['sha256']].append(row['relative_path'])
    duplicates=[paths for paths in hashes.values() if len(paths)>1]
    result={'expected_pdfs':200,'actual_pdfs':len(rows),'total_pages':sum(r['pages'] for r in rows),
            'total_bytes':sum(r['size_bytes'] for r in rows),'missing':missing,'duplicate_files':duplicates,
            'reports':rows}
    (ROOT/'data'/'validation_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f"Checked {len(rows)} PDFs; missing {len(missing)}; duplicate groups {len(duplicates)}",flush=True)
    print(f"Total pages: {result['total_pages']}; total bytes: {result['total_bytes']}",flush=True)


if __name__=='__main__':
    main()
