import io
import os
import re
import math
import traceback
import platform
import shutil
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
from PIL import Image
import fitz  # pymupdf
import cv2
import numpy as np
import pytesseract

load_dotenv()

WINDOWS_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def configure_tesseract():
    pytesseract.pytesseract.tesseract_cmd = WINDOWS_TESSERACT_PATH
    

configure_tesseract()

app = FastAPI()

class DocumentInput(BaseModel):
    document: HttpUrl

class ItemOut(BaseModel):
    item_name: str
    item_amount: float
    item_rate: float
    item_quantity: float
    inferred: Optional[Dict[str, bool]] = {}
    confidences: Optional[Dict[str, float]] = {}

class PageOut(BaseModel):
    page_no: str
    page_type: str
    bill_items: List[ItemOut]

class OutputSchema(BaseModel):
    is_success: bool
    token_usage: Dict[str, int]
    data: Dict[str, Any]

NUM_RE = re.compile(r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+\.\d+|-?\d+')

def extract_numbers(s: str) -> List[str]:
    if not s: return []
    return NUM_RE.findall(s.replace('₹','').replace('Rs','').replace(',',''))

def parse_float_from_string(s: str) -> Optional[float]:
    nums = extract_numbers(s)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(',',''))
    except:
        try:
            return float(nums[-1])
        except:
            return None

def safe_round(v) -> float:
    try:
        return round(float(v), 2)
    except:
        return -1.0

def preprocess_image(pil_img: Image.Image, max_dim=2200):
    img = np.array(pil_img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,25,10)
    coords = np.column_stack(np.where(th < 255))
    if coords.size > 0:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 0.5:
            (h2, w2) = gray.shape[:2]
            M = cv2.getRotationMatrix2D((w2/2, h2/2), angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w2, h2), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            th = cv2.warpAffine(th, M, (w2, h2), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return gray, th, scale

def render_pdf_page(pdf_bytes: bytes, page_no: int=0, zoom=2):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if page_no < 0 or page_no >= len(doc):
        raise IndexError("page_no out of range")
    page = doc[page_no]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

def run_ocr_image(pil_img: Image.Image):
    try:
        ocr = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT, config="--oem 1 --psm 6")
        return ocr
    except Exception as e:
        raise RuntimeError("Tesseract OCR failed. Ensure Tesseract is installed and accessible. " + str(e))

def group_words_to_rows(ocr_dict: Dict[str, List]):
    n = len(ocr_dict['text'])
    words = []
    for i in range(n):
        txt = str(ocr_dict['text'][i]).strip()
        conf = float(ocr_dict['conf'][i]) if str(ocr_dict['conf'][i]).strip() not in ("-1","") else -1.0
        if txt == "" or txt.lower() == "nan":
            continue
        words.append({
            "text": txt,
            "left": int(ocr_dict['left'][i]),
            "top": int(ocr_dict['top'][i]),
            "width": int(ocr_dict['width'][i]),
            "height": int(ocr_dict['height'][i]),
            "conf": conf
        })
    if not words:
        return []
    words = sorted(words, key=lambda w: w['top'])
    rows = []
    current = [words[0]]
    prev_top = words[0]['top']
    for w in words[1:]:
        if abs(w['top'] - prev_top) <= max(10, int(0.02 * (ocr_dict.get('height', [0])[0] if 'height' in ocr_dict else 0))):
            current.append(w)
            prev_top = int((prev_top + w['top'])/2)
        else:
            rows.append(current)
            current = [w]
            prev_top = w['top']
    if current:
        rows.append(current)
    rows = [sorted(r, key=lambda x: x['left']) for r in rows]
    return rows

def find_header_row(rows):
    header_keywords = ['description','desc','qty','quantity','rate','gross','amount','discount','sl#','cpt','code','date']
    for idx, r in enumerate(rows[:8]):
        text = " ".join([w['text'] for w in r]).lower()
        if any(k in text for k in header_keywords):
            return idx
    return None

def compute_column_boundaries(header_row, page_width):
    centers = [w['left'] + w['width']//2 for w in header_row]
    if not centers:
        return [0, page_width]
    mids = []
    for i in range(len(centers)-1):
        mids.append((centers[i] + centers[i+1])//2)
    boundaries = [0] + mids + [page_width]
    return boundaries

def assign_row_to_columns(row_words, boundaries):
    cols = {i: [] for i in range(len(boundaries)-1)}
    for w in row_words:
        cx = w['left'] + w['width']//2
        for i in range(len(boundaries)-1):
            if boundaries[i] <= cx < boundaries[i+1]:
                cols[i].append(w)
                break
    col_texts = []
    for i in range(len(boundaries)-1):
        parts = sorted(cols[i], key=lambda x: x['left'])
        col_texts.append(" ".join([p['text'] for p in parts]).strip())
    return col_texts

def normalize_row_from_cols(col_texts, header_map=None):
    desc_idx = None; qty_idx = None; rate_idx = None; gross_idx = None
    if header_map:
        for idx, label in header_map.items():
            h = label.lower()
            if 'desc' in h or 'description' in h:
                desc_idx = idx
            if 'qty' in h or 'quantity' in h:
                qty_idx = idx
            if 'rate' in h:
                rate_idx = idx
            if 'gross' in h or 'amount' in h:
                gross_idx = idx
    ncols = len(col_texts)
    def has_letters(s): return bool(re.search(r'[A-Za-z]', s))
    def has_digits(s): return bool(re.search(r'\d', s))
    if desc_idx is None:
        for i in range(ncols):
            if has_letters(col_texts[i]):
                desc_idx = i; break
    if gross_idx is None:
        for i in reversed(range(ncols)):
            if has_digits(col_texts[i]):
                gross_idx = i; break
    if qty_idx is None:
        for i in range(ncols):
            if re.search(r'\b\d+\b', col_texts[i]):
                qty_idx = i; break
    if rate_idx is None:
        if gross_idx is not None and desc_idx is not None:
            if gross_idx - desc_idx >= 2:
                rate_idx = gross_idx - 1
        if rate_idx is None:
            for i in range(ncols):
                if i==gross_idx: continue
                if has_digits(col_texts[i]):
                    rate_idx = i; break

    desc = col_texts[desc_idx] if desc_idx is not None and desc_idx < ncols else ""
    qty_s = col_texts[qty_idx] if qty_idx is not None and qty_idx < ncols else ""
    rate_s = col_texts[rate_idx] if rate_idx is not None and rate_idx < ncols else ""
    gross_s = col_texts[gross_idx] if gross_idx is not None and gross_idx < ncols else ""

    def parse_num_text(s):
        s = s.replace(',','').replace('₹','').replace('Rs.','')
        m = re.search(r'-?\d+\.\d+|-?\d+', s)
        if not m: return None
        try:
            return float(m.group(0))
        except:
            return None

    qty = parse_num_text(qty_s)
    rate = parse_num_text(rate_s)
    gross = parse_num_text(gross_s)

    full_row = " ".join(col_texts)
    all_nums = [float(x.replace(',','')) for x in extract_numbers(full_row)] if extract_numbers(full_row) else []
    if gross is None and all_nums:
        gross = all_nums[-1]
    if rate is None and len(all_nums) >= 2:
        rate = all_nums[-2]
    if qty is None:
        for n in all_nums:
            if float(n).is_integer() and 1 <= float(n) <= 1000:
                qty = float(n); break

    inferred = {"item_amount": False, "item_rate": False, "item_quantity": False}
    confidences = {"item_amount": 0.8, "item_rate": 0.8, "item_quantity": 0.8, "item_name": 0.8}

    if (rate is None or rate == 0) and qty and gross:
        try:
            r = gross / qty
            if 0 < r < 1e6:
                rate = safe_round(r)
                inferred['item_rate'] = True
                confidences['item_rate'] *= 0.6
        except:
            pass
    if qty is None and rate and gross:
        try:
            q = gross / rate
            if 0 < q < 1e6:
                if abs(q - round(q)) < 0.02:
                    qty = float(round(q))
                else:
                    qty = safe_round(q)
                inferred['item_quantity'] = True
                confidences['item_quantity'] *= 0.6
        except:
            pass

    item_amount = safe_round(gross) if gross is not None else -1.0
    item_rate = safe_round(rate) if rate is not None else -1.0
    item_qty = float(qty) if qty is not None else -1.0

    desc_clean = re.sub(r'^\d+\s*', '', desc).strip()

    return {
        "item_name": desc_clean or full_row,
        "item_amount": item_amount,
        "item_rate": item_rate,
        "item_quantity": item_qty,
        "inferred": inferred,
        "confidences": confidences
    }

def find_total_from_rows(rows_texts: List[str]):
    candidate = None
    for text in reversed(rows_texts):
        if re.search(r'\b(total|category total|grand total|net total|amount payable)\b', text, re.I):
            v = parse_float_from_string(text)
            if v is not None:
                candidate = v
                break
    return candidate

def process_pil_page(pil_img: Image.Image, page_no: int = 1, url_hint: str = ""):
    gray, th, scale = preprocess_image(pil_img)
    pil_proc = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
    ocr = run_ocr_image(pil_proc)
    rows = group_words_to_rows(ocr)
    if not rows:
        return {"page_no": str(page_no), "page_type": "Bill Detail", "bill_items": []}, 0, None

    header_idx = find_header_row(rows)
    header_row = rows[header_idx] if header_idx is not None else rows[0]
    page_width = pil_proc.width
    boundaries = compute_column_boundaries(header_row, page_width)
    header_map = {}
    for i, w in enumerate(sorted(header_row, key=lambda x: x['left'])):
        cx = w['left'] + w['width']//2
        for j in range(len(boundaries)-1):
            if boundaries[j] <= cx < boundaries[j+1]:
                header_map[j] = w['text']
                break

    items = []
    rows_texts = []
    for ridx, r in enumerate(rows):
        if header_idx is not None and ridx <= header_idx:
            continue
        col_texts = assign_row_to_columns(r, boundaries)
        rows_texts.append(" | ".join(col_texts))
        item = normalize_row_from_cols(col_texts, header_map)
        if item['item_amount'] != -1.0 and not re.search(r'\b(total|category total|subtotal|grand)\b', item['item_name'], re.I):
            items.append(item)

    reported_total = find_total_from_rows(rows_texts)
    sum_items = sum([it['item_amount'] for it in items if it['item_amount'] and it['item_amount']!=-1.0])
    return {"page_no": str(page_no), "page_type": ("Pharmacy" if "pharm" in url_hint.lower() else "Bill Detail"), "bill_items": items}, len(items), (reported_total, sum_items)

def process_url_to_json(url: str, timeout=60):
    import httpx
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(url)
            r.raise_for_status()
            content = r.content
            headers = r.headers
    except Exception as e:
        raise RuntimeError(f"Cannot fetch URL: {e}")

    ct = headers.get("content-type", "").lower()
    kind = None
    if "pdf" in ct or url.lower().endswith(".pdf") or (len(content) >= 4 and content[:4] == b"%PDF"):
        kind = "pdf"
    else:
        try:
            Image.open(io.BytesIO(content)).verify()
            kind = "image"
        except Exception:
            kind = "unknown"

    if kind == "unknown":
        raise RuntimeError("Unknown file type (not PDF or image)")

    pagewise_items = []
    total_items = 0
    reconciliation_issues = []

    if kind == "pdf":
        doc = fitz.open(stream=content, filetype="pdf")
        for pno in range(len(doc)):
            pil_page = render_pdf_page(content, page_no=pno, zoom=2)
            page_res, count, recon = process_pil_page(pil_page, page_no=pno+1, url_hint=url)
            pagewise_items.append(page_res)
            total_items += count
            if recon and recon[0] is not None:
                reported, summed = recon
                if reported is not None and summed is not None:
                    if abs(reported - summed) > max(1.0, 0.02 * reported):
                        reconciliation_issues.append({"page": pno+1, "reported": reported, "sum_items": summed})
    else:
        pil_page = Image.open(io.BytesIO(content)).convert("RGB")
        page_res, count, recon = process_pil_page(pil_page, page_no=1, url_hint=url)
        pagewise_items.append(page_res)
        total_items += count
        if recon and recon[0] is not None:
            reported, summed = recon
            if reported is not None and summed is not None:
                if abs(reported - summed) > max(1.0, 0.02 * reported):
                    reconciliation_issues.append({"page": 1, "reported": reported, "sum_items": summed})

    is_success = True
    if total_items == 0:
        is_success = False
    if len(reconciliation_issues) > 0:
        is_success = False

    response = {
        "is_success": is_success,
        "token_usage": {"total_tokens": -1, "input_tokens": -1, "output_tokens": -1},
        "data": {
            "pagewise_line_items": pagewise_items,
            "total_item_count": total_items
        }
    }
    if reconciliation_issues:
        response["data"]["reconciliation_issues"] = reconciliation_issues
    return response

@app.post("/process")
def process_endpoint(body: DocumentInput):
    url = str(body.document)
    try:
        res = process_url_to_json(url)
        return res
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=400, detail=f"Processing failed: {e}\n{tb}")

