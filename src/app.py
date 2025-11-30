from PIL import Image
import io
import json
import traceback
from typing import Dict, Any
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
from google import genai

load_dotenv() 

app = FastAPI()
client = genai.Client()

class DocumentInput(BaseModel):
    document: HttpUrl 

PROMPT= """USER:
I will provide one or more PDF files (or file handles). Parse the PDFs and return exactly one JSON object that conforms to the following schema (no extra keys, JSON only):

{
  "is_success": <boolean>,                       // true only if parsing succeeded and JSON schema is fully followed
  "token_usage": {
    "total_tokens": <integer>,                   // cumulative tokens used by the model across calls if available;
    "input_tokens": <integer>,                   // input tokens in input; 
    "output_tokens": <integer>                   // output tokens used in output;
  },
  "data": {
    "pagewise_line_items": [
      {
        "page_no": "<string>",                   // page number as string (e.g., "1", "2") depending upon the which image number u are parsing from the bundle of images that were provided
        "page_type": "Bill Detail | Final Bill | Pharmacy", // choose exactly one of these 3 labels
        "bill_items": [
          {
            "item_name": "<string>",            // EXACT text for item name as printed in bill; if multiple lines, join with single space
            "item_amount": <float>,             // net amount for the item post discounts, as float
            "item_rate": <float>,               // rate as float; if unknown put -1
            "item_quantity": <float>            // quantity as float; if unknown put -1
          }
        ]
      }
    ],
    "total_item_count": <integer>               // total number of items across all pages (0 if none)
  }
}

RULES / INSTRUCTIONS:
1. Output only JSON exactly matching the schema above. No explanations, no extra fields, no comments.
2. If parsing succeeded for all provided pages and the JSON follows schema -> set "is_success": true. If any required element is missing or the JSON cannot be produced in the required schema, set "is_success": false and still return a JSON object with token_usage fields filled as described and data fields set to empty arrays/zeros where appropriate.
3. Token usage: if you are executed in an environment that provides token counts, place them in the integers. If you cannot determine token counts, set each token_usage entry to -1.
4. page_no: use the page number from the PDF (1-indexed). If page number is unavailable, use the string "unknown".
5. page_type: choose exactly one label per page from: `Bill Detail`, `Final Bill`, or `Pharmacy`. If unsure, pick the best match. Do not invent other labels.
6. bill_items extraction: find all tabular line-items (rows with item description, qty, rate, amount). For each row:
   - `item_name`: copy the item description exactly as it appears (trim whitespace, join multi-line cells with single space).
   - `item_rate`, `item_quantity`, `item_amount`: return numeric floats. Remove currency symbols and commas. If the rate or quantity is not present or ambiguous, set that field to -1 for that item. `item_amount` should always be present if a row is a valid bill item; if missing set to -1.
7. Totals and summary: do NOT place aggregated totals inside `bill_items`. Only include per-row fields in bill_items. The `total_item_count` must equal the count of all `bill_items` across pages.
8. Empty pages: If a page has no bill items, include it with `"bill_items": []` and set `page_type` appropriately.
9. Precision: return float values with up to two decimal places (e.g., 123.45). If input shows integer amounts, still return as float (e.g., 100 -> 100.0).
10. Failure behavior: If parsing fails partially, still return the JSON with `is_success: false`. In `data.pagewise_line_items` include the pages you parsed successfully; for failed pages you may include an entry with `page_no` and an empty `bill_items` list.
11. Strict JSON only: If model outputs anything other than a parseable JSON object exactly following the schema, the consumer will treat the response as failure. So make best effort to conform exactly.
12. Use the attached PDFs as examples: data in these files may include multiple sections (Radiology, Bed charges, Pathology, Pharmacy). Classify pages with pharmacy line items as `Pharmacy`, pages showing diagnostic tests / final summary as `Final Bill` or `Bill Detail` appropriately. The three example files are provided for context (do not print their contents): train_sample_1.pdf, train_sample_2.pdf, train_sample_3.pdf. Use them as examples of layout and possible edge cases. :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}

OUTPUT EXAMPLE (exact JSON structure; fill with real values based on the document):
{
  "is_success": true,
  "token_usage": {
    "total_tokens": -1,
    "input_tokens": -1,
    "output_tokens": -1
  },
  "data": {
    "pagewise_line_items": [
      {
        "page_no": "1",
        "page_type": "Final Bill",
        "bill_items": [
          {
            "item_name": "2D echocardiography",
            "item_amount": 1180.00,
            "item_rate": 1180.00,
            "item_quantity": 1.00
          },
          {
            "item_name": "USG Whole Abdomen Including Pelvis and post Void urine",
            "item_amount": 640.00,
            "item_rate": 640.00,
            "item_quantity": 1.00
          }
        ]
      }
    ],
    "total_item_count": 2
  }
}

IMPORTANT: Return numbers as numeric JSON types (not strings). Strings must be JSON strings. All fields in the schema must be present.
"""    

def detect_file_type(url: str, content: bytes, headers):
    ct = headers.get("content-type", "").lower()
    if "pdf" in ct:
        return "application/pdf", "pdf"
    if ct.startswith("image/"):
        return ct, "image"

    lower = url.lower()
    if lower.endswith(".pdf"):
        return "application/pdf", "pdf"
    if any(lower.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff")):
        ext_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
            ".tiff": "image/tiff",
        }
        return ext_map.get(lower[-5:], "image/jpeg"), "image"

    if content[:4] == b"%PDF":
        return "application/pdf", "pdf"

    try:
        img = Image.open(io.BytesIO(content))
        mime = f"image/{img.format.lower()}"
        return mime, "image"
    except Exception:
        pass

    return "application/octet-stream", "unknown"


@app.post("/process")
def analyze_file(payload: DocumentInput):
    url = str(payload.document) 

    try:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client_http:
            resp = client_http.get(url)
            resp.raise_for_status()
            content = resp.content
            headers = resp.headers
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot fetch URL: {e}")
    mime_type, kind = detect_file_type(url, content, headers)

    if kind == "unknown":
        raise HTTPException(status_code=400, detail="Could not determine file type (not PDF or image)")

    try:
        file_buf = io.BytesIO(content)
        uploaded = genai_files_upload(file_buf, mime_type)
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Failed to upload file to Gemini: {e}\n{tb}")

    prompt = PROMPT
    prompt = f"Detected file type: {kind} (mime: {mime_type}).\n{prompt}"

    try:
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=[uploaded, prompt])
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Model call failed: {e}\n{tb}")

    text_out = resp.text if hasattr(resp, "text") else str(resp)

    parsed_json = None
    try:
        parsed_json = json.loads(text_out)
    except Exception:
        import re
        m = re.search(r"\{(?:.|\n)*\}", text_out)
        if m:
            try:
                parsed_json = json.loads(m.group(0))
            except Exception:
                parsed_json = None
    if parsed_json is None:
        failure_response = {
            "is_success": False,
            "token_usage": {
                "total_tokens": -1,
                "input_tokens": -1,
                "output_tokens": -1
            },
            "data": {
                "pagewise_line_items": [],
                "total_item_count": 0
            },
            "model_raw_output": text_out 
        }
        raise HTTPException(status_code=502, detail=json.dumps(failure_response))

    return parsed_json
def genai_files_upload(file_obj: io.BytesIO, mime_type: str):
    """
    Upload file bytes to Gemini developer files API and return the uploaded file handle/object
    that can be passed to client.models.generate_content. Raises if client is in Vertex mode.
    """
    try:
        uploaded = client.files.upload(file=file_obj, config={"mime_type": mime_type})
        return uploaded
    except Exception as e:
        raise RuntimeError(f"genai files.upload failed: {e}")