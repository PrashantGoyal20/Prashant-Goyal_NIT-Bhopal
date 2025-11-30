# Prashant-Goyal_NIT-Bhopal
The bills are received as scanned photos and pdfs (running into 30-40 pages). Bills have to be reviewed by processing agents and appropriate amounts have to be extracted to be entered into the system. We are digitizing/automating this process of bill data extraction - extracting the line item data and bill totals


# app.py — PDF/Image → Gemini JSON Extractor

Short: FastAPI service that uploads a PDF/image to Google Gemini (genai) and asks the model to return a strict JSON invoice/line-item extraction per a provided prompt.

## Features
- Fetches a file by URL (PDF or image).
- Detects MIME and file type.
- Uploads file to genai (Gemini) files API.
- Calls Gemini model (gemini-2.5-flash) with a strict prompt requiring a JSON schema.
- Parses model output and returns the JSON or a standardized failure object.

## Requirements
- Python 3.9+
- Packages (pip): fastapi, uvicorn, httpx, pillow, python-dotenv, google-genai (and dependencies)
- A valid Gemini (genai) client configuration in environment variables (see genai docs).

Example:
pip install fastapi uvicorn httpx pillow python-dotenv google-genai

## Environment
- .env should contain any credentials needed by google.genai client (per your genai SDK setup).
- No local OCR/Tesseract required — model handles parsing.

## How to run
From the src directory:
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

## API
POST /process
- Body: JSON { "document": "<public_file_url>" }
  - Use a publicly reachable URL (or one accessible from the server).
- Responses:
  - 200: JSON object produced by model (should follow the strict schema in PROMPT).
  - 400: fetch or type-detection errors.
  - 500/502: upload/model/parsing errors. On parse failure a structured failure JSON is returned inside the HTTP error detail.

Example curl:
curl -X POST "http://localhost:8000/process" -H "Content-Type: application/json" -d '{"document":"https://example.com/sample.pdf"}'

## Notes & Troubleshooting
- The service relies on genai file upload API and model; ensure your genai client is configured and has network access.
- If the model returns non-JSON or extra text, the service attempts to extract JSON from output. If that fails, it returns a structured failure object with model_raw_output.
- Increase httpx timeout if uploading large PDFs.

## Security
- Do not expose this publicly without auth — it will fetch arbitrary URLs and call external APIs.


# solution2.py — Local OCR-based PDF/Image Invoice Parser

Short: FastAPI skeleton that performs local OCR and heuristic table parsing (Tesseract + OpenCV + PyMuPDF) to extract bill line items into a JSON schema.

## Features
- Fetches a file from a URL.
- Detects PDF vs image and renders PDF pages to images (PyMuPDF).
- Preprocesses images (resize, denoise, threshold) with OpenCV.
- Runs Tesseract OCR to get word-level data, groups words into rows and columns, heuristically extracts item rows (name, qty, rate, amount).
- Produces the same JSON schema as app.py (is_success, token_usage, data.pagewise_line_items, total_item_count).

## Requirements
- Python 3.9+
- System: Windows (Tesseract guidance included), but works on other OS after installing dependencies.
- Tesseract OCR installed (Windows example path set by WINDOWS_TESSERACT_PATH).
- Python packages:
  - fastapi, uvicorn, httpx, python-dotenv, pillow, pytesseract, opencv-python, numpy, PyMuPDF (fitz), pydantic

Install example:
pip install fastapi uvicorn httpx python-dotenv pillow pytesseract opencv-python numpy pymupdf pydantic

## Tesseract (Windows)
- Install Tesseract-OCR (https://github.com/tesseract-ocr/tesseract).
- Update WINDOWS_TESSERACT_PATH in the file to match install location, or set tesseract path appropriately in code.

## How to run
From the src directory:
uvicorn solution2:app --host 0.0.0.0 --port 8001 --reload

## API
POST /process
- Body: JSON { "document": "<public_file_url>" }
- Returns the parsed JSON (same schema as PROMPT), with token_usage set to -1s (local processing).

Example curl:
curl -X POST "http://localhost:8001/process" -H "Content-Type: application/json" -d '{"document":"https://example.com/sample.pdf"}'

## Caveats & Notes
- This file contains partial/placeholder logic and many heuristic routines — review and test on sample PDFs.
- OCR quality depends on image resolution, preprocessing, and Tesseract accuracy. Tweak preprocess parameters (max_dim, thresholding) as needed.
- Header detection and column alignment are heuristic — expect to refine for edge case layouts.
- If processing PDFs, large documents may require more memory/time; adjust httpx timeout and image zoom/scale.

## Debugging
- Log OCR outputs and grouped rows to inspect column assignment issues.
- Visualize intermediate images (preprocessed, thresholded) when tuning preprocessing.
- For production, add authentication and rate limits before exposing the endpoint.