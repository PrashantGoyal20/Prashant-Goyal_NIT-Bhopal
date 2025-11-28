import io
import os
import httpx
from google import genai
from dotenv import load_dotenv
load_dotenv()

client = genai.Client()

def parse_pdf_from_url(pdf_url: str, prompt: str, model: str = "gemini-2.5-flash"):
    r = httpx.get(pdf_url, timeout=60.0)
    r.raise_for_status()
    pdf_bytes = io.BytesIO(r.content)
    uploaded = client.files.upload(
        file=pdf_bytes,
        config=dict(mime_type="application/pdf")
    )
    response = client.models.generate_content(
        model=model,
        contents=[uploaded, prompt]
    )

    return response.text

