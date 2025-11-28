from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import uuid
from src.extractor import stream_to_s3,start_textract_job,poll_textract,parse_tables,PREFIX,BUCKET

app = FastAPI()

class DocumentInput(BaseModel):
    document: HttpUrl 

@app.post("/process-pdf")
async def process_pdf(pdf: DocumentInput):
    pdf_url = pdf.document
    return {"success":"true"}
