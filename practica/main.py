from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
import uuid
import json
from datetime import datetime
from utils.models import process_with_model
from utils.report import generate_pdf_report

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
HISTORY_FILE = "history.json"
REPORT_DIR = "reports"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")
app.mount("/reports", StaticFiles(directory=REPORT_DIR), name="reports")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), mode: str = Form(...)):
    file_id = str(uuid.uuid4())
    ext = file.filename.split(".")[-1]
    filename = f"{file_id}.{ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result_img_path, stats = process_with_model(file_path, mode, file_id)

    report_path = os.path.join(REPORT_DIR, f"{file_id}.pdf")
    generate_pdf_report(file_path, result_img_path, stats, report_path)

    record = {
        "id": file_id,
        "filename": filename,
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
        "result_path": f"/results/{os.path.basename(result_img_path)}",
        "report_path": f"/reports/{file_id}.pdf",
        "stats": stats
    }
    _save_to_history(record)

    return JSONResponse(content=record)

@app.get("/history")
def get_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

@app.get("/report/{file_id}")
def download_report(file_id: str):
    path = os.path.join(REPORT_DIR, f"{file_id}.pdf")
    return FileResponse(path, media_type="application/pdf", filename=f"report_{file_id}.pdf")

def _save_to_history(entry):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    history.append(entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
