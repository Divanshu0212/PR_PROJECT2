from fastapi import FastAPI, Form, UploadFile

from rho.graph import run_pipeline
from rho.models.api import PipelineResponse

app = FastAPI(title="rho")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/optimize", response_model=PipelineResponse)
async def optimize(file: UploadFile, jd_text: str = Form(...)):
    data = await file.read()
    return run_pipeline(data, file.filename or "resume", jd_text)
