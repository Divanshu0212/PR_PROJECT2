from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from rho.api.jobs import JobStore
from rho.extraction import extract
from rho.ingestion import ingest
from rho.models.api import JobStatus, OptimizeJobRequest, ParseResponse

app = FastAPI(title="rho")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_jobs = JobStore()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/parse", response_model=ParseResponse)
async def parse(file: UploadFile):
    data = await file.read()
    try:
        md, prov = ingest(data, file.filename or "resume.txt")
        resume = extract(md, prov)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"parse failed: {exc}")
    return ParseResponse(structured_resume=resume, provenance_map=prov)


@app.post("/optimize", response_model=JobStatus)
def optimize(req: OptimizeJobRequest):
    job_id = _jobs.create(req)
    return _jobs.get(job_id)


@app.get("/optimize/{job_id}", response_model=JobStatus)
def optimize_status(job_id: str):
    js = _jobs.get(job_id)
    if js is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return js
