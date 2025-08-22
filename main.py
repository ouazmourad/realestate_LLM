from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel

from rag.llm import LLM
from rag.storage import ingest_docs, ingest_files, query_index

app = FastAPI()
llm = LLM()


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


class IngestJSONRequest(BaseModel):
    client_id: str
    docs: List[Dict[str, str]]


@app.post("/v1/ingest-json")
def ingest_json(req: IngestJSONRequest) -> Dict[str, str]:
    ingest_docs(req.client_id, req.docs)
    return {"status": "ok"}


@app.post("/v1/ingest-files")
async def ingest_files_endpoint(
    client_id: str = Form(...),
    files: List[UploadFile] = File(...),
) -> Dict[str, str]:
    paths: List[Path] = []
    for file in files:
        suffix = Path(file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            paths.append(Path(tmp.name))
    ingest_files(client_id, paths)
    for path in paths:
        path.unlink(missing_ok=True)
    return {"status": "ok"}


class AskRequest(BaseModel):
    client_id: str
    question: str


@app.post("/v1/ask")
def ask(req: AskRequest) -> Dict[str, object]:
    snippets = query_index(req.client_id, req.question)
    context = "\n".join(s["text"] for s in snippets)
    answer = llm.generate(req.question, context)
    return {"answer": answer, "citations": snippets}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
