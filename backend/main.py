from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import dotenv
from services.rag_service import RAGService

app = FastAPI(title="AI Document Q&A RAG API", version="1.0")

dotenv.load_dotenv('.env')   

rag = RAGService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # (You can restrict later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "AI Document Q&A RAG API is running."}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        upload_dir = "data/uploads"
        os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = await rag.ingest_document(file_path)
        return {"message": "File processed successfully", "details": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_rag(payload: dict):
    try:
        question = payload.get("question")
        if not question:
            raise HTTPException(status_code=400, detail="Missing 'question' field")

        answer = rag.generate_answer(question)
        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
