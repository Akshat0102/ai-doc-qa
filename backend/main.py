import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Document Q&A - Backend")

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"], 
                   allow_headers=["*"],
                   )

@app.get("/")
async def root():
    return {"status": "A-OK", "message": "AI Doc Q&A Backend is running."}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    contents = await file.read()
    size = len(contents)

    return {"filename": file.filename, "content-type": file.content_type, "size_bytes": size}