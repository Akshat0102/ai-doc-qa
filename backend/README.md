# AI Document Q&A Backend

This is the backend service for the AI Document Q&A application. It provides a Retrieval-Augmented Generation (RAG) pipeline to process uploaded documents, generate embeddings, store them in a vector database, and answer user queries based on the uploaded content.

## Features
- **Document Upload**: Upload PDF files to extract and process content.
- **Chunking**: Split document content into smaller chunks for efficient processing.
- **Embedding Generation**: Generate embeddings for document chunks using the Jina AI Embedding API.
- **Vector Store**: Store embeddings and document chunks in a vector database for retrieval.
- **Query Answering**: Answer user queries based on the uploaded documents using OpenAI's GPT 4o-mini model.

## Project Structure
```
backend/
├── .env                 # Environment variables
├── .gitignore           # Git ignore file
├── main.py              
├── requirements.txt     # Python dependencies
├── data/                # Directory for uploaded files
│   └── uploads/         # Directory for user-uploaded files
├── services/            # Service layer
│   └── rag_service.py   # RAG pipeline implementation
├── utils/               # Utility modules
│   ├── chunker.py       # Text chunking logic
│   ├── chunker_token.py # Token-based chunking logic
│   ├── embeddings.py    # Embedding generation logic
│   ├── file_loader.py   # File loading logic
│   └── vector_store.py  # Vector database logic
```

## Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Akshat0102/ai-doc-qa.git
   cd ai-doc-qa/backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the `backend/` directory with the following content:
     ```env
     OPENAI_API_KEY=your_openai_api_key
     JINA_API_KEY=your_jina_api_key
     ```

5. Start the server:
   ```bash
   uvicorn main:app --reload
   ```

6. Access the API documentation:
   - Open your browser and navigate to: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## API Endpoints

### Health Check
- **Endpoint**: `GET /health`
- **Description**: Check if the backend service is running.
- **Response**:
  ```json
  {
    "status": "ok",
    "message": "AI Document Q&A RAG API is running."
  }
  ```

### Upload Document
- **Endpoint**: `POST /upload`
- **Description**: Upload a document for processing.
- **Request**: Multipart file upload.
- **Response**:
  ```json
  {
    "message": "File processed successfully",
    "details": {
      "status": "success",
      "chunks_added": 10
    }
  }
  ```

### Query
- **Endpoint**: `POST /query`
- **Description**: Ask a question based on the uploaded documents.
- **Request**:
  ```json
  {
    "question": "What is the document about?"
  }
  ```
- **Response**:
  ```json
  {
    "answer": "The document is about..."
  }
  ```

## Acknowledgments
- [FastAPI](https://fastapi.tiangolo.com/)
- [Jina AI](https://jina.ai/)
- [OpenAI](https://openai.com/)