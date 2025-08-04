from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import os
import shutil
from typing import Optional

from main import (
    DOCUMENT_STORAGE_PATH,
    load_documents,
    chunk_documents,
    index_documents,
    find_related_documents,
    generate_answer
)

app = FastAPI(
    title="DocuMind AI API",
    description="API for document processing and question answering using RAG",
    version="1.0.0"
)

# Create storage directory if it doesn't exist
os.makedirs(DOCUMENT_STORAGE_PATH, exist_ok=True)

# In-memory storage for processed documents
processed_documents = {}

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document (PDF, Image, DOCX, PPTX, etc.)
    """
    try:
        # Create a unique filename
        file_extension = os.path.splitext(file.filename)[1].lower()
        file_path = os.path.join(DOCUMENT_STORAGE_PATH, file.filename)
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the document
        raw_docs = load_documents(file_path)
        if not raw_docs:
            raise HTTPException(status_code=400, detail="Failed to process document")
        
        # Chunk and index the documents
        processed_chunks = chunk_documents(raw_docs)
        index_documents(processed_chunks)
        
        # Store the document path for future reference
        processed_documents[file.filename] = file_path
        
        return JSONResponse(
            content={
                "message": "Document processed successfully",
                "filename": file.filename,
                "document_type": file_extension[1:].upper()  # Remove the dot
            },
            status_code=200
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    Ask a question about the processed documents
    """
    try:
        if not processed_documents:
            raise HTTPException(
                status_code=400,
                detail="No documents have been processed. Please upload a document first."
            )
        
        # Find relevant documents and generate answer
        relevant_docs = find_related_documents(request.question)
        answer = generate_answer(request.question, relevant_docs)
        
        return JSONResponse(
            content={
                "question": request.question,
                "answer": answer
            },
            status_code=200
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """
    List all processed documents
    """
    return JSONResponse(
        content={
            "documents": list(processed_documents.keys())
        },
        status_code=200
    )

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """
    Delete a processed document
    """
    if filename not in processed_documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Remove the file
        file_path = processed_documents[filename]
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Remove from processed documents
        del processed_documents[filename]
        
        return JSONResponse(
            content={
                "message": f"Document {filename} deleted successfully"
            },
            status_code=200
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


