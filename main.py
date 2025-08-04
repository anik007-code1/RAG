import os
import re
from typing import List, Union
from langchain_community.document_loaders import (
    PDFPlumberLoader,
    UnstructuredImageLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredFileLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

DOCUMENT_STORAGE_PATH = 'document_store/documents/'
os.makedirs(DOCUMENT_STORAGE_PATH, exist_ok=True)

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

EMBEDDING_MODEL = OllamaEmbeddings(model="qwen3:0.6b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="qwen3:0.6b")


def save_uploaded_file(file_path):
    """Dummy placeholder in CLI. Just returns file path."""
    return file_path


def get_document_loader(file_path: str):
    """Get the appropriate document loader based on file extension."""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    loaders = {
        '.pdf': PDFPlumberLoader,
        '.png': UnstructuredImageLoader,
        '.jpg': UnstructuredImageLoader,
        '.jpeg': UnstructuredImageLoader,
        '.docx': Docx2txtLoader,
        '.pptx': UnstructuredPowerPointLoader,
    }
    
    loader_class = loaders.get(file_extension, UnstructuredFileLoader)
    return loader_class(file_path)


def load_documents(file_path: str) -> List[Document]: # it will accept text from various sources 
    try:
        document_loader = get_document_loader(file_path)
        documents = document_loader.load()
        
        cleaned_documents = []
        for doc in documents:
            text = doc.page_content
            text = re.sub(r'\s+', ' ', text) 
            text = text.strip()
            if text:  
                doc.page_content = text
                cleaned_documents.append(doc)
        return cleaned_documents
    except Exception as e:
        print(f"Error loading document: {str(e)}")
        return []


def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)


def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)


def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)


def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})


def main():
    print("🧠 DocuMind AI (CLI Edition)")
    print("Supported formats: PDF, Images (PNG, JPG), Word (DOCX), PowerPoint (PPTX)")
    file_path = input("Enter path to your document: ").strip()
    # file_path = "data.pdf"  # Default for testing

    if not os.path.exists(file_path):
        print("❌ File not found.")
        return

    print("📄 Loading and processing document...")
    saved_path = save_uploaded_file(file_path)
    raw_docs = load_documents(saved_path)
    
    if not raw_docs:
        print("❌ Failed to load or process the document.")
        return
        
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)

    print("✅ Document indexed. You can now ask questions.")

    while True:
        user_input = input("\n❓ Your question (type 'exit' to quit): ").strip()
        if user_input.lower() == "exit":
            print("👋 Exiting. Goodbye!")
            break

        relevant_docs = find_related_documents(user_input)
        ai_response = generate_answer(user_input, relevant_docs)
        final_answer = re.sub(r'<think>.*?</think>', '', ai_response, flags=re.DOTALL).strip()
        print("\n🤖 Answer:")
        print(final_answer)
        print("-" * 50)


if __name__ == "__main__":
    main()
