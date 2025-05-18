from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import Retriever, RAGChain
import os
from dotenv import load_dotenv
import logging

# Explicitly load .env file in the current directory
load_dotenv(".env")

FILE_PATH = os.getenv("FILE_PATH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not FILE_PATH or not os.path.isfile(FILE_PATH):
    logging.error(f"❌ FILE_PATH is not set correctly or file does not exist: {FILE_PATH}")
    raise ValueError("Invalid or missing FILE_PATH in .env or file does not exist.")

if not OPENAI_API_KEY:
    logging.error("❌ OPENAI_API_KEY is not set in .env")
    raise ValueError("Missing OPENAI_API_KEY in .env")

app = FastAPI()

retriever = Retriever(FILE_PATH)
docs = retriever.load_and_split()
retriever.build_vector_store(docs)
rag_chain = RAGChain(retriever, OPENAI_API_KEY)

class QueryRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/query")
def ask(request: QueryRequest):
    answer = rag_chain.run(request.question)
    return {"answer": answer}

