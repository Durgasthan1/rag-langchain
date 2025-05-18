import os
import logging
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

import PyPDF2
from rank_bm25 import BM25Okapi
import numpy as np
from dotenv import load_dotenv

# ==========================
# Environment and Logging
# ==========================
load_dotenv()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

PDF_PATH = os.getenv("PDF_PATH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PDF_PATH or not os.path.exists(PDF_PATH):
    raise ValueError("PDF_PATH is missing or the file does not exist. Check .env file.")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing. Set it in .env file.")

# ==========================
# FastAPI App Initialization
# ==========================
app = FastAPI(title="RAG Web Service")

# ==========================
# Input Schema
# ==========================
class QueryRequest(BaseModel):
    query: str
    k: int = 5

# ==========================
# Retriever Class
# ==========================
class Retriever:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.db = None
        self.text_chunks = []
        self.bm25 = None
        self.prepare()

    def load_and_split(self):
        with open(self.file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for i, page in enumerate(reader.pages):
                content = page.extract_text()
                if content:
                    text += f"\n\nPage {i + 1}:\n{content}"

        docs = [Document(page_content=text, metadata={"source": self.file_path})]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=25,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        split_docs = splitter.split_documents(docs)
        self.text_chunks = [doc.page_content for doc in split_docs]
        return split_docs

    def build_vector_store(self, docs: List[Document]):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db = Chroma.from_documents(docs, embedding=embeddings)

    def build_bm25_index(self):
        tokenized = [doc.split() for doc in self.text_chunks]
        self.bm25 = BM25Okapi(tokenized)

    def vector_search(self, query: str, k: int):
        return self.db.similarity_search(query, k=k)

    def bm25_search(self, query: str, k: int):
        query_tokens = query.split()
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = np.argsort(scores)[::-1]
        return [Document(page_content=self.text_chunks[i]) for i in ranked_indices[:k]]

    def combined_search(self, query: str, k: int):
        vector_results = self.vector_search(query, k)
        bm25_results = self.bm25_search(query, k)
        seen = set()
        combined = []
        for doc in vector_results + bm25_results:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                combined.append(doc)
        return combined[:k]

    def prepare(self):
        docs = self.load_and_split()
        self.build_vector_store(docs)
        self.build_bm25_index()

# ==========================
# RAGChain Class
# ==========================
class RAGChain:
    def __init__(self, retriever: Retriever, openai_api_key: str):
        self.retriever = retriever
        self.llm = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "Use the following context to answer the question.\n"
                "Context:\n{context}\n\n"
                "Question: {question}"
            ),
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def run(self, query: str, k: int = 5):
        combined = self.retriever.combined_search(query, k)
        context = "\n\n".join([doc.page_content for doc in combined])
        return self.chain.run({"context": context, "question": query})

# ==========================
# Initialize Components
# ==========================
retriever = Retriever(PDF_PATH)
rag = RAGChain(retriever, OPENAI_API_KEY)

# ==========================
# FastAPI Route
# ==========================
@app.post("/query")
def query_docs(request: QueryRequest):
    try:
        answer = rag.run(request.query, request.k)
        return {"query": request.query, "answer": answer}
    except Exception as e:
        logger.error(f"Error in query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

