import logging
import sys
import os
from dotenv import load_dotenv
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

# =======================
# Load environment variables
# =======================
load_dotenv(".env")

# =======================
# Logging Configuration
# =======================
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        stream=sys.stdout
    )

# =======================
# Configuration Variables
# =======================
FILE_PATH = os.getenv("FILE_PATH")
QUERY = "Who is the first licensor and his/her name?"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ Ensure both environment variables are present
if not FILE_PATH or not os.path.exists(FILE_PATH):
    logging.error(f"❌ FILE_PATH is not set correctly or file does not exist: {FILE_PATH}")
    sys.exit(1)

if not OPENAI_API_KEY:
    logging.error("❌ OPENAI_API_KEY is not set in the environment.")
    sys.exit(1)

# =======================
# Retriever Class
# =======================
class Retriever:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.db = None
        self.text_chunks = []

    def load_and_split(self):
        with open(self.file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n\nPage {page_num + 1}:\n{page_text}"

        docs = [Document(page_content=text, metadata={"source": self.file_path})]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=25,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        split_docs = splitter.split_documents(docs)
        self.text_chunks = [doc.page_content for doc in split_docs]
        return split_docs

    def build_vector_store(self, texts):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db = Chroma.from_documents(texts, embedding=embeddings)

    def vector_search(self, query, k=5):
        return self.db.similarity_search(query, k=k)

    def build_bm25_index(self):
        return BM25Okapi([doc.split() for doc in self.text_chunks])

    def bm25_search(self, query, k=5):
        bm25 = self.build_bm25_index()
        query_tokens = query.split()
        scores = bm25.get_scores(query_tokens)
        ranked_indices = np.argsort(scores)[::-1]

        for i in ranked_indices[:k]:
            print(f"Document {i} contents: {self.text_chunks[i]}")

        return [Document(page_content=self.text_chunks[i]) for i in ranked_indices[:k]]

    def combined_search(self, query, k=5):
        vector_results = self.vector_search(query, k=k)
        bm25_results = self.bm25_search(query, k=k)

        seen = set()
        combined_results = []
        for doc in vector_results + bm25_results:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                combined_results.append(doc)

        return combined_results

    def precision_recall_at_k_combined(self, vector_indices, bm25_indices, ground_truth_indices, k=5):
        combined_indices = list(dict.fromkeys(vector_indices + bm25_indices))
        retrieved_set = set(combined_indices[:k])
        ground_truth_set = set(ground_truth_indices)

        precision = len(retrieved_set & ground_truth_set) / k
        recall = len(retrieved_set & ground_truth_set) / len(ground_truth_set) if ground_truth_set else 0

        return precision, recall

# =======================
# RAGChain Class
# =======================
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

    def run(self, query: str):
        combined_results = self.retriever.combined_search(query)
        combined_results = sorted(combined_results, key=lambda doc: doc.page_content)

        logging.info("\nRetrieved Documents (Sorted by Content):")
        for doc in combined_results:
            logging.info(f"Page Content: {doc.page_content[:100]}...")

        context = "\n\n".join([doc.page_content for doc in combined_results])
        return self.chain.run({"context": context, "question": query})

# =======================
# Main Execution
# =======================
def main():
    retriever = Retriever(FILE_PATH)
    docs = retriever.load_and_split()
    retriever.build_vector_store(docs)

    rag = RAGChain(retriever, OPENAI_API_KEY)
    answer = rag.run(QUERY)
    print("\n✅ Final Answer:\n", answer)

    vector_results = retriever.vector_search(QUERY, k=5)
    bm25_results = retriever.bm25_search(QUERY, k=5)

    bm25_indices = [
        i for i, chunk in enumerate(retriever.text_chunks)
        if any(chunk == doc.page_content for doc in bm25_results)
    ]
    vector_indices = [
        i for i, chunk in enumerate(retriever.text_chunks)
        if any(chunk == doc.page_content for doc in vector_results)
    ]
    ground_truth_indices = [0, 1]

    precision, recall = retriever.precision_recall_at_k_combined(
        vector_indices, bm25_indices, ground_truth_indices, k=5
    )

    logging.info(f"\n📊 Hybrid Precision@5: {precision:.2f}, Recall@5: {recall:.2f}")
    print(f"\n📊 Hybrid Precision@5: {precision:.2f}, Recall@5: {recall:.2f}")

if __name__ == "__main__":
    main()

