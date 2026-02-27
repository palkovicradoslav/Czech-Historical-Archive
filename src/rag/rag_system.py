"""
Simple RAG system for genealogical records using LangChain, Sentence Transformers, and Google Gemini.
Loads structured record JSON files and answers semantic queries about people and events.
"""

from pathlib import Path
from langchain_community.document_loaders import JSONLoader
import json
import logging
import os
import sys

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langgraph.checkpoint.memory import InMemorySaver
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from typing import Literal

_data_dir = os.environ.get("DATA_DIR")
if _data_dir:
    DEFAULT_RECORDS_DIR = Path(_data_dir) / "structured_records"
else:
    DEFAULT_RECORDS_DIR = Path(__file__).resolve(
    ).parents[2] / "data" / "structured_records"

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(1, os.path.join(THIS_DIR, '..'))

from utils import setup_logger  # NOQA


class SentenceTransformersEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        return embeddings.tolist()

    def embed_query(self, text):
        embedding = self.model.encode(
            [text], convert_to_numpy=True, show_progress_bar=False
        )[0]
        return embedding.tolist()


class GenealogyStructuredAnswer(BaseModel):
    answer: str = Field(description="Direct answer to the user's question.")
    found_in_records: bool = Field(
        description="Whether the answer is supported by retrieved records."
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="Confidence in the answer based on retrieved context."
    )
    supporting_facts: list[str] = Field(
        description="Short evidence bullets grounded in retrieved records."
    )


def format_structured_answer(payload: dict) -> str:
    facts = payload.get("supporting_facts", []) or []
    facts_text = "\n".join(f"  - {fact}" for fact in facts)
    if not facts_text:
        facts_text = "  - (none)"

    return (
        f"Answer: {payload.get('answer', '')}\n"
        f"  Found in records: {payload.get('found_in_records', False)}\n"
        f"  Confidence: {payload.get('confidence', 'low')}\n"
        f"  Supporting facts:\n{facts_text}"
    )


class GenealogyRAGSystem:
    """Simple RAG system for genealogical records."""

    def __init__(self, structured_records_path: str = None):
        """
        Initialize the RAG system.
        """
        if structured_records_path is None:
            repo_root = Path(__file__).resolve().parents[2]
            structured_records_path = repo_root / "data" / "structured_records"

        self.records_path = Path(structured_records_path)
        self.documents = []
        self.vector_store = None
        self.qa_chain = None

        # Local embeddings (MiniLM) + Gemini Pro for inference
        self.embeddings = SentenceTransformersEmbeddings(
            model_name="all-MiniLM-L6-v2")
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

    def load_records(self) -> int:
        """
        Load all JSON files from structured records directory.
        """
        if not self.records_path.exists():
            raise FileNotFoundError(
                f"Records path does not exist: {self.records_path}")

        total_records = 0

        # Find all JSON files in the directory
        json_files = list(self.records_path.glob("*.json"))
        logging.info("Found %s JSON files", len(json_files))

        for json_file in json_files:
            try:
                loader = JSONLoader(
                    str(json_file),
                    text_content=False,
                    jq_schema=".[]"
                )
                entries = loader.load()

                # Convert JSON page_content to readable plain text
                for doc in entries:
                    try:
                        record_data = json.loads(doc.page_content)
                        filtered = {}
                        for key, value in record_data.items():
                            if key in ("region_id", "file"):
                                continue
                            if value is None:
                                continue
                            if isinstance(value, str) and value.strip() == "":
                                continue
                            filtered[key] = value

                        text_parts = [f"{k}: {v}" for k, v in filtered.items()]

                        doc.page_content = "\n".join(text_parts)
                        doc.metadata["source_file"] = json_file.name

                    except json.JSONDecodeError:
                        logging.warning(
                            "Failed to parse JSON content from entry in %s; keeping original content",
                            json_file.name,
                        )
                        doc.metadata["source_file"] = json_file.name

                self.documents.extend(entries)
                total_records += len(entries)

            except (OSError, ValueError, TypeError, KeyError) as e:
                logging.error(
                    "Error loading %s with JSONLoader: %s", json_file, e)
                continue

        logging.info("Loaded %s genealogical records", total_records)
        return total_records

    def build_vector_store(self) -> None:
        """Build the in-memory vector store using FAISS."""
        if not self.documents:
            raise ValueError("No documents loaded")
        logging.info("Building vector store with %s documents...",
                     len(self.documents))

        # Split documents if needed
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        split_docs = text_splitter.split_documents(self.documents)
        logging.info("Split into %s chunks", len(split_docs))

        # Create a FAISS index with the correct embedding dimension
        embedding_dim = len(self.embeddings.embed_query("hello world"))
        index = faiss.IndexFlatL2(embedding_dim)

        vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        if hasattr(vector_store, "add_documents"):
            vector_store.add_documents(split_docs)
        else:
            texts = [d.page_content for d in split_docs]
            vector_store.add_texts(
                texts, metadatas=[d.metadata for d in split_docs])

        self.vector_store = vector_store
        logging.info("Vector store created successfully")


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(rag_runtime: GenealogyRAGSystem, k=5, similarity_threshold=None):
    """
    Builds a middleware-based RAG system.
    """
    if rag_runtime.vector_store is None:
        raise ValueError("System vector_store is not built")

    vector_store = rag_runtime.vector_store

    @dynamic_prompt
    def prompt_with_context(request: ModelRequest) -> str:
        """Inject retrieved context into the system prompt."""
        messages = request.state.get("messages", [])
        if not messages:
            return (
                "You are a genealogy assistant. "
                "If the information isn't in the records, say you don't know."
            )

        last_query = messages[-1].text

        if similarity_threshold is not None:
            results = vector_store.similarity_search_with_relevance_scores(
                last_query, k=100  # Fetch more to filter by threshold
            )
            retrieved_docs = [doc for doc,
                              score in results if score >= similarity_threshold]
        else:
            retrieved_docs = vector_store.similarity_search(last_query, k=k)

        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        return (
            "You are an expert genealogy assistant. You look for people, places, and dates."
            "If there are more than one relevant records, answer based on all of them."
            "Use only the provided context to answer accurately."
            "Work with previous answers and previously provided context to answer follow-up questions."
            "If the information isn't in the records, say you don't know."
            f"\n\nContext:\n{docs_content}"
        )

    checkpointer = InMemorySaver()

    runtime_system = create_agent(
        rag_runtime.llm,
        tools=[],
        middleware=[prompt_with_context],
        checkpointer=checkpointer,
        response_format=GenealogyStructuredAnswer,
    )
    return runtime_system


if __name__ == "__main__":
    setup_logger()

    runtime = GenealogyRAGSystem(structured_records_path=DEFAULT_RECORDS_DIR)
    runtime.load_records()
    runtime.build_vector_store()

    similarity_threshold = 0.3
    rag_chain = create_rag_chain(
        runtime, similarity_threshold=similarity_threshold)
    # or use fixed k: rag_chain = create_rag_chain(runtime, k=10)

    session_config = {"configurable": {"thread_id": "genealogy-cli-session"}}

    while True:
        user_query = input("\nAsk about a record (or 'exit'): ")
        if user_query.lower() in ['exit', 'quit']:
            break

        response = rag_chain.invoke(
            {"messages": [{"role": "user", "content": user_query}]},
            session_config,
        )

        structured = response.get("structured_response") if isinstance(
            response, dict) else None
        if structured is not None:
            if hasattr(structured, "model_dump"):
                payload = structured.model_dump()
            else:
                payload = structured
            print(format_structured_answer(payload))
            continue

        answer = ""
        if isinstance(response, dict) and "messages" in response and response["messages"]:
            last_msg = response["messages"][-1]
            answer = getattr(last_msg, "content", str(last_msg))
        else:
            answer = getattr(response, "content", str(response))

        logging.info("Answer: %s", answer)
