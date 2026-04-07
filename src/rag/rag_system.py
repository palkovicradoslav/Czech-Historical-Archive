from pathlib import Path
from langchain_community.document_loaders import JSONLoader
import json
import logging
import os
import sys
import uuid

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    ScalarQuantization, ScalarQuantizationConfig, ScalarType,
)
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

COLLECTION_NAME = "genealogy_records"
VECTOR_NAME = "content"


class SentenceTransformersEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts):
        return self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64
        ).tolist()

    def embed_query(self, text):
        return self.model.encode(
            [text], convert_to_numpy=True, show_progress_bar=False
        )[0].tolist()


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
    facts_text = "\n".join(f"  - {fact}" for fact in facts) or "  - (none)"
    return (
        f"Answer: {payload.get('answer', '')}\n"
        f"  Found in records: {payload.get('found_in_records', False)}\n"
        f"  Confidence: {payload.get('confidence', 'low')}\n"
        f"  Supporting facts:\n{facts_text}"
    )


class GenealogyRAGSystem:
    """Simple RAG system for genealogical records."""

    def __init__(self, structured_records_path: str = None, qdrant_url: str = None):
        if structured_records_path is None:
            structured_records_path = Path(__file__).resolve(
            ).parents[2] / "data" / "structured_records"

        self.records_path = Path(structured_records_path)
        self.embeddings = SentenceTransformersEmbeddings()
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

        _url = qdrant_url or os.environ.get(
            "QDRANT_URL", "http://localhost:6333")
        # Pass ":memory:" to run without a server
        self.client = QdrantClient(_url)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't already exist."""
        if self.client.collection_exists(COLLECTION_NAME):
            logging.info("Reusing existing collection '%s'", COLLECTION_NAME)
            return

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                # Named vectors allow adding more vector fields later without migration
                VECTOR_NAME: VectorParams(
                    size=self.embeddings.dimension, distance=Distance.COSINE)
            },
            quantization_config=ScalarQuantization(
                # INT8 quantization
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8, quantile=0.99, always_ram=True)
            ),
        )
        logging.info("Created collection '%s'", COLLECTION_NAME)

    def load_records(self) -> int:
        """Load all JSON files, chunk them, embed, and upsert into Qdrant if empty."""

        try:
            collection_info = self.client.get_collection(COLLECTION_NAME)
            if collection_info.points_count > 0:
                logging.info(
                    "Collection '%s' already contains %d points. Skipping upsert.",
                    COLLECTION_NAME,
                    collection_info.points_count
                )
                return 0
        except Exception as e:
            logging.warning("Could not fetch collection info: %s", e)

        if not self.records_path.exists():
            raise FileNotFoundError(
                f"Records path does not exist: {self.records_path}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=64)
        points = []

        for json_file in self.records_path.glob("*.json"):
            try:
                docs = JSONLoader(
                    str(json_file), text_content=False, jq_schema=".[]").load()
            except (OSError, ValueError, TypeError, KeyError) as e:
                logging.error("Error loading %s: %s", json_file, e)
                continue

            for doc in docs:
                try:
                    record = json.loads(doc.page_content)
                    text = "\n".join(
                        f"{k}: {v}" for k, v in record.items()
                        if k not in ("region_id", "file") and v is not None
                        and not (isinstance(v, str) and not v.strip())
                    )
                except json.JSONDecodeError:
                    text = doc.page_content

                for chunk in splitter.split_text(text):
                    points.append(PointStruct(
                        # Deterministic ID from content makes re-runs idempotent
                        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk)),
                        vector={
                            VECTOR_NAME: self.embeddings.embed_query(chunk)},
                        payload={"text": chunk, "source_file": json_file.name},
                    ))

        batch_size = 256
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=points[i:i + batch_size],
                wait=True,
            )

        logging.info("Upserted %d points into '%s'",
                     len(points), COLLECTION_NAME)
        return len(points)

    def search(self, query: str, k: int = 5, score_threshold: float = None) -> list[str]:
        """Return text payloads for the top-k nearest neighbours."""
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.embeddings.embed_query(query),
            using=VECTOR_NAME,
            limit=k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [hit.payload.get("text", "") for hit in results.points]


def create_rag_chain(rag_runtime: GenealogyRAGSystem, k=5, score_threshold=None):
    """Builds an autonomous RAG agent backed by Qdrant with rate-limit protections."""

    @tool
    def search_genealogy_records(search_query: str) -> str:
        """
        Search the genealogy database for people, places, and dates.
        """
        hits = rag_runtime.search(
            search_query,
            k=k,
            score_threshold=score_threshold
        )

        if not hits:
            return f"No records found matching: '{search_query}'. DO NOT search for this specific term again. Move on or stop."

        return "\n\n---\n\n".join(hits)

    system_prompt = (
        "You are an expert genealogy assistant. You look for people, places, and dates. "
        "You have access to a database of structured genealogy records via the `search_genealogy_records` tool.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. ALWAYS use the search tool to find information before answering.\n"
        "2. SEARCH LIMIT: You may search a MAXIMUM of 3 times per user question. "
        "Do not fall into an infinite loop of guessing.\n"
        "3. If you cannot find the answer after 1-3 targeted searches, STOP. "
        "Confidently state that the information is not present in the current records.\n"
        "4. Base your final answer strictly on the retrieved records.\n"
        "5. If there are more than one relevant records, answer based on all of them.\n"
        "6. Work with previous answers and previously provided context to answer follow-up questions."
    )

    return create_agent(
        rag_runtime.llm,
        tools=[search_genealogy_records],
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),
        response_format=GenealogyStructuredAnswer
    )


if __name__ == "__main__":
    setup_logger()

    runtime = GenealogyRAGSystem(
        structured_records_path=DEFAULT_RECORDS_DIR,  # ":memory:" to skip server
    )
    runtime.load_records()

    rag_chain = create_rag_chain(runtime, k=8, score_threshold=0.30)
    session_config = {"configurable": {
        "thread_id": "genealogy-cli-session"}, "recursion_limit": 8}

    while True:
        user_query = input("\nAsk about a record (or 'exit'): ")
        if user_query.lower() in {'exit', 'quit'}:
            break

        response = rag_chain.invoke(
            {"messages": [{"role": "user", "content": user_query}]},
            session_config,
        )

        structured = response.get("structured_response") if isinstance(
            response, dict) else None
        if structured is not None:
            payload = structured.model_dump() if hasattr(
                structured, "model_dump") else structured
            print(format_structured_answer(payload))
            continue

        if isinstance(response, dict) and response.get("messages"):
            last_msg = response["messages"][-1]
            print(getattr(last_msg, "content", str(last_msg)))
        else:
            print(getattr(response, "content", str(response)))
