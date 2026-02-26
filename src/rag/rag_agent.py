"""
Simple RAG agent for genealogical records using LangChain and Google Gemini.
Loads structured record JSON files and answers semantic queries about people and events.
"""

from pathlib import Path
from langchain_community.document_loaders import JSONLoader
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

DEFAULT_RECORDS_DIR = Path(__file__).resolve(
).parents[2] / "data" / "structured_records"


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


class GenealogyRAGAgent:
    """Simple RAG agent for genealogical records."""

    def __init__(self, structured_records_path: str = None):
        """
        Initialize the RAG agent.
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
        print(f"Found {len(json_files)} JSON files")

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
                        doc.metadata["source_file"] = json_file.name

                self.documents.extend(entries)
                total_records += len(entries)

            except Exception as e:
                print(f"Error loading {json_file} with JSONLoader: {e}")
                continue

        print(f"Loaded {total_records} genealogical records")
        return total_records

    def build_vector_store(self) -> None:
        """Build the in-memory vector store using FAISS."""
        if not self.documents:
            raise ValueError("No documents loaded")
        print(f"Building vector store with {len(self.documents)} documents...")

        # Split documents if needed
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        split_docs = text_splitter.split_documents(self.documents)
        print(f"Split into {len(split_docs)} chunks")

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
        print("Vector store created successfully")


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(runtime: GenealogyRAGAgent, k: int = 5):
    """Builds a middleware-based RAG agent from a populated vector store."""
    if runtime.vector_store is None:
        raise ValueError(
            "Agent vector_store is not built. Call build_vector_store().")

    vector_store = runtime.vector_store

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
        retrieved_docs = vector_store.similarity_search(last_query, k=k)
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        return (
            "You are a genealogy assistant. Use only the provided context to answer accurately. "
            "If the information isn't in the records, say you don't know."
            f"\n\nContext:\n{docs_content}"
        )

    runtime_agent = create_agent(
        runtime.llm,
        tools=[],
        middleware=[prompt_with_context]
    )
    return runtime_agent


if __name__ == "__main__":
    runtime = GenealogyRAGAgent(structured_records_path=DEFAULT_RECORDS_DIR)
    runtime.load_records()
    runtime.build_vector_store()

    rag_chain = create_rag_chain(runtime)

    while True:
        user_query = input("\nAsk about a record (or 'exit'): ")
        if user_query.lower() in ['exit', 'quit']:
            break

        response = rag_chain.invoke(
            {"messages": [{"role": "user", "content": user_query}]}
        )

        answer = ""
        if isinstance(response, dict) and "messages" in response and response["messages"]:
            last_msg = response["messages"][-1]
            answer = getattr(last_msg, "content", str(last_msg))
        else:
            answer = getattr(response, "content", str(response))

        print(f"\nAnswer: {answer}")
