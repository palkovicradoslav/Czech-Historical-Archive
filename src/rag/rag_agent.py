"""
Simple RAG agent for genealogical records using LangChain and Google Gemini.
Loads structured record JSON files and answers semantic queries about people and events.
"""

from langchain_core.prompts import ChatPromptTemplate
import json
from pathlib import Path

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough


class GenealogyRAGAgent:
    """Simple RAG agent for genealogical records."""

    # Temporary - Maximum embedding requests allowed (reserve 1 for query)
    MAX_EMBEDDING_REQUESTS = 99
    MAX_DOCUMENTS_FOR_EMBEDDING = MAX_EMBEDDING_REQUESTS - 1  # 98 for docs, 1 for query

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

        # Initialize Gemini models
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001")
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
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Each JSON file contains multiple records
                for record_id, record in data.items():
                    doc_text = self._format_record(record, record_id)
                    doc = Document(
                        page_content=doc_text,
                        metadata={
                            "record_id": record_id,
                            "source_file": json_file.name,
                            "record_type": record.get("record_type", "unknown")
                        }
                    )
                    self.documents.append(doc)
                    total_records += 1

            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue

        print(f"Loaded {total_records} genealogical records")
        return total_records

    def _format_record(self, record, record_id):
        """
        Format a record into readable text for embedding.
        """
        lines = [f"Record ID: {record_id}"]

        # Add key information
        if "name" in record and "surname" in record:
            lines.append(f"Person: {record['name']} {record['surname']}")

        if "record_type" in record:
            lines.append(f"Record Type: {record['record_type']}")

        if "birthdate" in record:
            lines.append(f"Birth Date: {record['birthdate']}")

        if "birthplace" in record:
            lines.append(f"Birth Place: {record['birthplace']}")

        if "father" in record:
            lines.append(f"Father: {record['father']}")

        if "father_place_of_living" in record:
            lines.append(f"Father's Place: {record['father_place_of_living']}")

        if "mother" in record:
            lines.append(f"Mother: {record['mother']}")

        if "mother_place_of_living" in record:
            lines.append(f"Mother's Place: {record['mother_place_of_living']}")

        # Add any other fields
        for key, value in record.items():
            if key not in ["name", "surname", "record_type", "birthdate", "birthplace",
                           "father", "father_place_of_living", "mother", "mother_place_of_living"]:
                if value and key not in ["region_id", "file", "person_id"]:
                    lines.append(f"{key.replace('_', ' ').title()}: {value}")

        return "\n".join(lines)

    def build_vector_store(self) -> None:
        """Build the in-memory vector store using FAISS."""
        if not self.documents:
            raise ValueError("No documents loaded. Call load_records() first.")

        print(f"Building vector store with {len(self.documents)} documents...")

        # Split documents if needed
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        split_docs = text_splitter.split_documents(self.documents)
        print(f"Split into {len(split_docs)} chunks")

        # Limit documents to avoid exceeding embedding API quota
        if len(split_docs) > self.MAX_DOCUMENTS_FOR_EMBEDDING:
            split_docs = split_docs[:self.MAX_DOCUMENTS_FOR_EMBEDDING]
            print(
                f"Limiting to {len(split_docs)} chunks to stay within {self.MAX_EMBEDDING_REQUESTS} embedding requests")

        self.vector_store = FAISS.from_documents(
            split_docs,
            self.embeddings
        )
        print("Vector store created successfully")


# Configuration: default records directory at repository root
DEFAULT_RECORDS_DIR = Path(__file__).resolve(
).parents[2] / "data" / "structured_records"


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(agent: GenealogyRAGAgent, k: int = 5):
    """Builds a simple RAG chain from a populated agent."""
    if agent.vector_store is None:
        raise ValueError(
            "Agent vector_store is not built. Call build_vector_store().")

    retriever = agent.vector_store.as_retriever(search_kwargs={"k": k})

    template = """You are a genealogy assistant. Use the provided records to answer the question accurately.
If the information isn't in the records, state that you don't know.

Records:
{context}

Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | agent.llm
    )

    return rag_chain


if __name__ == "__main__":
    agent = GenealogyRAGAgent(structured_records_path=DEFAULT_RECORDS_DIR)
    agent.load_records()
    agent.build_vector_store()

    rag_chain = create_rag_chain(agent)

    while True:
        user_query = input("\nAsk about a record (or 'exit'): ")
        if user_query.lower() in ['exit', 'quit']:
            break

        response = rag_chain.invoke(user_query)
        print(f"\nAnswer: {getattr(response, 'content', response)}")
