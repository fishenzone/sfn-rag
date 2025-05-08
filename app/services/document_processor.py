import os
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import logging
import pypdf

from ..config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self):
        self.embedding_model_name = settings.EMBEDDING_MODEL_NAME
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        logger.info("Embedding model loaded successfully")

    def load_document_text(self, file_path: str) -> Optional[str]:
        if os.path.exists(file_path):
            path = file_path
        elif os.path.exists(os.path.join("/app", file_path)):
            path = os.path.join("/app", file_path)
        else:
            logger.error(f"File not found: {file_path}")
            return None

        logger.info(f"Loading document from: {path}")
        ext = os.path.splitext(path)[1].lower()

        try:
            if ext == ".txt":
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
            elif ext == ".pdf":
                reader = pypdf.PdfReader(path)
                content = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        content += page_text + "\n\n"
            else:
                logger.error(f"Unsupported file format: {ext}")
                return None

            logger.info(f"Document loaded: {len(content)} chars")
            return content

        except Exception as e:
            logger.error(f"Error loading document: {e}")
            return None

    def create_index(self, collection_name: str, file_path: str, vector_store) -> bool:
        text = self.load_document_text(file_path)
        if not text:
            return False

        vector_store.recreate_collection(collection_name)
        chunks = self.text_splitter.split_text(text)
        if not chunks:
            logger.warning("No chunks generated")
            return False

        unique_chunks = []
        seen_chunks = set()
        for chunk in chunks:
            if chunk not in seen_chunks:
                unique_chunks.append(chunk)
                seen_chunks.add(chunk)
        
        logger.info(f"Reduced {len(chunks)} chunks to {len(unique_chunks)} unique chunks")

        try:
            embeddings = self.embedding_model.encode(unique_chunks, show_progress_bar=True)

            processed_chunks = []
            for i, (chunk_text, emb) in enumerate(zip(unique_chunks, embeddings)):
                processed_chunks.append(
                    {"text": chunk_text, "embedding": emb.tolist(), "chunk_index": i}
                )

            vector_store.upsert_chunks(collection_name, processed_chunks)
            logger.info(
                f"Successfully indexed {len(unique_chunks)} chunks into {collection_name}"
            )
            return True

        except Exception as e:
            logger.error(f"Error indexing document: {e}")
            return False

    def get_query_embedding(self, query_text: str) -> List[float]:
        try:
            embedding = self.embedding_model.encode(query_text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return None
