import logging
from typing import List, Dict, Optional
import uuid
from datetime import datetime
import asyncio

from ..utils.llm_client import LLMClient
from ..services.vector_store import VectorStoreService
from ..services.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


class ChatEngine:
    def __init__(
        self,
        vector_store: VectorStoreService,
        document_processor: DocumentProcessor,
        llm_client: LLMClient,
    ):
        self.vector_store = vector_store
        self.document_processor = document_processor
        self.llm_client = llm_client
        self.chat_histories = {}
        self.collection_name = "sfn_knowledge"

    def create_chat_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.chat_histories[session_id] = []
        logger.info(f"Created new chat session: {session_id}")
        return session_id

    def get_chat_history(self, session_id: str) -> List[Dict]:
        if session_id not in self.chat_histories:
            logger.warning(f"Chat history for session {session_id} not found.")
            return []
        return self.chat_histories[session_id]

    async def search_knowledge_base(self, query: str, limit: int = 5) -> List[Dict]:
        logger.info(f"Searching knowledge base for query: {query[:50]}...")
        query_embedding = await asyncio.to_thread(
            self.document_processor.get_query_embedding, query
        )
        if not query_embedding:
            logger.warning("Failed to generate query embedding.")
            return []

        search_results = await asyncio.to_thread(
            self.vector_store.search,
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        logger.info(f"Found {len(search_results)} results from knowledge base.")
        return [result.payload for result in search_results if result.payload]


    def generate_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        context_texts = [chunk["text"] for chunk in context_chunks if chunk and "text" in chunk]
        context_combined = "\n\n".join(context_texts)

        prompt = f"""Ты являешься интеллектуальным чат-ботом компании ООО «СФН», специализирующейся на инвестиционных фондах и управлении активами.
        
Твоя задача - предоставлять точные, профессиональные и полезные ответы на вопросы инвесторов о продуктах и услугах компании.

Используй только информацию из предоставленного контекста. Если информации недостаточно, признай это и не придумывай факты.
Отвечай кратко и по существу, в профессиональном тоне.

КОНТЕКСТ:
{context_combined}

ВОПРОС:
{query}

ОТВЕТ:"""
        logger.debug(f"Generated prompt: {prompt[:200]}...")
        return prompt

    async def process_query(self, session_id: Optional[str], query: str) -> Dict:
        if not session_id or session_id not in self.chat_histories:
            logger.info(f"Invalid or missing session_id: {session_id}. Creating new session.")
            session_id = self.create_chat_session()

        logger.info(f"Processing query for session {session_id}: {query[:50]}...")
        context_chunks = await self.search_knowledge_base(query)
        
        prompt = self.generate_prompt(query, context_chunks)
        
        response_content = await self.llm_client.get_completion(prompt)
        
        user_message = {"role": "user", "content": query, "timestamp": datetime.now().isoformat()}
        assistant_message = {"role": "assistant", "content": response_content, "timestamp": datetime.now().isoformat()}

        self.chat_histories[session_id].append(user_message)
        self.chat_histories[session_id].append(assistant_message)
        
        logger.info(f"Generated response for session {session_id}: {response_content[:50]}...")

        return {
            "session_id": session_id,
            "response": response_content,
            "sources": [
                {
                    "text": (
                        (chunk["text"][:150] + "...")
                        if chunk and "text" in chunk and len(chunk["text"]) > 150
                        else (chunk["text"] if chunk and "text" in chunk else "N/A")
                    )
                }
                for chunk in context_chunks
            ],
        }
