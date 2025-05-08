from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import logging
import os
import json
from datetime import datetime
import time

from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStoreService
from app.services.chat_engine import ChatEngine
from app.utils.llm_client import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

document_processor = DocumentProcessor()
vector_store = VectorStoreService()
llm_client = LLMClient()
chat_engine = ChatEngine(vector_store, document_processor, llm_client)

app = FastAPI(title="SFN AI Chat Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHAT_HISTORY_DIR = "chat_histories"
COLLECTION_NAME = "sfn_knowledge"
DATA_FILE = "app/data/sfn_data.txt"


@app.on_event("startup")
async def startup_event():
    try:
        os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
        load_chat_histories()

        collections = vector_store.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if COLLECTION_NAME not in collection_names:
            logger.info(f"Creating knowledge base from {DATA_FILE}")
            success = document_processor.create_index(
                COLLECTION_NAME, DATA_FILE, vector_store
            )
            if success:
                logger.info("Knowledge base created successfully")
            else:
                logger.error("Failed to create knowledge base")
    except Exception as e:
        logger.error(f"Error during startup: {e}")


def load_chat_histories():
    try:
        if not os.path.exists(CHAT_HISTORY_DIR):
            return

        for filename in os.listdir(CHAT_HISTORY_DIR):
            if filename.endswith(".json"):
                session_id = filename.replace(".json", "")
                file_path = os.path.join(CHAT_HISTORY_DIR, filename)

                with open(file_path, "r", encoding="utf-8") as f:
                    chat_engine.chat_histories[session_id] = json.load(f)

        logger.info(f"Loaded {len(chat_engine.chat_histories)} chat sessions from disk")
    except Exception as e:
        logger.error(f"Error loading chat histories: {e}")


def save_chat_history(session_id: str, history: List[Dict]):
    try:
        file_path = os.path.join(CHAT_HISTORY_DIR, f"{session_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving chat history for session {session_id}: {e}")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    request_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{os.urandom(4).hex()}"
    client_host = request.client.host if request.client else "unknown"
    request_path = request.url.path
    if not request_path.startswith("/static/"):
        logger.info(
            f"Request [{request_id}] from {client_host}: {request.method} {request_path}"
        )

        if request.method != "GET":
            try:
                body = await request.body()
                if body:
                    body_text = body.decode("utf-8")
                    logger.info(f"Request body [{request_id}]: {body_text}")
            except Exception as e:
                logger.error(f"Error logging request body: {e}")

    response = await call_next(request)
    if not request_path.startswith("/static/"):
        process_time = time.time() - start_time
        logger.info(
            f"Response [{request_id}]: status={response.status_code}, time={process_time:.4f}s"
        )

    return response


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    sources: List[Dict]


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest, background_tasks: BackgroundTasks):
    try:
        result = await chat_engine.process_query(chat_request.session_id, chat_request.query)

        session_id = result["session_id"]
        history = chat_engine.get_chat_history(session_id)
        if history:
            background_tasks.add_task(save_chat_history, session_id, history)

        return result
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error while processing chat request.")

@app.get("/api/sessions/list")
async def list_sessions_endpoint():
    sessions_summary = []
    sorted_session_ids = sorted(
        chat_engine.chat_histories.keys(),
        key=lambda sid: (
            chat_engine.chat_histories[sid][0].get("timestamp", "")
            if chat_engine.chat_histories.get(sid)
            else ""
        ),
        reverse=True,
    )

    for session_id in sorted_session_ids:
        history = chat_engine.chat_histories.get(session_id, [])
        if history:
            first_message_timestamp = history[0].get("timestamp", "N/A")
            last_message_timestamp = history[-1].get("timestamp", "N/A")
            message_count = len(history)
            first_user_query = next((msg.get("content", "N/A") for msg in history if msg.get("role") == "user"), "N/A")

            sessions_summary.append({
                "session_id": session_id,
                "first_message_timestamp": first_message_timestamp,
                "last_message_timestamp": last_message_timestamp,
                "message_count": message_count,
                "title": first_user_query[:50] + "..." if len(first_user_query) > 50 else first_user_query
            })
        else:
            sessions_summary.append({
                "session_id": session_id,
                "first_message_timestamp": "N/A (empty history)",
                "last_message_timestamp": "N/A",
                "message_count": 0,
                "title": "Empty Session"
            })

    return {"sessions": sessions_summary}


@app.get("/api/sessions/{session_id}/history")
async def get_chat_history_endpoint(session_id: str):
    history = chat_engine.get_chat_history(session_id)
    return {"session_id": session_id, "history": history}


@app.post("/api/sessions")
async def create_session():
    session_id = chat_engine.create_chat_session()
    return {"session_id": session_id}


@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    try:
        with open("app/static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Index page not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error loading index page.")


@app.get("/history", response_class=HTMLResponse)
async def read_history_viewer():
    try:
        with open("app/static/history_viewer.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
         raise HTTPException(status_code=404, detail="History page not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error loading history page.")