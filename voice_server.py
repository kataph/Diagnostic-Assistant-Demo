# voice_server.py
import asyncio
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, Optional, List

import whisper
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ---- Load Whisper model once ----
model = whisper.load_model("small")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- In-memory session state (only in this server process) ----

@dataclass
class Session:
    prompt_log: List[str]              # all prompts sent to the user
    incoming: asyncio.Queue[str]       # recognized texts from user


_sessions: Dict[str, Session] = {}


def get_session(session_id: str) -> Session:
    if session_id not in _sessions:
        _sessions[session_id] = Session(
            prompt_log=[],
            incoming=asyncio.Queue(),
        )
    return _sessions[session_id]


# ---- Request models ----

class PromptBody(BaseModel):
    prompt: str


# ---- API: phone uploads audio for STT ----

@app.post("/stt/{session_id}")
async def stt(session_id: str, audio: UploadFile = File(...)):
    """
    Phone sends recorded audio here; we run Whisper and enqueue the text.
    """
    session = get_session(session_id)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        data = await audio.read()
        tmp.write(data)
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path, language="en")  # adjust language if needed
    finally:
        os.remove(tmp_path)

    text = result["text"].strip()
    await session.incoming.put(text)
    print(f"[SERVER] STT for {session_id}: {text!r}")
    return {"text": text}


# ---- API: diagnostic code sets prompt for phone ----

@app.post("/prompt/{session_id}")
async def set_prompt(session_id: str, body: PromptBody):
    """
    Append a new prompt message to this session's prompt log.
    """
    session = get_session(session_id)
    session.prompt_log.append(body.prompt)
    print(f"[SERVER] set_prompt: {session_id} -> {body.prompt!r}")
    return {"ok": True, "index": len(session.prompt_log)}


# ---- API: phone polls new prompts since from_index ----

@app.get("/prompt/{session_id}")
async def get_prompt(
    session_id: str,
    from_index: int = Query(0, ge=0),   # start index of unseen prompts
):
    """
    Return all prompts from prompt_log[from_index:].
    Also return next_index so the client can update its cursor.
    """
    session = get_session(session_id)
    log = session.prompt_log
    new_prompts = log[from_index:]
    next_index = len(log)
    print(
        f"[SERVER] get_prompt: {session_id} from_index={from_index} "
        f"-> {len(new_prompts)} new prompts (next_index={next_index})"
    )
    return {"prompts": new_prompts, "next_index": next_index}


# ---- API: diagnostic code waits for next user text (long poll) ----

@app.get("/next_text/{session_id}")
async def next_text(session_id: str):
    """
    Diagnostic code calls this to wait until the user speaks and STT finishes.
    """
    session = get_session(session_id)
    text = await session.incoming.get()
    print(f"[SERVER] next_text: {session_id} -> {text!r}")
    return {"text": text}


# ---- Static: serve the browser client ----

@app.get("/client.html")
async def client_html():
    """
    Serve the phone UI. Expects file static/client.html relative to this script.
    """
    return FileResponse("static/client.html")