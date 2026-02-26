# voice_client.py
import httpx

BASE_URL = "http://127.0.0.1:8000"  # FastAPI server on your laptop


async def send_prompt(session_id: str, text: str) -> None:
    """
    Tell the voice server to set the current prompt for this session.
    """
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{BASE_URL}/prompt/{session_id}",
            json={"prompt": text},
        )


async def get_user_text(session_id: str) -> str:
    """
    Ask the voice server to wait for the next transcribed user utterance.
    This will block until the phone sends audio and STT completes.
    """
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.get(f"{BASE_URL}/next_text/{session_id}")
        resp.raise_for_status()
        data = resp.json()
        return data["text"].strip()