import base64
import os
from pathlib import Path

import httpx
from ddgs import DDGS
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

app = FastAPI(title="SceneScout")

# Local frontend is served by FastAPI. On Vercel, the frontend is hosted seperately.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("APP_ORIGIN", "http://localhost:8000"),
        "http://localhost:3000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class AskRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    media_type: str = "anime"
    genre: str = "any"


async def ask_openrouter(messages, max_tokens=700):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENROUTER_API_KEY is missing. Add it to your environment variables.",
        )

    model = os.getenv("OPENROUTER_MODEL", "openrouter/free")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("APP_ORIGIN", "http://localhost:8000"),
        "X-Title": "SceneScout",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }

    try:
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
            )
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="Could not reach the AI service.")

    if response.is_error:
        # Keep the useful part of the provider error, but don't dump a huge response.
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter returned {response.status_code}: {response.text[:300]}",
        )

    try:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        raise HTTPException(status_code=502, detail="The AI service sent an unexpected response.")


@app.get("/")
def home():
    # Handy for running the whole project locally from one server.
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend files were not found.")
    return FileResponse(index_file)


@app.get("/api/health")
def health():
    return {"ok": True, "service": "SceneScout"}


@app.get("/api/config")
def config():
    return {"ai_configured": bool(os.getenv("OPENROUTER_API_KEY"))}


@app.post("/api/ask")
async def ask_scene(body: AskRequest):
    system_message = (
        "You are SceneScout, a helpful assistant for finding scenes from anime, "
        "manga, films, TV and games. Read the user's memory carefully. Give a short "
        "explanation, one useful web search query, and a few alternative queries. "
        "Do not make up titles, characters or episode numbers. If you are unsure, "
        "say that clearly. Use these headings: RESPONSE, SEARCH QUERY, ALTERNATE QUERIES."
    )
    user_message = (
        f"Media: {body.media_type}\n"
        f"Genre: {body.genre}\n"
        f"Scene description: {body.text}"
    )

    answer = await ask_openrouter(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
    )
    return {"answer": answer}


@app.post("/api/vision")
async def inspect_screenshot(
    image: UploadFile = File(...),
    hint: str = Form(""),
):
    # Only accept common image uploads and keep the request size reasonable.
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/gif"}
    if image.content_type not in allowed_types:
        raise HTTPException(status_code=415, detail="Please upload a JPG, PNG, WEBP or GIF image.")

    raw_image = await image.read()
    if not raw_image:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(raw_image) > 7 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be smaller than 7 MB.")

    encoded_image = base64.b64encode(raw_image).decode("ascii")
    prompt = (
        "Look at this screenshot and describe it as clues for finding a scene. "
        "Mention the visible characters or objects without guessing their identities, "
        "the action, setting, colours, art style and any text that can be read. "
        "Finish with three search queries. Don't claim an exact source unless it is clear. "
        f"Extra clue from the user: {hint or 'None'}"
    )

    answer = await ask_openrouter(
        [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image.content_type};base64,{encoded_image}"
                    },
                },
            ],
        }],
        max_tokens=800,
    )
    return {"answer": answer}


@app.get("/api/search")
def search_web(q: str, kind: str = "text"):
    query = q.strip()[:400]
    if not query:
        raise HTTPException(status_code=400, detail="Please enter something to search for.")
    if kind not in {"text", "images"}:
        raise HTTPException(status_code=400, detail="Search type must be text or images.")

    try:
        with DDGS() as ddgs:
            if kind == "images":
                rows = list(ddgs.images(query, max_results=12))
                results = [
                    {
                        "title": row.get("title") or "Image result",
                        "url": row.get("url") or row.get("image") or "",
                        "image": row.get("image") or "",
                        "source": row.get("source") or "",
                    }
                    for row in rows
                ]
            else:
                rows = list(ddgs.text(query, max_results=10))
                results = [
                    {
                        "title": row.get("title") or "Search result",
                        "url": row.get("href") or "",
                        "snippet": row.get("body") or "",
                    }
                    for row in rows
                ]
    except Exception:
        # DDG can occasionally rate-limit requests. The frontend can show this message.
        raise HTTPException(status_code=502, detail="Search is temporarily unavailable. Try again in a moment.")

    return {"query": query, "kind": kind, "results": results}
