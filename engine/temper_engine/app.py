import asyncio
import hmac
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__, account, attachments, catalog, chat_import, chats, compaction, compare, convert, datasets, doctor, engines, export, files, generations, hardware, hub, instructions, jobs, library, loaded, llamacpp, mcp, mcp_server, memory, model_settings, ollama, presets, prompts, recipes, runtime, sandbox, sources, storage, tools, training

APP_ORIGINS = ["tauri://localhost", "http://tauri.localhost", "http://localhost:1420"]
OPEN_PATHS = {"/api/health"}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    runtime.manager.unload()
    compare.pool.unload_all()
    await asyncio.to_thread(mcp.manager.close_all)  # closing a session over HTTP blocks
    jobs.registry.cancel_all()


def create_app(token: str = "") -> FastAPI:
    app = FastAPI(title="Temper engine", version=__version__, lifespan=lifespan)
    sources.apply()
    storage.apply()

    @app.middleware("http")
    async def require_token(request: Request, call_next):
        path = request.url.path
        if token and request.method != "OPTIONS" and path not in OPEN_PATHS and path.startswith(("/api", "/v1", "/mcp")):
            auth = request.headers.get("authorization", "")
            given = auth.removeprefix("Bearer ").strip() or request.headers.get("x-api-key", "")
            if not hmac.compare_digest(given, token):
                return JSONResponse({"detail": "Missing or wrong API token"}, status_code=401)
        return await call_next(request)

    # Added after the auth middleware so it wraps it and 401s still carry CORS headers.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=APP_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health():
        return {"ok": True, "version": __version__}

    for module in (attachments, chat_import, chats, compaction, instructions, hardware, doctor, account, sources, storage, catalog, memory, hub, library, convert, ollama, jobs, model_settings, engines, loaded, runtime, generations, compare, presets, prompts, tools, mcp, mcp_server, sandbox, training, export, datasets, files, llamacpp, recipes):
        app.include_router(module.router)

    return app
