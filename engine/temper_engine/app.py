import asyncio
import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from . import __version__, account, accounts, anthropic_api, api_keys, attachments, catalog, chat_import, chats, classifiers, compaction, compare, convert, crashes, daemon, datasets, decisions, devices, discovery, doctor, embedders, engines, export, files, generations, hardware, hub, images, instructions, jobs, knowledge, lan, library, loaded, llamacpp, mcp, mcp_server, memory, model_settings, ollama, presets, prompts, providers, publish, recipes, research, runtime, sandbox, search, skills, sources, speech, stems, storage, tts, tools, training, tunnel, usage

APP_ORIGINS = ["tauri://localhost", "http://tauri.localhost", "http://localhost:1420"]
OPEN_PATHS = {"/api/health", "/api/accounts/status", "/api/accounts/login"}


def ui_folder() -> Path | None:
    """The built interface, for browsers at /ui/: TEMPER_UI, beside the bundled engine (resources/ui), or the repo's dist."""
    here = Path(__file__).resolve().parents[2]
    for candidate in (os.environ.get("TEMPER_UI"), here / "ui", here / "dist"):
        if candidate and (Path(candidate) / "index.html").is_file():
            return Path(candidate)
    return None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    knowledge.watcher.start()
    await asyncio.to_thread(lan.listener.apply)
    await asyncio.to_thread(discovery.apply)
    yield
    await asyncio.to_thread(lan.listener.stop)
    await asyncio.to_thread(tunnel.tunnel.stop)
    await asyncio.to_thread(discovery.stop_all)
    knowledge.watcher.stop()
    runtime.manager.unload()
    compare.pool.unload_all()
    await asyncio.to_thread(mcp.manager.close_all)  # closing a session over HTTP blocks
    jobs.registry.cancel_all()
    knowledge.embedder.stop()
    speech.worker.stop()
    tts.worker.stop()
    images.worker.stop()
    classifiers.worker.stop()
    api_keys.flush()
    usage.flush()


def create_app(token: str = "") -> FastAPI:
    app = FastAPI(title="Temper engine", version=__version__, lifespan=lifespan)
    sources.apply()
    storage.apply()

    @app.middleware("http")
    async def require_token(request: Request, call_next):
        path = request.url.path
        key = account = None
        if token and request.method != "OPTIONS" and path not in OPEN_PATHS and path.startswith(("/api", "/v1", "/mcp")):
            auth = request.headers.get("authorization", "")
            given = auth.removeprefix("Bearer ").strip() or request.headers.get("x-api-key", "")
            # The model API also opens to the API keys made in Connect, and everything to a signed-in account
            # (accounts.py), a member only to chatting; the rest needs the app's own token.
            if hmac.compare_digest(given, token):
                pass
            elif account := accounts.session(given):
                if not account.get("admin") and not accounts.member_may(request.method, path):
                    return JSONResponse({"detail": "Only an admin can do that."}, status_code=403)
            elif not (path.startswith("/v1") and (key := api_keys.find(given))):
                return JSONResponse({"detail": "Missing or wrong API token"}, status_code=401)
        signed_in = accounts.current.set(account)
        try:
            if not path.startswith("/v1") or request.method == "OPTIONS":
                return await call_next(request)
            request.state.api_entry = entry = api_keys.start(path, request.method, key)
            response = await call_next(request)
            api_keys.finish(entry, response.status_code)
            return response
        finally:
            accounts.current.reset(signed_in)

    # Added after the auth middleware so it wraps it and 401s still carry CORS headers.
    # TEMPER_ORIGINS adds origins (comma-separated) for an interface served elsewhere, like the end-to-end tests'.
    extra = [o.strip() for o in os.environ.get("TEMPER_ORIGINS", "").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=APP_ORIGINS + extra,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):
        crash = crashes.from_exception(exc, f"{request.method} {request.url.path}")
        return JSONResponse({"detail": f"Temper hit an unexpected error: {crash['error']}", "crash_id": crash["id"]}, status_code=500)

    @app.get("/api/health")
    def health():
        return {"ok": True, "version": __version__}

    for module in (accounts, attachments, chat_import, chats, compaction, instructions, hardware, doctor, account, sources, storage, catalog, memory, hub, library, convert, ollama, jobs, model_settings, engines, loaded, runtime, anthropic_api, api_keys, generations, compare, presets, prompts, tools, mcp, mcp_server, sandbox, search, skills, training, export, publish, datasets, files, lan, discovery, llamacpp, recipes, research, decisions, knowledge, usage, crashes, daemon, devices, providers, tunnel, classifiers, embedders, images, speech, stems, tts):
        app.include_router(module.router)
    lan.listener.app = app
    tunnel.tunnel.app = app
    discovery.token = token
    # The interface for a browser (`temper serve`): static files, open like any web page; the API it calls still
    # needs the key, which the page asks for.
    if ui := ui_folder():
        app.mount("/ui", StaticFiles(directory=ui, html=True), name="ui")

    return app
