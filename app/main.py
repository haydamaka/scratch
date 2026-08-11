import os
import logging
import threading
from dotenv import load_dotenv
from typing import Any, Dict, Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.r2d2_config import load_r2d2_credentials_into_env, get_env_mode, expand_and_stage_r2d2_yaml
from app.utils.fallback_sync import sync_fallback_with_init
from contextlib import asynccontextmanager

if os.name == "posix":
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Decide environment (dev/uat/prod)
env_mode = os.getenv("ENV", "dev").lower()   # can also come from pod spec/YAML
env_file = f".env.{env_mode}" if env_mode in ["uat", "prod"] else ".env.dev"
print(f"✅ [main] Loading environment from {env_file}")
load_dotenv(dotenv_path=env_file, override=True)

# ==========================
# Fetch R2D2 creds FIRST
# ==========================
load_dotenv(override=False)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("main")

load_r2d2_credentials_into_env()

_required = ("TOKEN_API_URL", "TOKEN_CLIENT_ID", "TOKEN_CLIENT_SECRET", "TOKEN_SCOPE")
_missing = [k for k in _required if not os.getenv(k)]
if _missing:
    raise RuntimeError(f"R2D2 credentials missing after fetch: {', '.join(_missing)}")
log.info("✅Startup ENV=%s - R2D2 creds present (masked logging only).", get_env_mode())

expanded_yaml_path = expand_and_stage_r2d2_yaml()  # uses ./r2d2_credentials.yaml by default
log.info("✅R2D2 credentials staged at %s for ENV=%s", expanded_yaml_path, get_env_mode())
# 3) (Optional) fail fast if TOKEN_* are missing
for k in ("TOKEN_API_URL", "TOKEN_CLIENT_ID", "TOKEN_CLIENT_SECRET", "TOKEN_SCOPE"):
    if not os.getenv(k):
        raise RuntimeError(f"Missing {k} after config-service fetch")

# ==============================
# Sync fallback files every time app initializes
# ==============================
try:
    sync_status = sync_fallback_with_init()
    log.info(f"✅ Fallback sync complete: {sync_status}")
except Exception as e:
    log.error(f"⚠ Fallback sync failed: {e}")


# =========================================================
# Import app modules only after ENV, secrets, and fallbak data are ready
# =========================================================
from app.api.api_v1.endpoints import query
from app.core.runner.database_manager import DatabaseManager
from app.core.middleware import context_cleanup_middleware
from app.api.api_v1.endpoints.external_router import external_router
from app.utils.table_relationship_graph_cache_service import init_graph_cache_service, stop_graph_cache_service
from app.rag.vectordb_loader import init_vector_store_loader, stop_vector_store_loader

@asynccontextmanager
async def lifespan(app):
    """Handle application startup and shutdown"""
    # Startup
    log.info("Application starting up...")
    try:
        init_graph_cache_service()
        log.info("✅ Table relationship graph cache service initialized")
    except Exception as e:
        log.error(f"❌ Failed to initialize graph cache service: {e}")
        # Don't stop app startup, but log the error

    try:
        # Background thread: the catalog load and the keyword-index build run off
        # the startup path, so the service accepts traffic straight away. Table
        # search blocks on it (hybrid_search.search → wait_until_ready); nothing
        # else does.
        init_vector_store_loader()
        log.info("✅ Vector store loader started (background)")
    except Exception as e:
        log.error(f"❌ Failed to start vector store loader: {e}")

    yield  # Application runs here

    # Shutdown
    log.info("Application shutting down...")
    try:
        stop_graph_cache_service()
        log.info("✅ Graph cache service stopped")
    except Exception as e:
        log.error(f"Error stopping graph cache service: {e}")

    try:
        stop_vector_store_loader()
        log.info("✅ Vector store loader stopped")
    except Exception as e:
        log.error(f"Error stopping vector store loader: {e}")

app = FastAPI(title="DC Lite Access", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(context_cleanup_middleware)

current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

app.include_router(external_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
