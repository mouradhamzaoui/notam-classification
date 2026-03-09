"""
scripts/start_api.py
Lance l'API FastAPI avec Uvicorn.

Usage :
    uv run python scripts/start_api.py
    uv run python scripts/start_api.py --port 8080
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from src.utils.config import Config

def main():
    parser = argparse.ArgumentParser(description="Start NOTAM API")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    cfg  = Config.get()
    host = args.host   or cfg.api.host or "0.0.0.0"
    port = args.port   or int(cfg.api.port or 8000)

    print(f"\n🚀 Starting NOTAM API on http://{host}:{port}")
    print(f"   📖 Docs    : http://localhost:{port}/docs")
    print(f"   ❤️  Health  : http://localhost:{port}/api/v1/health\n")

    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level="info",
    )

if __name__ == "__main__":
    main()