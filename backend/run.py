#!/usr/bin/env python3
"""
NeuroScreen Backend — Entry Point

Run the FastAPI server:
    python backend/run.py
    python backend/run.py --host 0.0.0.0 --port 8000

Or via uvicorn directly:
    uvicorn backend.app.main:app --host localhost --port 8000 --reload
"""

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="NeuroScreen FastAPI Server")
    parser.add_argument("--host", default="localhost", help="Host to bind (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is not installed. Install it with:")
        print("  pip install uvicorn")
        sys.exit(1)

    print(f"Starting NeuroScreen backend on {args.host}:{args.port}")
    uvicorn.run(
        "backend.app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
