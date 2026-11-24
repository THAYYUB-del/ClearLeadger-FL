"""FastAPI ASGI entrypoint for ClearLedger-FL Python backend."""

from __future__ import annotations

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

load_dotenv()

app = FastAPI(
    title="ClearLedger-FL API",
    description="Blockchain-Based Secure and Explainable Federated Model Aggregation for Healthcare Data Sharing",
    version="1.0.0",
)

allowed_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[allowed_origin, "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
