import os
from fastapi import Header, HTTPException


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("PROCWISE_API_KEY")
    if not expected:
        return  # auth disabled when no key configured (current default)
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
