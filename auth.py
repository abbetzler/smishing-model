import os
from fastapi import Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define API key
API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "X-API-Key"

# Ensure API key is set
if not API_KEY:
    raise RuntimeError("API_KEY is not set in the environment variables!")

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)


async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key
