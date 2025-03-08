from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from typing import Optional

app = FastAPI()

# Define the expected API key and header name.
API_KEY = "2rq82hasdflawsk"
API_KEY_NAME = "access_token"

# Create an APIKeyHeader instance; setting auto_error=False allows us to customize the error.
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key: Optional[str] = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key",
        headers={"WWW-Authenticate": "Bearer"},
    )


@app.get("/secure-data")
async def secure_data(api_key: str = Depends(get_api_key)):
    """Protected endpoint that returns a message when a valid API key is provided."""
    return {"message": "Your Bank is Secure with Bank name: ABL and Bank ID: 123456789"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
