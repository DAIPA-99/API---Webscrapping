import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from src.app import get_application

app = get_application()

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.get("/hello")
async def hello():
    return {"message": "Hello World"}

if __name__ == "__main__":
    uvicorn.run("main:app", debug=True, reload=True, port=8080)
