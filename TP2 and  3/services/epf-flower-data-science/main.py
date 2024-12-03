import os
from fastapi import FastAPI
import uvicorn
from fastapi.responses import RedirectResponse
from src.app import get_application
from src.api.routes import data
 
# Set the path to kaggle.json explicitly (assuming it's in the same folder as main.py)
kaggle_json_path = os.path.join(os.path.dirname(__file__), "kaggle.json")
 
# Set the environment variable for the Kaggle API to the directory of kaggle.json
os.environ['KAGGLE_CONFIG_DIR'] = os.path.dirname(kaggle_json_path)
 
app = get_application()
 
# Redirect root endpoint to Swagger UI
@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")
 
# Include the data route with a prefix
app.include_router(data.router, prefix="/api/data", tags=["Dataset"])
 
if __name__ == "__main__":
    uvicorn.run("main:app", debug=True, reload=True, port=8080)
 
 