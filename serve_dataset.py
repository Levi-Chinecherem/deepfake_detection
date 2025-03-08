# serve_dataset.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import os
from pathlib import Path

app = FastAPI(title="Deepfake Dataset Server")

# Base directory for datasets (adjust to your local path)
BASE_DIR = Path("./dataset")  # Replace with your dataset path

@app.get("/list/{path:path}")
async def list_files(path: str):
    """
    List all files in the specified directory.
    
    Args:
        path (str): Directory path relative to BASE_DIR (e.g., 'dataset/frames/train/real/').
    
    Returns:
        list: List of file names in the directory.
    """
    dir_path = BASE_DIR / path
    if not dir_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {path}")
    files = [f.name for f in dir_path.iterdir() if f.is_file()]
    return files

@app.get("/{path:path}")
async def serve_file(path: str):
    """
    Serve a file from the local dataset directory.
    
    Args:
        path (str): File path relative to BASE_DIR (e.g., 'dataset/frames/train/real/real_0.jpg').
    
    Returns:
        FileResponse: The requested file.
    """
    file_path = BASE_DIR / path
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)