import os
import uuid
import shutil

try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    bin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ffmpeg_bin")
    os.makedirs(bin_dir, exist_ok=True)
    local_ffmpeg = os.path.join(bin_dir, "ffmpeg.exe")
    if not os.path.exists(local_ffmpeg):
        shutil.copyfile(ffmpeg_exe, local_ffmpeg)
    os.environ["PATH"] += os.pathsep + bin_dir
except ImportError:
    pass
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from core import process_video_pipeline, create_collage

app = FastAPI(title="Snipflow API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("data/inputs", exist_ok=True)
os.makedirs("data/outputs", exist_ok=True)

jobs = {}

@app.get("/api/status/health-check")
async def health_check():
    return {"status": "ok"}

def background_processor(job_id: str, input_path: str, output_path: str, target_duration: float = 30.0, song_path: str | None = None, replace_audio: bool = False):
    try:
        print(f"Background task started for Job {job_id}")
        jobs[job_id]["status"] = "processing"
        
        def progress_cb(msg):
            jobs[job_id]["progress_msg"] = msg
            
        process_video_pipeline(input_path, output_path, target_duration=target_duration, song_path=song_path, replace_audio=replace_audio, progress_callback=progress_cb)
        