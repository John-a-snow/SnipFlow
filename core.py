import os
import numpy as np
import cv2
import whisper
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, CompositeAudioClip
from moviepy.audio.fx.audio_loop import audio_loop

_model_cache = {}

def get_whisper_model(model_size: str = "tiny"):
    if model_size not in _model_cache:
        print(f"Loading Whisper model: {model_size}...")
        _model_cache[model_size] = whisper.load_model(model_size)
    return _model_cache[model_size]

def extract_audio(video_path: str, audio_path: str) -> None:
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        clip.close()
        raise ValueError("Video has no audio track")
    clip.audio.write_audiofile(audio_path, logger=None, fps=16000) 
    clip.close()

def detect_audio_energy(audio_path: str, chunk_duration: float = 1.0) -> list[dict]:
    import librosa
    
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    energies = []
    samples_per_chunk = int(chunk_duration * sr)
    
    for i in range(0, len(y), samples_per_chunk):
        chunk = y[i:i + samples_per_chunk]
        if len(chunk) == 0:
            continue
        energy = np.mean(chunk**2)
        t = i / sr
        energies.append({
            "start": t, 
            "end": min(t + chunk_duration, duration), 
            "energy": float(energy)
        })
        
    return energies

def transcribe_audio(audio_path: str, model_size: str = "tiny") -> list[dict]:
    model = get_whisper_model(model_size)
    print("Starting transcription...")
    result = model.transcribe(audio_path)
    return result["segments"]

def detect_scene_changes(video_path: str, threshold: float = 30.0) -> list[float]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    
    scene_changes = []
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return scene_changes
        
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_count = 0
    
    sample_rate = 15
    
    while True:
        for _ in range(sample_rate - 1):
            cap.grab()
            frame_count += 1
            
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        mean_diff = np.mean(diff)
        
        if mean_diff > threshold:
            scene_changes.append(frame_count / fps)
            
        prev_gray = gray
        
    cap.release()
    return scene_changes

def score_segments(energies: list[dict], transcriptions: list[dict], scenes: list[float], duration: float) -> list[dict]:
    segments = []
    chunk_size = 3.0
    
    energy_vals = [e["energy"] for e in energies]
    max_energy = np.percentile(energy_vals, 95) if energy_vals else 1.0
    if max_energy <= 0: max_energy = 1.0
    
    for t in np.arange(0, duration, chunk_size):
        end_t = min(t + chunk_size, duration)
        if end_t - t < 1.0:
            continue
        score = 0.0