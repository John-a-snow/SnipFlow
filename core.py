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