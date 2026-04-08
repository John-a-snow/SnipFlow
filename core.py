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
        
        chunk_energy = [e["energy"] for e in energies if e["start"] >= t and e["start"] < end_t]
        if chunk_energy:
            avg_e = sum(chunk_energy) / len(chunk_energy)
            score += (avg_e / max_energy) * 0.4
            
        important_phrases = ["important", "key", "remember", "highlight", "amazing", "woah", "look", "wow"]
        for seg in transcriptions:
            if seg["start"] < end_t and seg["end"] > t:
                text = seg["text"].lower()
                if any(phrase in text for phrase in important_phrases):
                    score += 0.4
                    
        scene_count = len([s for s in scenes if s >= t and s < end_t])
        if scene_count > 0:
            score += 0.2
            
        segments.append({"start": t, "end": end_t, "score": score})
        
    return sorted(segments, key=lambda x: x["score"], reverse=True)

def generate_highlight(video_path: str, segments: list[dict], output_path: str, target_duration: float = 30.0, song_path: str | None = None, replace_audio: bool = False) -> str:
    import moviepy.video.fx.all as vfx
    
    print(f"Creating highlight with target duration up to: {target_duration}s")
    clip = VideoFileClip(video_path)
    
    if clip.h > 480:
        print(f"Downscaling video from {clip.w}x{clip.h} to 480p for faster presentation processing...")
        clip = clip.fx(vfx.resize, height=480)
    
    dynamic_target = min(target_duration, clip.duration)
    print(f"Dynamic target duration set to: {dynamic_target:.2f}s (Original duration: {clip.duration:.2f}s)")
    
    selected_segments = []
    current_duration = 0.0
    
    for seg in segments:
        seg_duration = seg["end"] - seg["start"]
        if current_duration + seg_duration <= dynamic_target:
            selected_segments.append(seg)
            current_duration += seg_duration
        if current_duration >= dynamic_target:
            break
            
    if not selected_segments:
        selected_segments = [{"start": 0, "end": min(clip.duration, dynamic_target)}]

    selected_segments.sort(key=lambda s: s["start"])
    
    def add_zoom_effect(c, zoom_ratio=0.1):
        w, h = c.size
        w, h = int(w), int(h)
        def crop_frame(get_frame, t):
            frame = get_frame(t)
            dur = c.duration or 1.0
            current_zoom = zoom_ratio * (t / dur)
            new_w = max(1, int(w * (1 - current_zoom)))
            new_h = max(1, int(h * (1 - current_zoom)))
            x1 = (w - new_w) // 2
            y1 = (h - new_h) // 2
            cropped = frame[y1:y1+new_h, x1:x1+new_w].astype(np.uint8)
            if cropped.size == 0: return frame
            return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_NEAREST)
        return c.fl(crop_frame)

    processed_clips = []
    for i, seg in enumerate(selected_segments):
        sub = clip.subclip(seg["start"], seg["end"])
        
        sub = add_zoom_effect(sub, zoom_ratio=0.15)
        
        if i > 0:
            sub = sub.crossfadein(0.5)
        
        processed_clips.append(sub)
        
    final_clip = concatenate_videoclips(processed_clips, padding=-0.5 if len(processed_clips) > 1 else 0, method="compose")
    
    if song_path is not None and os.path.exists(str(song_path)):
        try:
            print(f"Applying background music: {song_path} (Replace: {replace_audio})")
            bg_music = AudioFileClip(str(song_path))
            
            if bg_music.duration < final_clip.duration:
                bg_music = audio_loop(bg_music, duration=final_clip.duration)
            else:
                bg_music = bg_music.subclip(0, final_clip.duration)
                
            if replace_audio or not final_clip.audio:
                final_clip.audio = bg_music
            else:
                original_audio = final_clip.audio.volumex(0.4)
                final_clip.audio = CompositeAudioClip([original_audio, bg_music.volumex(1.0)])
        except Exception as e:
            print(f"Error applying background music: {e}")

    print(f"Rendering final highlight with cinematic effects...")
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None, threads=4, preset="ultrafast")
    
    clip.close()
    for c in processed_clips:
        c.close()
    final_clip.close()
    
    return output_path

def process_video_pipeline(video_path: str, output_path: str, target_duration: float = 30.0, song_path: str | None = None, replace_audio: bool = False, progress_callback=None) -> str:
    print(f"--- Starting pipeline for {video_path} ---")
    audio_path = f"{video_path}_audio.wav"
    
    def report(msg):
        print(msg)
        if progress_callback:
            progress_callback(msg)
            
    try:
        report("Step 1/6: Extracting audio...")
        extract_audio(video_path, audio_path)
        
        report("Step 2/6: Detecting audio energy...")
        energies = detect_audio_energy(audio_path)
        
        report("Step 3/6: Transcribing audio with AI...")
        transcriptions = transcribe_audio(audio_path)
        
        report("Step 4/6: Detecting scene shifts...")
        scenes = detect_scene_changes(video_path)
        
        clip = VideoFileClip(video_path)
        duration = clip.duration
        clip.close()
        
        report(f"Step 5/6: Scoring best segments ({duration:.1f}s)...")
        segments = score_segments(energies, transcriptions, scenes, duration)
        
        target_dur = min(target_duration, duration)
            
        report("Step 6/6: Generating final highlight video...")
        result = generate_highlight(video_path, segments, output_path, target_duration=target_dur, song_path=song_path, replace_audio=replace_audio)
            
        print(f"--- Pipeline complete! Output: {output_path} ---")
        return result
    finally:
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                print("Cleaned up temporary audio file.")
            except:
                pass

def create_collage(image_paths: list[str], output_path: str, gap: int = 40, bg_color_hex: str = "#f5f5f5", title: str = "") -> str:
    from PIL import Image, ImageOps, ImageDraw, ImageFont
    import math
    import os
    
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass

    if not image_paths:
        raise ValueError("No images provided for collage.")

    images = []
    last_err = "No images were passed."
    for path in image_paths:
        try:
            img = Image.open(path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)
        except Exception as e:
            last_err = str(e)
            print(f"Skipping image {path} due to error: {e}")

    if not images:
        raise ValueError(f"Could not load any valid images. Error details: {last_err}")

    n = len(images)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    cell_w, cell_h = 800, 800
    
    bg_color_hex = bg_color_hex.lstrip('#')
    try:
        if len(bg_color_hex) == 6:
            r = int(bg_color_hex[0:2], 16)
            g = int(bg_color_hex[2:4], 16)
            b = int(bg_color_hex[4:6], 16)
            bg_color = (r, g, b)
        elif len(bg_color_hex) == 3:
            r = int(bg_color_hex[0] * 2, 16)
            g = int(bg_color_hex[1] * 2, 16)
            b = int(bg_color_hex[2] * 2, 16)
            bg_color = (r, g, b)
        else:
            bg_color = (255, 255, 255)
    except:
        bg_color = (255, 255, 255)

    width = cols * cell_w + (cols + 1) * gap
    base_height = rows * cell_h + (rows + 1) * gap
    
    title_padding = 160 if title else 0
    final_height = base_height + title_padding

    canvas = Image.new("RGB", (width, final_height), bg_color)
    draw = ImageDraw.Draw(canvas)

    for idx, img in enumerate(images):
        col = idx % cols
        row = idx // cols

        img_fit = ImageOps.fit(img, (cell_w, cell_h), Image.Resampling.LANCZOS)
        
        img_bordered = ImageOps.expand(img_fit, border=4, fill=(50, 50, 50))
        
        x = gap + col * (cell_w + gap)
        y = gap + row * (cell_h + gap)
        
        canvas.paste(img_bordered, (x, y))

    if title:
        try:
            brightness = (bg_color[0] * 299 + bg_color[1] * 587 + bg_color[2] * 114) / 1000
            text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
            
            font_path = "C:\\Windows\\Fonts\\arial.ttf"
            if not os.path.exists(font_path):
                font = ImageFont.load_default()
            else:
                font = ImageFont.truetype(font_path, 60)
                
            if hasattr(draw, "textbbox"):
                bbox = draw.textbbox((0, 0), title, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            else:
                text_w, text_h = draw.textsize(title, font=font)
                
            text_x = (width - text_w) // 2
            text_y = base_height - (gap // 2) + (title_padding // 2) - (text_h // 2)
            
            draw.text((text_x, text_y), title, fill=text_color, font=font)
        except Exception as e:
            print(f"Error drawing title: {e}")

    canvas.save(output_path)
    print(f"Collage saved to: {output_path}")
    return output_path

