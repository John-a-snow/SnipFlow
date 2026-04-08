# Snipflow

**Live Demo:** [https://snipflow-crtr.onrender.com](https://snipflow-crtr.onrender.com)

Snipflow is a web application that takes long videos and automatically edits them down into short highlight reels. It also includes a secondary tool for creating photo grid collages (with native support for iPhone HEIC files).

I built this to experiment with audio analysis, OpenCV, and local AI transcription. Instead of manually scrubbing through long footage to find interesting parts, the backend analyzes the video, looks for loud audio spikes or specific keywords, and stitches the best segments together automatically.

## How it Works

The backend is built with FastAPI and processes videos through a pipeline:
1. **Audio Extraction:** Rips the audio track from the uploaded video.
2. **Energy Detection:** Uses `librosa` to find peaks in volume (loud moments, cheering, etc).
3. **Transcription:** Runs the audio through OpenAI's `Whisper` AI (tiny model) to generate text and find keywords.
4. **Scene Detection:** Uses OpenCV frame differencing to find where the camera cuts are, ensuring the highlight doesn't slice directly through an action shot.
5. **Scoring:** Ranks chunks of the video based on the audio energy and text data.
6. **Rendering:** Stitches the highest-scoring chunks together, applies cinematic camera zooms, handles audio ducking for background music, and exports the final MP4 using MoviePy.

## Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **Processing:** OpenCV, MoviePy, Librosa, Pillow
- **AI:** OpenAI Whisper (running locally)
- **Frontend:** Vanilla HTML, CSS, and Javascript

## Running it Locally

If you want to run this natively on your own machine:

1. Clone the repository.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the API server:
   ```bash
   uvicorn main:app --reload
   ```
4. Simply open `index.html` or `app.html` in your web browser. The frontend will communicate directly with `localhost:8000`.
