<h1 align="center">
  <br>
  🎬
  <br>
  Snipflow
</h1>

<p align="center">
  a smart web application that automatically transforms long videos into short highlight clips and creates beautiful photo collages without the need for manual editing.
</p>

**Live Demo:** [https://snipflow-crtr.onrender.com](https://snipflow-crtr.onrender.com)

---

### Features

- **Automated Video Editing:**
  - **Audio Energy Detection:** Automatically identifies the loudest, most exciting moments in your video.
  - **AI Transcription:** Uses OpenAI's Whisper to understand speech and locate key phrases.
  - **Smart Scene Cuts:** Analyzes camera transitions so your clips never cut abruptly in the middle of an action sequence.
  
- **Photo Collage Generator:**
  - **Grid Layout:** Seamlessly stitch multiple photos into a clean, modern grid.
  - **iPhone Photo Support:** Native support for `.HEIC` formats straight from your phone.
  - **Customization:** Add custom background colors, adjust spacing gaps, and apply custom text watermarks.

- **Dynamic UI:**
  - **Drag-and-Drop Interface:** Easily upload media with a beautifully designed, distraction-free UI.
  - **Real-Time Progress Tracking:** Watch the backend process your video step-by-step directly from the web browser window.

### Technologies Used

- **Backend:** Python, FastAPI, Uvicorn
- **AI & Media Engine:** OpenCV, Librosa, MoviePy, OpenAI Whisper
- **Image Processing:** Pillow, Pillow-HEIF
- **Frontend:** HTML5, Vanilla CSS, Vanilla JavaScript

### Local Setup and Installation

Follow these steps to get the application running on your local machine.

#### 1. Prerequisites
- Python 3.9+
- FFmpeg installed on your system path (optional but highly recommended for fast video rendering).

#### 2. Clone the Repository
Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/Snipflow.git
cd Snipflow
```

#### 3. Install Dependencies
Create a virtual environment (optional but highly recommended) and install all required Python packages:

```bash
python -m venv venv

venv\Scripts\activate

source venv/bin/activate

pip install -r requirements.txt
```

#### 4. Run the Server
Start the FastAPI backend server:

```bash
uvicorn main:app --reload
```

#### 5. Open the Application
You do not need a complex frontend framework! Simply double-click and open `index.html` or `app.html` directly in your web browser and start uploading.
