# Tamil OCR Project

This is a comprehensive Tamil Optical Character Recognition (OCR) system. It features a modern web-based frontend, a robust API backend, and a custom-trained Convolutional Recurrent Neural Network (CRNN) model designed entirely for recognizing Tamil characters.

## Project Structure

The repository is organized into three main component directories:

- **`frontend/ocr-ui/`**: A React application powered by Vite and Tailwind CSS. It provides an intuitive user interface for uploading target images and viewing Tamil OCR extraction results interactively.
- **`backend/`**: A Python-based backend server using FastAPI that processes image uploads, communicates with the Machine Learning model, and streams back the extracted text reliably.
- **`ml/`**: Machine learning scripts and utilities for training the core CRNN model. This encompasses downloading Tamil fonts, compiling robust synthetic datasets, defining the model architecture, and comprehensive evaluation scripts.

## Core Features
### 1. Custom CRNN Architecture
A fine-tuned Convolutional Recurrent Neural Network specially configured for sequence prediction, optimized extensively for the curvilinear shapes of Tamil characters.
### 2. Streamlined Real-time Backend Engine
Utilizes lightweight endpoints for instantaneous communication between the model and the UI, parsing the text accurately while maintaining low latency.
### 3. Beautiful & Responsive Frontend
Designed using state-of-the-art styling constraints utilizing Tailwind CSS, bringing a modern and slick experience across desktop and mobile devices.

## Getting Started

### 1. Training the Model (`ml/`)
To train or test the model locally, configure the environment in the `ml/` directory.

```bash
cd ml
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
pip install -r requirements.txt
```
*(Make sure to run the dataset generation steps provided in subsequent scripts prior to invoking `train_crnn.py`)*

### 2. Starting the Backend API (`backend/`)
The backend serves the trained model directly to the React interface. You should copy or place the generated `tamil_crnn_v2.pth` model and `vocab.json` into the `ml` namespace required by the server.

```bash
cd backend
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
pip install -r requirements.txt
python main.py
```
The FastAPI documentation defaults to bounding on `http://localhost:8000/docs`.

### 3. Launching the Frontend UI (`frontend/ocr-ui/`)
Initialize the web application node dependencies and trigger the developer server:

```bash
cd frontend/ocr-ui
npm install
npm run dev
```
Navigate to your local application URL (e.g., `http://localhost:5173`) to experience the interface in action!
