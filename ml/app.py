from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import easyocr
import numpy as np

app = FastAPI()
reader = easyocr.Reader(['ta','en'], gpu=False)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img_np = np.array(img)

    result = reader.readtext(img_np, detail=0)
    return {"text": result}
