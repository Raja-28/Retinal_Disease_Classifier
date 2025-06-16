from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

# App initialization
app = FastAPI(title="Eye Disease Detection API")

# Enable CORS (allow frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for testing/local); restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = load_model("../model/eye_model.keras")
class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# Image preprocessing
def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_arr = np.array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr

# Root endpoint for health check
@app.get("/")
def read_root():
    return {"message": "👁️ Eye Disease Detection API is running."}

# Predict endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png"]:
            return JSONResponse(content={"error": "Only .jpg or .png files are supported."}, status_code=400)

        # Read and process image
        contents = await file.read()
        processed_img = preprocess(contents)

        # Make prediction
        prediction = model.predict(processed_img)
        pred_class = np.argmax(prediction)
        confidence = float(np.max(prediction))

        result = {
            "prediction": class_names[pred_class],
            "confidence": f"{confidence * 100:.2f}%"
        }
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
