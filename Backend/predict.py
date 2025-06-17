from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

# --- Initialize FastAPI App ---
app = FastAPI(title="Eye Disease Detection API")

# --- Enable CORS for frontend communication ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ For production, replace "*" with your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load the trained Keras model ---
model = load_model("../model/eye_model.keras")  # adjust if needed
class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# --- Image Preprocessing Function ---
def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_arr = np.array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr

# --- Root Endpoint (for testing/health check) ---
@app.get("/")
def read_root():
    return {"message": "👁️ Eye Disease Detection API is running."}

# --- Prediction Endpoint ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png"]:
            return JSONResponse(
                content={"error": "Only .jpg or .png files are supported."},
                status_code=400
            )

        # Read and preprocess image
        contents = await file.read()
        processed_img = preprocess(contents)

        # Predict
        prediction = model.predict(processed_img)
        pred_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return JSONResponse(content={
            "prediction": class_names[pred_class],
            "confidence": f"{confidence * 100:.2f}%"
        })

    except Exception as e:
        return JSONResponse(
            content={"error": f"Internal Server Error: {str(e)}"},
            status_code=500
        )
