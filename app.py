from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

model = load_model("cycle_condition_model.keras")

IMG_SIZE = 224

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def calculate_price(mrp, months_used, prediction):
    depreciation_rate = 0.20
    years_used = months_used / 12
    base_price = mrp * (1 - depreciation_rate * years_used)

    binary_pred = (prediction > 0.5).astype(int)[0]

    rust = binary_pred[0]
    damage_seat = binary_pred[2]
    missing_mudguard = binary_pred[6]

    deduction = 0
    if rust:
        deduction += 800
    if damage_seat:
        deduction += 500
    if missing_mudguard:
        deduction += 600

    return base_price - deduction

@app.post("/predict/")
async def predict(file: UploadFile = File(...), mrp: float = 10000, months_used: int = 12):
    image_bytes = await file.read()
    processed_image = preprocess_image(image_bytes)

    prediction = model.predict(processed_image)

    price = calculate_price(mrp, months_used, prediction)

    return {
        "predictions": prediction.tolist(),
        "estimated_price": price
    }
