# api/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException

# Import our ML logic from the neighbor file
from model import predict_image

app = FastAPI(title="Cats vs Dogs Classifier")

@app.get("/")
def root():
    """Health check — visit http://localhost:8000/ to confirm the API is up."""
    return {"status": "ok", "message": "Cats vs Dogs API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Receive an uploaded image, run it through the model, return JSON."""
    # Reject anything that's not an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read the file's bytes and run our predictor
    image_bytes = await file.read()
    return predict_image(image_bytes)