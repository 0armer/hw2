import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from deepface import DeepFace

app = FastAPI(
    title="Age Prediction API",
    description="A simple API for face recognition and age prediction using FastAPI and DeepFace.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Age Prediction API. Use the /predict-age/ endpoint to upload an image."}

@app.post("/predict-age/")
async def predict_age(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded is not an image.")

    try:
        # Read image to memory
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
             raise HTTPException(status_code=400, detail="Invalid image file.")

        # DeepFace expects BGR format (same as cv2 default)
        
        # Analyze the image for age prediction
        # actions=['age'] ensures we only load the lightweight age model
        # enforce_detection=True ensures that a face is present in the image
        result = DeepFace.analyze(img_path=img, actions=['age'], enforce_detection=True)

        if isinstance(result, list):
            # If multiple faces are detected, result is a list
            result = result[0]

        age = result.get('age')

        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "predicted_age": age,
            "details": result # Contains bounding box context
        })

    except ValueError as e:
        # DeepFace raises ValueError if no face is detected
        raise HTTPException(status_code=400, detail=f"Face detection failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# To run the server, use: uvicorn main:app --reload
