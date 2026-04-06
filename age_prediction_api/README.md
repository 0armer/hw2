# Age Prediction API

This is a very simple face recognition and age prediction server built using FastAPI and `deepface`. 

## Project Structure
```text
age_prediction_api/
├── main.py
├── requirements.txt
└── README.md
```

## Setup & Running the Server

1. **Create and Activate a Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Mac/Linux
   source venv/bin/activate
   ```

2. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the FastAPI Server**
   ```bash
   uvicorn main:app --reload
   ```

## Usage

Once the server is running on `http://127.0.0.1:8000`, you can view the automatic interactive documentation by visiting:
**http://127.0.0.1:8000/docs**

Here, you can easily test the endpoint `/predict-age/` by uploading an image!

> Note: The first time you make a request, `deepface` will automatically download the lightweight facial recognition weights locally.
