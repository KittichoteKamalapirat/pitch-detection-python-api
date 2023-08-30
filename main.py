from fastapi import FastAPI, UploadFile, File
from pitch_analysis import get_pitch
import numpy as np
import os


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/pitch")
async def analyze_pitch(file: UploadFile):
    
    # Save the uploaded file temporarily
    with open("./assets/temp.wav", "wb") as temp_file:
        temp_file.write(file.file.read())

    # Call the get_pitch function with the saved file
    pitch_list, confidence_list = get_pitch("./assets/temp.wav")

    # Clean up the temporary file (optional)

    os.remove("./assets/temp.wav")
    # Convert pitch_list (tensor) to a NumPy array
    pitch_np_array = np.array(pitch_list)
    pitch_array = pitch_np_array.tolist()

    response = {"pitch": pitch_array, "confidence": confidence_list}
    print(response)
    
    return response