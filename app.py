import os
import subprocess
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
import shutil

app = FastAPI()

# Define paths and parameters
weights_path = 'yolo/yolov7-main/runs/train/best.pt'
img_size = 640
conf = 0.20
source_folder = 'dataset/images/train/'
output_folder = 'out/fixed_folder/'

# Ensure folders exist
os.makedirs(source_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Root endpoint to prevent 404 on /
@app.get("/")
async def root():
    return PlainTextResponse("Welcome to the YOLOv7 Object Detection API. Use the /detect endpoint to upload an image.")

# Optional: Add a route for favicon.ico to avoid 404
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return PlainTextResponse("", status_code=204)

# Define the detect function
def detect_and_crop(image_path: str):
    # Run the detection command
    command = [
        'python', 'yolo/yolov7-main/detect.py',
        '--weights', weights_path,
        '--conf-thres', str(conf),
        '--img-size', str(img_size),
        '--source', image_path,
        '--project', 'out/',  # Output directory
        '--name', 'fixed_folder',  # Folder name for results
        '--exist-ok'  # Don't increment folder name
    ]

    # Execute the command and check for errors
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Detection failed: {result.stderr}")

    # Locate the output image in the expected output directory
    output_files = os.listdir(output_folder)
    output_image_path = None
    for file_name in output_files:
        if file_name.endswith(".jpg") or file_name.endswith(".jpeg") or file_name.endswith(".png"):
            output_image_path = os.path.join(output_folder, file_name)
            break

    if not output_image_path or not os.path.exists(output_image_path):
        raise HTTPException(status_code=404, detail="Output image not found.")

    return output_image_path

# FastAPI endpoint to accept an image, perform detection, and return the processed image
@app.post("/detect")
async def detect_endpoint(file: UploadFile = File(...)):
    # Save the uploaded file to the source folder
    input_image_path = os.path.join(source_folder, 'input_image.jpg')
    with open(input_image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Perform detection and get the path to the output image
    try:
        output_image_path = detect_and_crop(input_image_path)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

    # Return the output image as a response
    return FileResponse(output_image_path, media_type="image/jpeg")
