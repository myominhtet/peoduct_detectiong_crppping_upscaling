import gradio as gr
import os
import subprocess
import cv2
import numpy as np

# Define the detect function
def detect_and_crop(input_image):
    # Define paths and parameters
    weights_path = 'yolo/yolov7-main/runs/train/best.pt'
    img_size = 640
    conf = 0.20
    source = 'dataset/images/train/'  # Folder for input images

    # Ensure the input image folder exists
    os.makedirs(source, exist_ok=True)

    # Save the input image to the source directory
    input_image.save(os.path.join(source, 'input_image.jpg'))

    # Run the detection command
    command = [
        'python', 'yolo/yolov7-main/detect.py',
        '--weights', weights_path,
        '--conf-thres', str(conf),
        '--img-size', str(img_size),
        '--source', os.path.join(source, 'input_image.jpg'),
        '--project', 'out/',  # Output directory
        '--name', 'fixed_folder',  # Folder name for results
        '--exist-ok'  # Don't increment folder name
    ]

    # Execute the command
    subprocess.run(command)

    # Load the result image
    output_image_path = 'out/fixed_folder/input_image_upscaled.jpg'

    # Check if the image exists
    if not os.path.exists(output_image_path):
        return "No output image found."

    # Read the output image
    output_image = cv2.imread(output_image_path)

    # Convert BGR (OpenCV format) to RGB (Gradio format)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    return output_image

# Set up the Gradio interface
iface = gr.Interface(
    fn=detect_and_crop,
    inputs=gr.Image(type="pil"),  # Input type
    outputs=gr.Image(type="numpy"),  # Output type
    title="YOLOv7 Object Detection",
    description="Upload an image for object detection and cropping."
)

# Launch the app
iface.launch()
