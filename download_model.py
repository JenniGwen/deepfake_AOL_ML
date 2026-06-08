import requests
import os

# Link to the specific v2 ONNX model
url = "https://huggingface.co/onnx-community/Deep-Fake-Detector-v2-Model-ONNX/resolve/main/onnx/model.onnx"

def download_model():
    print("Downloading Deep-Fake-Detector-v2 (ONNX)... This may take a moment.")
    response = requests.get(url, stream=True)
    with open("model.onnx", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download Complete! 'model.onnx' is ready.")

if __name__ == "__main__":
    download_model()