FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir flask flask-cors onnxruntime numpy Pillow scipy torch timm opencv-python-headless scikit-learn joblib

ENV PORT=7860

EXPOSE 7860

CMD ["python", "app.py"]
