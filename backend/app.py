"""
Deepfake detector backend — FIXED.

Pipeline replica yang sama persis kayak di notebook training:
    image (BGR)
      ├── 4-channel tensor (RGB normalized + FFT power spectrum) → EfficientNet-B0 → 1280
      ├── FFT azimuthal avg at 256×256                                          →  128
      └── noise (gray - GaussianBlur) FFT azimuthal avg at 256×256              →  128
                                                                                 ────
                                                                                 1536  → SVM (ONNX)

Run:
    pip install flask flask-cors onnxruntime torch timm opencv-python-headless scipy numpy pillow
    python app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
import numpy as np
import cv2
import io
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from scipy import ndimage
from PIL import Image

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Config — these MUST match the notebook exactly
# ---------------------------------------------------------------------------
FFT_SIZE = 256
AZ_BINS = FFT_SIZE // 2           # 128
CNN_WEIGHTS_PATH = "best_model.pth"
SVM_ONNX_PATH    = "svm_linear_model.onnx"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# CNN feature extractor (4-channel EfficientNet-B0, copied from training cell)
# ---------------------------------------------------------------------------
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        old_conv = self.backbone.conv_stem
        new_conv = nn.Conv2d(
            4, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        self.backbone.conv_stem = new_conv
        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(1280, 256), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(256, 1),
        )

    def forward(self, x):
        # In feature-extraction mode we return the 1280-dim backbone embedding
        return self.backbone(x)


feature_model = DeepfakeDetector().to(DEVICE)
state = torch.load(CNN_WEIGHTS_PATH, map_location=DEVICE, weights_only=False)
feature_model.load_state_dict(state, strict=False)
feature_model.eval()
print("✅ CNN feature extractor loaded")

svm_session = ort.InferenceSession(SVM_ONNX_PATH, providers=["CPUExecutionProvider"])
SVM_INPUT_NAME = svm_session.get_inputs()[0].name
print(f"✅ SVM ONNX loaded — input '{SVM_INPUT_NAME}', expects {svm_session.get_inputs()[0].shape}")

# Same torchvision transform as the notebook
torch_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Feature engineering (same math as compute_azimuthal_average / extract_combined_features)
# ---------------------------------------------------------------------------
def compute_azimuthal_average(spectrum_2d: np.ndarray) -> np.ndarray:
    h, w = spectrum_2d.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    max_r = min(cy, cx)
    return ndimage.mean(spectrum_2d, labels=r, index=np.arange(0, max_r))


def extract_combined_features(img_bgr: np.ndarray) -> np.ndarray:
    """Returns a (1536,) float32 vector. Mirrors the notebook exactly."""
    # ---- 4-channel input for the CNN ----
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor_3ch = torch_transform(img_rgb)

    gray224 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray224 = cv2.resize(gray224, (224, 224)).astype(np.float32)
    f = np.fft.fft2(gray224)
    f_shift = np.fft.fftshift(f)
    ps = np.log1p(np.abs(f_shift) ** 2)
    ps_norm = (ps - ps.min()) / (ps.max() - ps.min() + 1e-8)
    fft_ch = torch.tensor(ps_norm, dtype=torch.float32).unsqueeze(0)
    tensor_4ch = torch.cat([tensor_3ch, fft_ch], dim=0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        cnn_feat = feature_model(tensor_4ch).cpu().numpy().flatten()   # 1280

    # ---- spectral features at FFT_SIZE ----
    gray = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),
                      (FFT_SIZE, FFT_SIZE)).astype(np.float32)

    f2 = np.fft.fft2(gray)
    f2_shift = np.fft.fftshift(f2)
    power = np.log1p(np.abs(f2_shift) ** 2)
    az_avg = compute_azimuthal_average(power)                          # 128

    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    noise = gray - blurred
    nf = np.fft.fft2(noise)
    nf_shift = np.fft.fftshift(nf)
    noise_power = np.log1p(np.abs(nf_shift) ** 2)
    noise_az = compute_azimuthal_average(noise_power)                  # 128

    combined = np.concatenate([cnn_feat, az_avg, noise_az]).astype(np.float32)
    assert combined.shape == (1536,), f"Bad feature dim: {combined.shape}"
    return combined


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    try:
        # Decode upload → BGR numpy (matches cv2.imread used during training)
        raw = request.files["image"].read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        img_rgb = np.array(pil)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        features = extract_combined_features(img_bgr).reshape(1, -1)

        # SVM ONNX returns:
        #   outputs[0] → np.ndarray of int64 labels, shape (batch,)
        #   outputs[1] → list[dict[int, float]], length = batch
        label_arr, prob_list = svm_session.run(None, {SVM_INPUT_NAME: features})
        prob_dict = prob_list[0]            # e.g. {0: 0.12, 1: 0.88}
        p_fake = float(prob_dict[1])
        p_real = float(prob_dict[0])
        predicted_label = int(label_arr[0])  # 0 = real, 1 = fake

        display_percent = round(p_fake * 100, 2)
        print(f"DEBUG: p_real={p_real:.4f}  p_fake={p_fake:.4f}  label={predicted_label}")

        return jsonify({
            "probability": display_percent,
            "label": "AI Generated / Fake" if predicted_label == 1 else "Authentic Media",
            "is_fake": predicted_label == 1,
            "p_real": round(p_real * 100, 2),
            "p_fake": round(p_fake * 100, 2),
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Failed: {e}"}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=False)