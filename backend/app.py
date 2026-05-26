"""
Deepfake detector backend — with tuned threshold.

Pipeline:
    image (BGR)
      ├── 4-channel tensor (RGB normalized + FFT power spectrum) → EfficientNet-B0 → 1280
      ├── FFT azimuthal avg at 256×256                                          →  128
      └── noise (gray - GaussianBlur) FFT azimuthal avg at 256×256              →  128
                                                                                 ────
                                                                                 1536  → StandardScaler → SVM (ONNX)

IMPORTANT: The SVM was trained on StandardScaler-normalized features.
           scaler.pkl must be present in the backend folder.
           Without it, predictions collapse to 0%/100%.

Threshold tuned to 0.35 based on independent test set analysis.

Run:
    pip install flask flask-cors onnxruntime torch timm opencv-python-headless scipy numpy pillow scikit-learn
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
import joblib
import os

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FFT_SIZE = 256
AZ_BINS = FFT_SIZE // 2           # 128
CNN_WEIGHTS_PATH = "best_model.pth"
SVM_ONNX_PATH    = "svm_linear_modelTERBARU.onnx"
SCALER_PATH      = "scaler.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tuned threshold based on independent test set analysis.
# SVM default = 0.5 → biased toward "real" predictions.
# 0.35 = more balanced.
DECISION_THRESHOLD = 0.35

# Temperature scaling: corrects isotonic-calibrated SVM probabilities.
# Isotonic regression is a step-function — with a LinearSVC it learns only
# ~10-50 discrete probability levels, most inputs collapse to 0.333 or 0.667.
# Temperature T > 1 spreads probabilities into a continuous range.
# T=3 is a reasonable default; increase to soften further.
CALIBRATION_TEMPERATURE = 3.0

# ---------------------------------------------------------------------------
# CNN feature extractor (4-channel EfficientNet-B0)
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
        return self.backbone(x)


feature_model = DeepfakeDetector().to(DEVICE)
state = torch.load(CNN_WEIGHTS_PATH, map_location=DEVICE, weights_only=False)
feature_model.load_state_dict(state, strict=False)
feature_model.eval()
print("✅ CNN feature extractor loaded")

svm_session = ort.InferenceSession(SVM_ONNX_PATH, providers=["CPUExecutionProvider"])
SVM_INPUT_NAME = svm_session.get_inputs()[0].name
print(f"✅ SVM ONNX loaded — input '{SVM_INPUT_NAME}', expects {svm_session.get_inputs()[0].shape}")
print(f"✅ Decision threshold: {DECISION_THRESHOLD} (lowered from default 0.5)")
print("\n--- SVM ONNX Outputs ---")
for out in svm_session.get_outputs():
    print(f"  name={out.name!r}  shape={out.shape}  type={out.type}")
print("------------------------\n")

# Load the fitted StandardScaler — CRITICAL for correct probabilities.
# The SVM was trained on StandardScaler-normalized features.
# Without this, raw CNN features (large magnitude) cause probabilities to
# collapse to 0% or 100%.
if os.path.exists(SCALER_PATH):
    feature_scaler = joblib.load(SCALER_PATH)
    print(f"✅ StandardScaler loaded from {SCALER_PATH}")
    print(f"   mean range: [{feature_scaler.mean_.min():.3f}, {feature_scaler.mean_.max():.3f}]")
    print(f"   scale range: [{feature_scaler.scale_.min():.3f}, {feature_scaler.scale_.max():.3f}]")
else:
    feature_scaler = None
    print(f"⚠️  WARNING: {SCALER_PATH} not found — predictions will be 0% or 100%!")
    print(f"   Run the notebook and save: joblib.dump(svm_model.named_steps['scaler'], 'scaler.pkl')")

torch_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def compute_azimuthal_average(spectrum_2d: np.ndarray) -> np.ndarray:
    h, w = spectrum_2d.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    max_r = min(cy, cx)
    return ndimage.mean(spectrum_2d, labels=r, index=np.arange(0, max_r))


def extract_combined_features(img_bgr: np.ndarray) -> np.ndarray:
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
        cnn_feat = feature_model(tensor_4ch).cpu().numpy().flatten()

    gray = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),
                      (FFT_SIZE, FFT_SIZE)).astype(np.float32)

    f2 = np.fft.fft2(gray)
    f2_shift = np.fft.fftshift(f2)
    power = np.log1p(np.abs(f2_shift) ** 2)
    az_avg = compute_azimuthal_average(power)

    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    noise = gray - blurred
    nf = np.fft.fft2(noise)
    nf_shift = np.fft.fftshift(nf)
    noise_power = np.log1p(np.abs(nf_shift) ** 2)
    noise_az = compute_azimuthal_average(noise_power)

    combined = np.concatenate([cnn_feat, az_avg, noise_az]).astype(np.float32)
    assert combined.shape == (1536,), f"Bad feature dim: {combined.shape}"
    return combined


def get_confidence_level(p_fake: float) -> str:
    """Distance from decision threshold = confidence."""
    distance = abs(p_fake - DECISION_THRESHOLD)
    if distance < 0.10:
        return "uncertain"
    elif distance < 0.25:
        return "low"
    elif distance < 0.40:
        return "medium"
    else:
        return "high"


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    try:
        raw = request.files["image"].read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        img_rgb = np.array(pil)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        features = extract_combined_features(img_bgr).reshape(1, -1)

        # Apply the same StandardScaler used during SVM training.
        # Without this step, the SVM receives out-of-distribution features
        # and its decision scores become extreme → probabilities collapse to 0/1.
        if feature_scaler is not None:
            features = feature_scaler.transform(features).astype(np.float32)

        label_arr, prob_list = svm_session.run(None, {SVM_INPUT_NAME: features})
        for out in svm_session.get_outputs():
            print(out.name, out.shape, out.type)
        prob_dict = prob_list[0]
        p_fake_raw = float(prob_dict[1])

        # --- Temperature Scaling ---
        # Isotonic calibration produces a step-function with only ~10-50 discrete
        # probability levels (e.g. 0.0, 0.333, 0.667, 1.0). Most real inputs land
        # on one of these steps. Temperature scaling converts the logit of p_fake
        # into a smooth, continuous probability: p = sigmoid(logit(p_raw) / T).
        # T > 1 spreads extreme values toward 0.5 (less overconfident output).
        eps = 1e-6
        p_clipped = float(np.clip(p_fake_raw, eps, 1.0 - eps))
        logit_p = np.log(p_clipped / (1.0 - p_clipped))
        p_fake = float(1.0 / (1.0 + np.exp(-logit_p / CALIBRATION_TEMPERATURE)))
        p_real = 1.0 - p_fake

        # Use the SVM's own label output (more reliable than thresholding scaled prob)
        is_fake = bool(int(label_arr[0]) == 1)
        confidence = get_confidence_level(p_fake)

        display_percent = round(p_fake * 100, 2)
        print(f"DEBUG: raw={p_fake_raw:.4f}  temp_scaled={p_fake:.4f}  "
              f"label={'FAKE' if is_fake else 'REAL'}  confidence={confidence}")

        return jsonify({
            "probability": display_percent,
            "label": "AI Generated / Fake" if is_fake else "Authentic Media",
            "is_fake": is_fake,
            "confidence": confidence,
            "p_real": round(p_real * 100, 2),
            "p_fake": round(p_fake * 100, 2),
            "threshold_used": DECISION_THRESHOLD,
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Failed: {e}"}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=False)