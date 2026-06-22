"""
Pipeline:
    image (BGR)
      ├── 4-channel tensor (RGB normalized + FFT power spectrum) → EfficientNet-B0 → 1280
      ├── FFT azimuthal avg at 256×256                                           →  128
      └── noise (gray - GaussianBlur) FFT azimuthal avg at 256×256               →  128
                                                                                  ────
                                                                                  1536 → StandardScaler (inside Pipeline) → SVM / RF

Model files : v5 (preferred):
    best_model_v5.pth       CNN weights  (EfficientNet-B0 4-ch, v5 head: GELU + Dropout 0.4/0.3)
    rbf_svm_v5.onnx         sklearn Pipeline (StandardScaler + RBF SVC)
    features_v5.npz         training features for k-NN type attribution 

Run:
    pip install flask flask-cors onnxruntime torch timm opencv-python-headless scipy numpy pillow scikit-learn joblib
    python app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import io
import torch
import torch.nn as nn
import timm
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
AZ_BINS  = FFT_SIZE // 2   # 128
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# v5 paths
CNN_V5_PATH   = "best_model_v5.pth"
SKL_PKL_PATHS = ["linear_svm_v5.pkl", "rbf_svm_v5.pkl", "rf_model_v5.pkl"]
FEATURES_NPZ  = "features_v5.npz"

# Used only in ONNX fallback mode
DECISION_THRESHOLD      = 0.35
CALIBRATION_TEMPERATURE = 3.0

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ---------------------------------------------------------------------------
# Manipulation type human-readable labels & descriptions
# ---------------------------------------------------------------------------
MANIPULATION_LABELS = {
    "faceswap_autoencoder": "Face Swap (Autoencoder)",
    "stylegan2":            "StyleGAN2 Synthesis",
    "stylegan_ffhq":        "StyleGAN (FFHQ-trained)",
    "stylegan_celeba":      "StyleGAN (CelebA-trained)",
    "stylegan_variant":     "StyleGAN Variant",
    "gan_mixed":            "GAN Synthesis (Mixed)",
    "faceswap_dfd":         "Face Swap (DeepFaceLab-style)",
    "stargan":              "StarGAN Attribute Editing",
    "pggan_v1":             "Progressive GAN v1",
    "pggan_v2":             "Progressive GAN v2",
    "faceapp":              "FaceApp AI Manipulation",
}

MANIPULATION_DESCRIPTIONS = {
    "faceswap_autoencoder": "This replaces a person's face using an AI autoencoder. It often leaves behind subtle blending lines and mismatched lighting.",
    "stylegan2": "This generates entirely fake, photorealistic faces. It usually leaves hidden grid-like artifacts in the image pixels.",
    "stylegan_ffhq": "This creates highly realistic fake faces, leaving behind subtle AI frequency traces.",
    "stylegan_celeba": "This generates fake faces based on celebrity photos, showing specific AI noise patterns.",
    "stylegan_variant": "This is a variant of the StyleGAN family, which typically leaves repeated upsampling artifacts hidden in the image.",
    "gan_mixed": "This looks like a mix of different AI generators, showing unusual and diverse artificial noise patterns.",
    "faceswap_dfd": "This is a DeepFaceLab-style face swap. It usually leaves tiny seam artifacts around the edges of the face.",
    "stargan": "This edits specific facial features like hair, gender, or age, often leaving behind a slight checkerboard effect.",
    "pggan_v1": "This progressively generates fake faces from low to high resolution, which often leaves blurry artifacts in the background.",
    "pggan_v2": "An improved AI generator, but it still leaves faint artificial traces around hair and backgrounds.",
    "faceapp": "This applies heavy AI filters to a real photo. It drastically smooths out the skin and flattens natural textures.",
}

# CNN : v5 architecture 
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
        d = self.backbone.num_features  # 1280
        self.head = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(d, 256), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.head(self.backbone(x))

    def get_features(self, x):
        """Return 1280-d backbone embedding (no head). Used for SVM + k-NN."""
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Load CNN weights : prefer v5, fall back to v4
# ---------------------------------------------------------------------------
_cnn_path = CNN_V5_PATH 
if not os.path.exists(_cnn_path):
    raise FileNotFoundError(
        f"CNN weights not found. Expected '{CNN_V5_PATH}'."
    )

feature_model = DeepfakeDetector().to(DEVICE)
state = torch.load(_cnn_path, map_location=DEVICE, weights_only=False)
feature_model.load_state_dict(state, strict=False)
feature_model.eval()
print(f"✅ CNN loaded from {_cnn_path}")


USE_SKL        = False
skl_clf        = None
svm_session    = None
SVM_INPUT_NAME = None
feature_scaler = None

for pkl_path in SKL_PKL_PATHS:
    if os.path.exists(pkl_path):
        skl_clf = joblib.load(pkl_path)
        USE_SKL = True
        print(f"✅ Sklearn classifier loaded from {pkl_path}  (StandardScaler embedded in pipeline)")
        break

if not USE_SKL:
    import onnxruntime as rt

    ONNX_PATH = "rbf_svm_v5.onnx"   # ← change to your actual filename

    if os.path.exists(ONNX_PATH):
        svm_session    = rt.InferenceSession(ONNX_PATH)
        SVM_INPUT_NAME = svm_session.get_inputs()[0].name
        print(f"✅ ONNX classifier loaded from {ONNX_PATH}")
    else:
        print(f"❌ ONNX file not found at '{ONNX_PATH}'")

_knn              = None
_fake_types_train = None

if os.path.exists(FEATURES_NPZ):
    try:
        from sklearn.neighbors import NearestNeighbors
        _d           = np.load(FEATURES_NPZ, allow_pickle=True)
        X_tr         = _d["X_train"]
        y_tr         = _d["y_train"].astype(int)
        dt_tr        = _d["dtype_train"]
        _d.close()
        _fake_mask        = y_tr == 1
        _knn              = NearestNeighbors(n_neighbors=min(15, int(_fake_mask.sum())))
        _knn.fit(X_tr[_fake_mask])
        _fake_types_train = np.asarray(dt_tr)[_fake_mask]
        print(f"✅ k-NN type attributor ready")
    except Exception as exc:
        print(f"⚠️  k-NN setup failed ({exc})")
else:
    print(f"ℹ️  {FEATURES_NPZ} not found")


def compute_azimuthal_average(spectrum_2d: np.ndarray) -> np.ndarray:
    h, w   = spectrum_2d.shape
    cy, cx = h // 2, w // 2
    Y, X   = np.ogrid[:h, :w]
    r      = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    max_r  = min(cy, cx)
    return ndimage.mean(spectrum_2d, labels=r, index=np.arange(0, max_r))


def extract_combined_features(img_bgr: np.ndarray):
    """
    Extract the 1536-d feature vector that was used to train the SVM.

    Returns
    -------
    feat     : np.ndarray  shape (1536,)
    spectral : dict        'az' and 'nz' lists for explanation / debug
    """
    # ── 4-channel input tensor ─────────────────────────────────────────────
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r224    = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    norm    = (r224.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    t3      = torch.from_numpy(norm.transpose(2, 0, 1))

    gray224 = cv2.cvtColor(r224, cv2.COLOR_RGB2GRAY).astype(np.float32)
    fs0     = np.fft.fftshift(np.fft.fft2(gray224))
    ps      = np.log1p(np.abs(fs0) ** 2)
    ps      = (ps - ps.min()) / (ps.max() - ps.min() + 1e-8)
    fft_ch  = torch.from_numpy(ps.astype(np.float32)).unsqueeze(0)

    tensor_4ch = torch.cat([t3, fft_ch], dim=0).unsqueeze(0).to(DEVICE)

    # ── CNN backbone → 1280-d (matches notebook Cell 11 / Cell 17) ─────────
    with torch.no_grad():
        cnn_feat = feature_model.get_features(tensor_4ch).float().cpu().numpy()  # (1, 1280)

    # ── FFT azimuthal profile at FFT_SIZE ──────────────────────────────────
    gray = cv2.resize(
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (FFT_SIZE, FFT_SIZE)
    ).astype(np.float32)
    fs    = np.fft.fftshift(np.fft.fft2(gray))
    power = np.log1p(np.abs(fs) ** 2)
    az    = compute_azimuthal_average(power).astype(np.float32)   # 128-d

    # ── Noise-residual FFT azimuthal profile ───────────────────────────────
    blurred     = cv2.GaussianBlur(gray, (5, 5), 1.0)
    noise       = gray - blurred
    nfs         = np.fft.fftshift(np.fft.fft2(noise))
    noise_power = np.log1p(np.abs(nfs) ** 2)
    nz          = compute_azimuthal_average(noise_power).astype(np.float32)  # 128-d

    combined = np.concatenate([cnn_feat[0], az, nz]).astype(np.float32)
    assert combined.shape == (1536,), f"Bad feature dim: {combined.shape}"

    spectral = {"az": az.tolist(), "nz": nz.tolist()}
    return combined, spectral

def attribute_type(feat_1d: np.ndarray) -> list:
    # Vote among 15 nearest fake-training-set neighbours.
    if _knn is None:
        return []
    _, idx   = _knn.kneighbors(feat_1d.reshape(1, -1))
    neigh    = _fake_types_train[idx[0]]
    vals, counts = np.unique(neigh, return_counts=True)
    order    = np.argsort(-counts)
    return [(str(vals[o]), float(counts[o] / len(neigh) * 100)) for o in order]

def analyze_spectral_profile(az: list, nz: list) -> tuple[list, str]:
    # Inspect the azimuthal power and noise-residual profiles.
    az_arr = np.array(az, dtype=np.float32)
    nz_arr = np.array(nz, dtype=np.float32)
    n = len(az_arr)
    if n < 8:
        return [], "Image is too small for a clear frequency analysis."

    low_end  = max(1, n // 8)
    mid_s    = n // 4
    mid_e    = n // 2
    hi_s     = 3 * n // 4

    low_mean = float(az_arr[:low_end].mean())
    hi_mean  = float(az_arr[hi_s:].mean())
    nz_low   = float(nz_arr[:low_end].mean())
    nz_hi    = float(nz_arr[hi_s:].mean())

    flags = []

    # ── Flag 1: elevated high-frequency power (GAN upsampling) ────────────
    decay_ratio = hi_mean / (low_mean + 1e-6)
    if decay_ratio > 0.55:
        flags.append("high_freq_elevation")
    
    # ── Flag 2: noise-residual elevation in high bands (blend / re-encode) ─
    noise_ratio = nz_hi / (nz_low + 1e-6)
    if noise_ratio > 0.75:
        flags.append("noise_floor_elevated")

    # ── Flag 3: non-monotonic spectral profile (GAN periodic spikes) ───────
    diffs        = np.diff(az_arr[mid_s:])
    sign_changes = int(((diffs[:-1] * diffs[1:]) < 0).sum())
    if sign_changes > n // 6:
        flags.append("spectral_oscillation")
    
    # ── Flag 4: flat mid-band (over-smoothing : FaceApp / inpainting) ──────
    mid_std = float(az_arr[mid_s:mid_e].std())
    if mid_std < 0.04 * (low_mean + 1e-6):
        flags.append("mid_band_flat")
    return flags

def build_explanation(is_fake: bool, p_fake: float, manip_types: list, spectral_flags: list) -> str:
    pct_fake = round(p_fake * 100, 1)
    pct_real = round((1.0 - p_fake) * 100, 1)

    if not is_fake:
        if not spectral_flags:
            return f"This image looks authentic. We are {pct_real}% confident it is real. The underlying frequency patterns are completely natural with no obvious AI traces."
        else:
            return f"This image looks authentic. We are {pct_real}% confident it is real. We picked up a few minor frequency quirks, but our core model confirms the overall image structure is natural and not manipulated."

    parts = [f"We detected that this image is AI-generated or manipulated (estimated {pct_fake}% probability)."]

    if manip_types:
        top_type, top_pct = manip_types[0]
        label = MANIPULATION_LABELS.get(top_type, top_type)
        desc  = MANIPULATION_DESCRIPTIONS.get(top_type, "")
        parts.append(f"The patterns closely match {label}. {desc}")
    elif spectral_flags:
        if "high_freq_elevation" in spectral_flags or "spectral_oscillation" in spectral_flags:
            parts.append("The image contains distinct hidden pixel patterns that are a dead giveaway for GAN-style AI generators.")
        elif "noise_floor_elevated" in spectral_flags:
            parts.append("The background noise levels are uneven, which is usually a strong sign of a face-swap.")
        elif "mid_band_flat" in spectral_flags:
            parts.append("The textures are unnaturally smooth, which usually points to a pyt AI filter like FaceApp.")
        else:
            parts.append("While the frequency traces are subtle, our deep learning model strongly recognized features from known deepfake datasets.")

    return " ".join(parts)

def get_confidence_level(p_fake: float) -> str:
    threshold = 0.5 if USE_SKL else DECISION_THRESHOLD
    distance  = abs(p_fake - threshold)
    if distance < 0.10:
        return "uncertain"
    elif distance < 0.25:
        return "low"
    elif distance < 0.40:
        return "medium"
    else:
        return "high"


# ---------------------------------------------------------------------------
# /analyze endpoint
# ---------------------------------------------------------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image provided."}), 400

    try:
        raw     = request.files["image"].read()
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        img_rgb = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # ── Feature extraction ─────────────────────────────────────────────
        features, spectral = extract_combined_features(img_bgr)
        feat_2d = features.reshape(1, -1)

        # ── Classification ─────────────────────────────────────────────────
        if USE_SKL:
            # sklearn Pipeline handles scaling internally
            p_fake = float(skl_clf.predict_proba(feat_2d)[0, 1])
            p_real = 1.0 - p_fake
            is_fake = p_fake >= 0.5

        else:
            # ONNX 
            scaled = (
                feature_scaler.transform(feat_2d).astype(np.float32)
                if feature_scaler is not None
                else feat_2d.astype(np.float32)
            )
            label_arr, prob_list = svm_session.run(None, {SVM_INPUT_NAME: scaled})
            p_fake_raw = float(prob_list[0][1])

            # Temperature scaling (smooths isotonic step-function probabilities)
            eps      = 1e-6
            p_clip   = float(np.clip(p_fake_raw, eps, 1.0 - eps))
            logit_p  = np.log(p_clip / (1.0 - p_clip))
            p_fake   = float(1.0 / (1.0 + np.exp(-logit_p / CALIBRATION_TEMPERATURE)))
            p_real   = 1.0 - p_fake
            is_fake  = bool(int(label_arr[0]) == 1)

        confidence = get_confidence_level(p_fake)
        manip_types = attribute_type(features) if is_fake else []
        spectral_flags = analyze_spectral_profile(spectral["az"], spectral["nz"])
        spectral_summary = ""
        explanation = build_explanation(
            is_fake, p_fake, manip_types, spectral_flags
        )

        manip_label  = None
        manip_scores = []

        if is_fake:
            if manip_types:
                top_type, _ = manip_types[0]
                manip_label  = MANIPULATION_LABELS.get(top_type, top_type)
                manip_scores = [
                    {
                        "type":       t,
                        "label":      MANIPULATION_LABELS.get(t, t),
                        "confidence": round(p, 1),
                    }
                    for t, p in manip_types[:5]
                ]
            elif spectral_flags:
                if "high_freq_elevation" in spectral_flags or "spectral_oscillation" in spectral_flags:
                    manip_label = "GAN Synthesis"
                elif "noise_floor_elevated" in spectral_flags:
                    manip_label = "Face Swap / Re-encoding"
                elif "mid_band_flat" in spectral_flags:
                    manip_label = "AI Filter / Smoothing"
                else:
                    manip_label = "AI Manipulation Type Unknown"

        print(
            f"DEBUG: p_fake={p_fake:.4f}  label={'FAKE' if is_fake else 'REAL'}  "
            f"confidence={confidence}  manip={manip_label}"
        )

        return jsonify({
            "probability":         round(p_fake * 100, 2),
            "label":               "AI Generated / Fake" if is_fake else "Authentic Media",
            "is_fake":             is_fake,
            "confidence":          confidence,
            "p_real":              round(p_real * 100, 2),
            "p_fake":              round(p_fake * 100, 2),
            "explanation":         explanation,
            "manipulation_type":   manip_label,
            "manipulation_scores": manip_scores,
            "spectral_flags":      spectral_flags,
            "spectral_summary":    spectral_summary,
            "threshold_used":      0.5 if USE_SKL else DECISION_THRESHOLD,
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=False)