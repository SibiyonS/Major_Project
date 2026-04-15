import base64
import json
import os
import shutil
import textwrap
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import librosa
import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template_string, request, send_file, send_from_directory

from src.config import AudioConfig, MODELS_DIR, XAI_DIR
from src.data import load_and_preprocess_audio
from src.model import CnnBiLstmDetector

DEFAULT_MODEL_PATH = "artifacts/models/best_cnn_bilstm.pt"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_KEYS = [
    "AIzaSyDYm4QXE0azSxb_5aikb7K5YScnizNQ9XI",
    "AIzaSyCPNNSVmnxrNDZ95fCnOmpIetyvyO6DB6w",
    "AIzaSyC8F7X1X1X1X1X1X1X1X1X1X1X1X1X1X1",
]
ALLOWED_AUDIO_EXTENSIONS = {"wav", "mp3", "flac", "m4a", "ogg"}


def _load_runtime_threshold(default: float = 0.563) -> float:
    summary_path = MODELS_DIR / "summary.json"
    if not summary_path.exists():
        return default
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        value = float(payload.get("best_threshold", default))
        return min(0.95, max(0.05, value))
    except Exception:
        return default


def gradcam_for_sample(model: CnnBiLstmDetector, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, float]:
    activations: Dict[str, torch.Tensor] = {}
    gradients: Dict[str, torch.Tensor] = {}

    def forward_hook(_module, _input, output):
        activations["value"] = output

    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    target_layer = model.features[-1]
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    was_training = model.training
    model.train()

    with torch.enable_grad():
        with torch.backends.cudnn.flags(enabled=False):
            model.zero_grad(set_to_none=True)
            logits, attention = model(x, return_attention=True)
            prob = torch.sigmoid(logits)[0]
            target = logits[0] if prob >= 0.5 else -logits[0]
            target.backward()

    acts = activations.get("value")
    grads = gradients.get("value")

    if acts is None or grads is None:
        h1.remove()
        h2.remove()
        model.train(was_training)
        blank = np.zeros((x.shape[-2], x.shape[-1]), dtype=np.float32)
        return blank, attention.squeeze().detach().cpu().numpy(), float(prob.detach().cpu().item())

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam_np = cam.squeeze().detach().cpu().numpy()
    cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)

    attn_np = attention.squeeze().detach().cpu().numpy()
    h1.remove()
    h2.remove()
    model.train(was_training)
    return cam_np, attn_np, float(prob.detach().cpu().item())


'''def generate_reason_with_gemini(pred: str, score: float, cam_path: str, attn_path: str, model_name: str, api_key: str) -> str:
    try:
        import google.generativeai as genai  # type: ignore
        from PIL import Image

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        confidence = "High" if score > 0.8 else ("Moderate" if score > 0.6 else "Low")

        response = model.generate_content(
            contents=[
                "Explain the detail in about 200 words by analyzing the image given which is a grad-cam and attention Blistm image"
                "convert you point into the statement and produce to the end user in simple,neat and in the understandable formate"
                "make simple and easy to understand for the non technical user and teach like teaching to the beginner",
                f"Prediction: {pred}\nFake Probability: {score:.3f}\nConfidence: {confidence}",
                Image.open(cam_path),
                Image.open(attn_path),
            ]
        )
        text = getattr(response, "text", "")
        return text.strip() if text else f"Prediction: {pred}\nConfidence: {confidence}\nNo LLM text returned."
    except Exception as e:
        return f"Gemini API Error: {e}\nFallback prediction: {pred} ({score:.3f})"'''

def generate_reason_with_gemini(pred, score, cam_path, attn_path, model_name):
    import google.generativeai as genai
    from PIL import Image

    confidence = "Low" if score > 0.8 else ("Moderate" if score > 0.6 else "High")

    last_error = None

    for api_key in GEMINI_API_KEYS:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)

            response = model.generate_content(
              contents=[
                  "Explain the detail in about 200 words by analyzing the image given which is a grad-cam and attention Blistm image"
                  "convert you point into the statement and produce to the end user in simple,neat and in the understandable formate"
                  "make simple and easy to understand for the non technical user and teach like teaching to the beginner",
                  f"Prediction: {pred}\nFake Probability: {score:.3f}\nConfidence: {confidence}",
                  Image.open(cam_path),
                  Image.open(attn_path),  
              ]
            )

            text = getattr(response, "text", "")
            if text:
                print(f"[Gemini] Success with key: {api_key[:6]}***")
                return text.strip()

        except Exception as e:
            print(f"[Gemini] Key failed: {api_key[:6]}*** | Error: {e}")
            last_error = e
            continue

    return f"All API keys failed. Last error: {last_error}"


def save_gradcam_overlay(mel: np.ndarray, cam: np.ndarray, attn: np.ndarray, title: str, cfg: AudioConfig, audio_path: str, out_path: Path) -> Path:
    audio, sr = librosa.load(audio_path, sr=cfg.sample_rate)
    duration = len(audio) / max(1, sr)

    if hasattr(attn, "detach"):
        attn = attn.detach().cpu().numpy()
    attn = np.asarray(attn, dtype=np.float32)
    if attn.size == 0:
        attn = np.zeros((mel.shape[1],), dtype=np.float32)

    attn_resized = np.interp(
        np.linspace(0, max(0, len(attn) - 1), mel.shape[1]),
        np.arange(len(attn)) if len(attn) > 0 else np.array([0]),
        attn if len(attn) > 0 else np.array([0.0]),
    )

    plt.figure(figsize=(10, 4))
    plt.imshow(mel, aspect="auto", origin="lower", extent=[0, duration, 0, mel.shape[0]])
    plt.imshow(cam, cmap="jet", alpha=0.35, origin="lower", aspect="auto", extent=[0, duration, 0, mel.shape[0]])
    plt.plot(np.linspace(0, duration, mel.shape[1]), attn_resized * mel.shape[0], color="white", linewidth=2, label="Attention")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Mel bins")
    plt.colorbar(label="Grad-CAM intensity")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def save_attention_plot(attn: np.ndarray, out_path: Path) -> Path:
    plt.figure(figsize=(10, 3))
    plt.plot(attn)
    plt.title("BiLSTM Attention Over Time Frames")
    plt.xlabel("Time frame")
    plt.ylabel("Attention weight")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def load_model(model_path: str, device: str) -> CnnBiLstmDetector:
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    mcfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    model = CnnBiLstmDetector(
        hidden_size=mcfg.get("hidden_size", 64),
        dropout=mcfg.get("dropout", 0.3),
    ).to(device)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def classify_audio_from_chunks(probs: list[float], attn: np.ndarray | None = None, threshold: float = 0.6):
    probs_np = np.asarray(probs, dtype=np.float32)
    probs_np = np.clip(np.nan_to_num(probs_np, nan=0.5, posinf=1.0, neginf=0.0), 1e-6, 1 - 1e-6)

    if attn is not None:
        attn_np = np.asarray(attn, dtype=np.float32)
        attn_np = np.clip(attn_np, 0.0, None)
        attn_np = attn_np / (attn_np.sum() + 1e-8)

        if len(attn_np) != len(probs_np):
            attn_np = np.ones_like(probs_np) / max(1, len(probs_np))

        score_mean = float(np.sum(probs_np * attn_np))
    else:
        score_mean = float(np.mean(probs_np))

    score_max = float(np.max(probs_np))

    # 🔥 hybrid score
    score = 0.6 * score_mean + 0.4 * score_max

    # 🔥 smarter decision
    if score >= threshold or score_max >= 0.85:
        label = "FAKE"
    else:
        label = "REAL"

    debug = {
        "threshold": threshold,
        "mean_prob": float(np.mean(probs_np)),
        "max_prob": score_max,
        "min_prob": float(np.min(probs_np)),
        "score": score,
        "chunks": int(len(probs_np)),
        "probs": probs_np.tolist(),
    }

    return label, score, debug



'''def _publish_latest_images(grad_path: Path, attn_path: Path):
    shutil.copy2(grad_path, XAI_DIR / "gradcam_overlay.png")
    shutil.copy2(attn_path, XAI_DIR / "attention_weights.png")'''


def run_single_audio_pipeline(audio_path: str, model_path: str, gemini_model: str) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = AudioConfig()
    threshold = _load_runtime_threshold(0.563)

    XAI_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model(model_path, device)
    features = load_and_preprocess_audio(audio_path, cfg)
    print(f"[pipeline] uploaded_file={audio_path}")
    print(f"[pipeline] chunks_extracted={len(features)}")

    probs: list[float] = []
    inputs: list[torch.Tensor] = []
    chunk_attn: list[float] = []

    with torch.no_grad():
        for idx, mel in enumerate(features):
            #print(f"[pipeline] chunk[{idx}] mel_shape={tuple(mel.shape)}")
            x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            inputs.append(x)
            logits, attn = model(x, return_attention=True)
            probs.append(float(torch.sigmoid(logits).item()))
            chunk_attn.append(float(attn.mean().item()))

    attn_weights = np.asarray(chunk_attn, dtype=np.float32)
    attn_weights = attn_weights / (attn_weights.sum() + 1e-8)
    print(f"[pipeline] chunk_probabilities={np.asarray(probs, dtype=np.float32).tolist()}")
    print(f"[pipeline] chunk_attention_weights={attn_weights.tolist()}")

    pred, score, debug = classify_audio_from_chunks(probs, attn_weights, threshold=threshold)
    print(f"[pipeline] aggregated_prediction={pred} score={score:.4f} threshold={threshold:.2f}")

    best_idx = int(np.argmax(probs))
    print(f"[pipeline] highest_probability_chunk_index={best_idx}")
    x_best = inputs[best_idx]
    mel_best = features[best_idx]

    cam, attn_map, _ = gradcam_for_sample(model, x_best)

    #run_prefix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    grad_path = XAI_DIR / f"gradcam_overlay.png"
    attn_path = XAI_DIR / f"attention_weights.png"

    overlay_path = save_gradcam_overlay(
        mel=mel_best,
        cam=cam,
        attn=attn_map,
        title=f"Grad-CAM | Prediction={pred} | fake_prob={score:.3f} | th={threshold:.2f}",
        cfg=cfg,
        audio_path=audio_path,
        out_path=grad_path,
    )
    attention_path = save_attention_plot(attn_weights, out_path=attn_path)
    #_publish_latest_images(overlay_path, attention_path)

    #api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
    try:
        explanation = generate_reason_with_gemini(
            pred,
            score,
            str(overlay_path),
            str(attention_path),
            gemini_model
        )
    except Exception as e:
        explanation = (
            f"Gemini failed: {e}\n"
            f"Prediction: {pred} (score={score:.3f}, threshold={threshold:.2f}).\n"
            "Grad-CAM and attention outputs are generated and saved."
        )

    out_txt = XAI_DIR / "gemini_explanation.txt"
    out_txt.write_text(explanation, encoding="utf-8")

    return {
        "prediction": pred,
        "fake_probability": score,
        "threshold": threshold,
        "debug": debug,
        "explanation": explanation,
        "gradcam_path": str(overlay_path),
        "attention_path": str(attention_path),
        "text_path": str(out_txt),
    }


def _is_allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS


def _slugify_name(filename: str) -> str:
    stem = Path(filename).stem
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    return sanitized[:60] or "audio"


app = Flask(__name__)

BASE_STYLE = """
<style>
  :root{--bg:#090217;--bg-soft:#1a0f35;--card:rgba(24,18,48,.62);--line:#5c4ea0;--text:#f2ecff;--muted:#c1b6e4;--accent:#6ee7ff;--accent2:#ae82ff;--danger:#ff6f8f;--good:#33e6a4;--shadow:0 20px 45px rgba(0,0,0,.45)}
  body.light{--bg:#f2f8ff;--bg-soft:#ddeeff;--title-color:#7C80EB;--card:#ffffffd9;--line:#bbd4f2;--text:#0d2238;--muted:#45627d;--accent:#1558ff;--accent2:#7c48ff;--danger:#de2b48;--good:#008d5f;--shadow:0 18px 32px rgba(13,34,56,.18)}
  *{box-sizing:border-box} html{scroll-behavior:smooth}
  body{margin:0;color:var(--text);font-family:Inter,Segoe UI,Arial,sans-serif;overflow-x:hidden;background:
      radial-gradient(1000px 500px at -10% -20%, color-mix(in srgb,var(--accent2) 25%, transparent), transparent 60%),
      radial-gradient(900px 500px at 110% -20%, color-mix(in srgb,var(--accent) 22%, transparent), transparent 62%),
      linear-gradient(180deg,var(--bg),color-mix(in srgb,var(--bg) 76%,#000) 70%);min-height:100vh}
  .texture:before,.texture:after{content:"";position:fixed;inset:0;pointer-events:none}
  .texture:before{background:repeating-linear-gradient(transparent 0 2px,rgba(255,255,255,.03) 3px 4px);opacity:.25;z-index:0}
  .texture:after{background-image:linear-gradient(rgba(110,231,255,.07) 1px,transparent 1px),linear-gradient(90deg,rgba(174,130,255,.07) 1px,transparent 1px);background-size:42px 42px;opacity:.22;z-index:0}
  .particles span{position:fixed;width:5px;height:5px;border-radius:50%;opacity:.8;filter:blur(.2px);z-index:0;animation:particleMove linear infinite}
  @keyframes particleMove{0%{transform:translateY(100vh) translateX(0)}100%{transform:translateY(-10vh) translateX(24px)}}
  .app{position:relative;z-index:2;width:min(1220px,94vw);margin:0 auto;padding:0 0 34px}
  .topbar{position:sticky;top:0;z-index:9;display:flex;align-items:center;justify-content:space-between;background:color-mix(in srgb,var(--card) 88%, #040812);border-bottom:2px solid var(--accent);border-left:none;border-right:none;border-top:none;border-radius:0;padding:14px 24px;box-shadow:var(--shadow);backdrop-filter:blur(10px);width:100vw;margin-left:calc(50% - 50vw)}
  .brand{display:flex;align-items:center;gap:10px;font-weight:800;letter-spacing:.06em}.brand-dot{width:10px;height:10px;border-radius:50%;background:var(--accent);box-shadow:0 0 16px var(--accent)}
  .nav{display:flex;align-items:center;gap:16px}.nav a{color:var(--text);text-decoration:none;font-size:18px;opacity:.9}.nav a:hover{color:var(--accent)}
  .controls{display:flex;align-items:center;gap:10px}.icon-btn,.burger{border:1px solid var(--line);background:transparent;color:var(--text);cursor:pointer;border-radius:10px;padding:8px 10px;transition:.25s}
  .icon-btn:hover,.burger:hover{border-color:var(--accent);box-shadow:0 0 16px rgba(0,231,255,.25);transform:translateY(-1px)}.burger{display:none}
  .icon-btn:active,.burger:active{transform:translateY(1px)}
  .mobile-menu{display:none;flex-direction:column;gap:8px;margin-top:10px;padding:10px;border:1px solid var(--line);border-radius:12px;background:var(--card)}.mobile-menu a{color:var(--text);text-decoration:none}
  .panel{background:var(--card);border:2px solid color-mix(in srgb,var(--line) 65%, #8f7bff);border-radius:20px;box-shadow:var(--shadow);backdrop-filter:blur(10px);animation:reveal .65s ease both;transition:.3s}
  .panel:hover{transform:translateY(-4px) scale(1.01);border-color:var(--accent);background:color-mix(in srgb,var(--card) 75%, var(--accent2));box-shadow:0 0 0 2px color-mix(in srgb,var(--accent) 60%, transparent),0 26px 45px rgba(0,0,0,.35)}
  .module{border:2px solid var(--line);padding:14px;border-radius:14px;transition:.25s;background:rgba(8,12,26,.22)}
  .module:hover{border-color:var(--accent);background:rgba(110,231,255,.12)}
  .cta-wrap{text-align:center;margin-top:20px}
  .cta-btn{display:inline-block;text-decoration:none;border:2px solid var(--accent);padding:12px 20px;border-radius:14px;color:var(--text);font-weight:800;transition:.25s;background:linear-gradient(135deg,color-mix(in srgb,var(--accent) 35%, transparent),color-mix(in srgb,var(--accent2) 30%, transparent));box-shadow:0 8px 18px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.15)}
  .cta-btn:hover{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#001018;box-shadow:0 0 24px color-mix(in srgb,var(--accent) 60%, transparent),0 10px 20px rgba(0,0,0,.25);transform:translateY(-1px)}
  .cta-btn:active{transform:translateY(1px)}
  .site-footer{margin-top:22px;background:#22034f;border-top:2px solid #7f57ff;padding:24px;display:flex;justify-content:space-between;align-items:flex-start;gap:20px;flex-wrap:wrap;width:100vw;margin-left:calc(50% - 50vw)}
  .site-footer .links{display:flex;gap:18px;flex-wrap:wrap;font-weight:600}
  .site-footer a{color:#f6edff;text-decoration:none}
  @keyframes reveal{from{opacity:0;transform:translateY(16px) scale(.99)}to{opacity:1;transform:translateY(0) scale(1)}}
  .foot{margin-top:14px;color:var(--muted);font-size:12px;text-align:right}
  @media (max-width: 990px){.nav{display:none}.burger{display:block}.mobile-menu.show{display:flex}}
</style>
"""


def _shared_script() -> str:
    return """
<script>
const body = document.body;
const themeBtn = document.getElementById('themeBtn');
const burgerBtn = document.getElementById('burgerBtn');
const mobileMenu = document.getElementById('mobileMenu');
function setTheme(mode){
  if(mode === 'light'){ body.classList.add('light'); themeBtn.textContent='☀️'; }
  else { body.classList.remove('light'); themeBtn.textContent='🌙'; }
  localStorage.setItem('theme_mode', mode);
}
setTheme(localStorage.getItem('theme_mode') || 'dark');
themeBtn.onclick = () => setTheme(body.classList.contains('light') ? 'dark' : 'light');
burgerBtn.onclick = () => mobileMenu.classList.toggle('show');
const particleHost = document.getElementById('particles');
if (particleHost){
  for(let i=0;i<120;i++){
    const p=document.createElement('span');
    p.style.left=(Math.random()*100)+'vw';
    p.style.top=(Math.random()*100)+'vh';
    p.style.animationDuration=(5+Math.random()*9)+'s';
    p.style.animationDelay=(Math.random()*6)+'s';
    p.style.background=(Math.random()>0.5?'#6ee7ff':'#ae82ff');
    particleHost.appendChild(p);
  }
}
</script>
"""


def _shell(page_title: str, content: str) -> str:
    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{page_title}</title>
  {BASE_STYLE}
</head>
<body class="texture">
  <div class="particles" id="particles"></div>
  <div class="app">
    <header class="topbar">
      <div class="brand"><span class="brand-dot"></span>DEEPFAKE VOICE DETECTION</div>
      <div class="controls">
        <nav class="nav">
          <a href="/">Home</a>
          <a href="/analyze">Analyzer</a>
          <a href="/overcome">Overcome</a>
        </nav>
        <button class="icon-btn" id="themeBtn" title="Toggle theme">🌙</button>
        <button class="burger" id="burgerBtn">☰</button>
      </div>
    </header>
    <div class="mobile-menu" id="mobileMenu">
      <a href="/">Home</a>
      <a href="/analyze">Analyzer</a>
      <a href="/overcome">Overcome</a>
    </div>
    {content}
    <footer class="site-footer">
      <div style="min-width:260px">
        <div style="font-size:22px;font-weight:800;margin-bottom:8px">🧠 Deepfake Voice Detection</div>
        <p style="color:#d7c3ff;margin:0">Forensic AI platform for trustworthy voice authenticity analysis.</p>
      </div>
      <div class="links">
        <a href="/">Home</a><a href="/analyze">Analyzer</a><a href="/overcome">Overcome</a><a href="#">Privacy Policy</a><a href="#">Terms of Use</a><a href="#">Contact</a>
      </div>
      <div style="max-width:560px">
        <p style="color:#f2dfff;margin:0 0 6px">Copyright © 2026 Deepfake Voice Detection Platform. All Rights Reserved.</p>
        <p style="color:#d7c3ff;margin:0">This website is designed for investigative assistance and educational use. Outputs should not be treated as standalone legal verdicts; combine with human expert review, metadata checks, and chain-of-custody documentation.</p>
      </div>
    </footer>
  </div>
  {_shared_script()}
</body>
</html>
"""


@app.get("/")
def home():
    content = """
<section class="panel" style="padding:26px;margin-top:18px">
  <h1 style="font-size:44px;margin:0 0 10px">Deepfake Voice Detection Platform</h1>
  <p style="color:var(--muted);font-size:17px;max-width:960px">A multi-page digital-forensics system that converts uploaded audio into prediction, visual evidence, and analyst-friendly explanation. This website is designed to explain not only the final label but the complete backend working process.</p>
  <div class="cta-wrap">
    <a href="/analyze" class="cta-btn">▶ Start Analyze</a>
    <a href="/overcome" class="cta-btn" style="margin-left:8px">🛡 Why Deepfake?</a>
  </div>
</section>
<section class="panel" style="padding:22px;margin-top:14px">
  <h2>Background working and operation flow</h2>
  <p style="color:var(--muted)">Operation starts when an audio file is uploaded. The backend normalizes waveform and extracts mel-spectrogram features. The CNN block captures local spectral artifacts, the BiLSTM block captures temporal behavior, and attention pooling highlights the most influential time-frames before binary classification (REAL/FAKE).</p>
  <p style="color:var(--muted)">After prediction, Grad-CAM generates frequency-time heat intensity and attention curve reveals sequential importance. Finally, a human-readable AI statement is generated from these forensic artifacts.</p>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px">
    <div class="module"><img src="https://images.unsplash.com/photo-1555255707-c07966088b7b?auto=format&fit=crop&w=900&q=80" alt="detection module" style="width:100%;height:140px;object-fit:cover;border-radius:10px;margin-bottom:8px"/><strong>🔍 1. Detection Module</strong><br/><small>Classifies REAL/FAKE using trained CNN-BiLSTM model architecture and probability thresholding for initial triage.</small></div>
    <div class="module"><img src="https://images.unsplash.com/photo-1555949963-aa79dcee981c?auto=format&fit=crop&w=900&q=80" alt="explainable ai module" style="width:100%;height:140px;object-fit:cover;border-radius:10px;margin-bottom:8px"/><strong>🧠 2. Explainable AI Module</strong><br/><small>Generates Grad-CAM and attention traces to explain where and when the model found synthetic voice cues.</small></div>
    <div class="module"><img src="https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?auto=format&fit=crop&w=900&q=80" alt="document evidence module" style="width:100%;height:140px;object-fit:cover;border-radius:10px;margin-bottom:8px"/><strong>📄 3. Document Evidence Module</strong><br/><small>Compiles decision, AI statement, and forensic visuals into exportable evidence document for audit/legal workflows.</small></div>
  </div>
</section>
<section class="panel" style="padding:22px;margin-top:14px;background-image:linear-gradient(rgba(9,2,23,.65),rgba(9,2,23,.65)), url('https://images.unsplash.com/photo-1516321165247-4aa89a48be28?auto=format&fit=crop&w=1400&q=80');background-size:cover;background-position:center">
  <h2 style="color: var(--title-color);">Model architecture diagram</h2>
  <svg viewBox="0 0 980 260" style="width:100%;background:rgba(0,0,0,.14);border:1px solid var(--line);border-radius:12px">
    <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#34d8ff"/><stop offset="100%" stop-color="#8f7bff"/></linearGradient></defs>
    <rect x="20" y="90" width="150" height="70" rx="10" fill="url(#g)" opacity=".9"/><text x="34" y="130" fill="#001018" font-size="16" font-weight="700">Audio Input</text>
    <rect x="210" y="90" width="190" height="70" rx="10" fill="url(#g)" opacity=".9"/><text x="220" y="130" fill="#001018" font-size="16" font-weight="700">Mel Spectral Feature</text>
    <rect x="420" y="50" width="170" height="60" rx="10" fill="url(#g)" opacity=".85"/><text x="450" y="85" fill="#001018" font-size="15" font-weight="700">CNN Features</text>
    <rect x="420" y="140" width="170" height="60" rx="10" fill="url(#g)" opacity=".85"/><text x="445" y="175" fill="#001018" font-size="15" font-weight="700">BiLSTM + Attn</text>
    <rect x="620" y="75" width="210" height="90" rx="10" fill="url(#g)" opacity=".9"/><text x="650" y="125" fill="#001018" font-size="17" font-weight="700">Explainable AI</text>
    <rect x="850" y="95" width="110" height="60" rx="10" fill="url(#g)" opacity=".95"/><text x="866" y="132" fill="#001018" font-size="16" font-weight="700">REAL/FAKE</text>
    <line x1="170" y1="125" x2="210" y2="125" stroke="#9fdcff" stroke-width="3"/>
    <line x1="400" y1="125" x2="420" y2="80" stroke="#9fdcff" stroke-width="3"/>
    <line x1="400" y1="125" x2="420" y2="170" stroke="#9fdcff" stroke-width="3"/>
    <line x1="590" y1="80" x2="620" y2="105" stroke="#9fdcff" stroke-width="3"/>
    <line x1="590" y1="170" x2="620" y2="135" stroke="#9fdcff" stroke-width="3"/>
    <line x1="830" y1="120" x2="850" y2="125" stroke="#9fdcff" stroke-width="3"/>
  </svg>
  <p style="color:var(--muted);margin-top:10px">Definition: Grad-CAM localizes the spectral regions influencing decision; attention weights identify the most critical time windows. Together they convert black-box prediction into interpretable forensic evidence.</p>
</section>
<section class="panel" style="padding:22px;margin-top:14px">
  <h2>How the model was developed & tools used</h2>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
    <div class="module">⚙️ <strong>Development stack:</strong><br/><small>Python, PyTorch, Librosa, NumPy, Matplotlib, and Flask were used to build training, explainability, and deployment workflow.</small></div>
    <div class="module">🧪 <strong>Training strategy:</strong><br/><small>Audio preprocessing to mel features, CNN-based feature extraction, BiLSTM temporal modeling, and attention pooling for weighted sequence representation.</small></div>
    <div class="module">📊 <strong>Validation tools:</strong><br/><small>F1/AUC metrics, confusion matrix, ROC/PR curves, and visual checks on Grad-CAM/attention alignment for trust.</small></div>
    <div class="module">🧾 <strong>Reporting tools:</strong><br/><small>Automatic document generation embeds final prediction, explanation statement, and forensic visuals.</small></div>
  </div>
</section>
<section class="panel" style="padding:22px;margin-top:14px;display:grid;grid-template-columns:1fr 1fr;gap:14px">
  <div>
    <h2>Industrial workflow readiness</h2>
    <p style="color:var(--muted)">Recommended deployment chain: audio ingestion → model inference → analyst review → signed report archive. Pair with case ID, chain-of-custody logs, and multi-analyst validation before legal submission.</p>
    <p style="color:var(--muted)">This portal is built to support both technical users and non-technical reviewers by combining probabilities, visual evidence, and plain-language explanation.</p>
  </div>
  <img src="https://images.unsplash.com/photo-1518773553398-650c184e0bb3?auto=format&fit=crop&w=1200&q=80" alt="forensic lab setup" style="width:100%;height:280px;object-fit:cover;border-radius:14px;border:1px solid var(--line)"/>
</section>
"""
    return render_template_string(_shell("Deepfake Voice Detection - Home", content))


@app.get("/analyze")
def analyze_page():
    content = """
<style>
  .hero{display:grid;grid-template-columns:1.15fr .85fr;gap:18px;margin-top:18px}
  .dropzone{border:1px dashed color-mix(in srgb, var(--accent) 65%, #fff 10%);border-radius:14px;padding:18px;text-align:center;background:linear-gradient(180deg, rgba(0,231,255,.06), rgba(0,231,255,.015));transition:.25s}
  .dropzone.drag{transform:translateY(-2px); border-color:var(--accent); box-shadow:0 0 0 1px var(--accent) inset}
  .btn{border:1px solid color-mix(in srgb, var(--accent) 70%, #0ff);color:var(--text);background:linear-gradient(135deg, rgba(0,231,255,.2), rgba(35,255,208,.16));padding:11px 18px;border-radius:12px;cursor:pointer;font-weight:800;box-shadow:0 8px 14px rgba(0,0,0,.25), inset 0 1px 0 rgba(255,255,255,.18);transition:.2s}
  .btn:hover{transform:translateY(-1px);box-shadow:0 12px 20px rgba(0,0,0,.3),0 0 18px rgba(110,231,255,.2)}
  .btn:active{transform:translateY(1px)}
  .btn-row{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-top:10px}
  .cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:16px}
  .card{padding:14px}.k{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted)}.v{font-size:24px;font-weight:800;margin-top:4px}.good{color:var(--good)} .bad{color:var(--danger)}
  .result-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}
  .img-wrap{padding:14px}.img-wrap img{width:100%;min-height:240px;object-fit: contain;border:1px solid var(--line);border-radius:12px;background:rgba(0,0,0,.18)}
  .audio-shell{margin-top:10px;padding:12px;border-radius:14px;border:2px solid var(--line);background:linear-gradient(135deg,color-mix(in srgb,var(--accent) 25%, transparent),color-mix(in srgb,var(--accent2) 20%, transparent))}
  .audio-shell:hover{border-color:var(--accent);background:rgba(110,231,255,.16)}
  audio{width:100%;filter:drop-shadow(0 0 8px color-mix(in srgb,var(--accent) 45%, transparent))}
  .download-unique{font-size:15px;letter-spacing:.04em;padding:12px 20px;border-radius:14px;border:2px solid var(--accent2);background:linear-gradient(135deg,color-mix(in srgb,var(--accent2) 30%, transparent),color-mix(in srgb,var(--accent) 45%, transparent));box-shadow:0 0 24px color-mix(in srgb,var(--accent2) 30%, transparent)}
  .download-unique:hover{background:linear-gradient(135deg,var(--accent2),var(--accent));color:#020917}
  .loader-wrap{display:none;align-items:center;justify-content:center;gap:10px;margin-top:10px;color:var(--muted)}
  .loader{width:18px;height:18px;border:3px solid color-mix(in srgb,var(--accent) 45%, #fff);border-top-color:transparent;border-radius:50%;animation:spin .8s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  @media (max-width: 990px){.hero,.result-grid{grid-template-columns:1fr}.cards{grid-template-columns:1fr 1fr}}
  @media (max-width: 600px){.cards{grid-template-columns:1fr}}
</style>
<section class="hero">
  <div class="panel" style="padding:24px">
    <h1 style="font-size:40px;margin:0 0 8px">Forensic Analyzer Workspace</h1>
    <p style="color:var(--muted)">Upload one audio file and generate prediction, explainable visuals, AI statement, and downloadable forensic report.</p>
  </div>
  <div class="panel" style="padding:22px">
    <div id="dropzone" class="dropzone">
      <h3>Initiate Audio Sequence</h3>
      <p style="color:var(--muted)">Accepted: WAV, MP3, FLAC, M4A, OGG</p>
      <input id="audioInput" type="file" accept=".wav,.mp3,.flac,.m4a,.ogg,audio/*" hidden>
      <div class="btn-row">
        <button class="btn" id="pickBtn">Select File</button>
        <button class="btn" id="runBtn">Analyze</button>
      </div>
      <div class="audio-shell"><audio id="audioPlayer" controls style="margin-top:2px;display:none"></audio></div>
      <div id="status" style="min-height:22px;font-size:13px;color:var(--muted);margin-top:8px">Awaiting upload...</div>
      <div id="loaderWrap" class="loader-wrap"><span class="loader"></span><span>Analyzing audio, generating Grad-CAM and report...</span></div>
    </div>
  </div>
</section>
<section class="cards">
  <div class="panel card"><div class="k">Final Prediction</div><div class="v" id="pred">--</div></div>
  <div class="panel card"><div class="k">Fake Probability</div><div class="v" id="prob">--</div></div>
  <div class="panel card"><div class="k">Timestamp (UTC)</div><div class="v" id="ts" style="font-size:16px">--</div></div>
  <div class="panel card"><div class="k">Report</div><div class="v" id="reportState" style="font-size:13px">Run analysis first</div></div>
</section>
<section class="result-grid">
  <div class="panel img-wrap"><h3>Grad-CAM Evidence Overlay</h3><img id="gradcam" alt="Grad-CAM output" /></div>
  <div class="panel img-wrap"><h3>Attention Dynamics</h3><img id="attn" alt="Attention output" /></div>
</section>
<section class="panel" id="explainBox" style="padding:18px;white-space:pre-wrap;line-height:1.55;margin-top:14px">Human explanation will appear here after analysis.</section>
<section class="panel" style="padding:14px;margin-top:12px;text-align:center">
  <button class="btn download-unique" id="downloadBtn" disabled>⬇ Download Forensic Report (.pdf)</button>
</section>
<script>
const input = document.getElementById('audioInput');
const pickBtn = document.getElementById('pickBtn');
const runBtn = document.getElementById('runBtn');
const statusEl = document.getElementById('status');
const predEl = document.getElementById('pred');
const probEl = document.getElementById('prob');
const tsEl = document.getElementById('ts');
const gradEl = document.getElementById('gradcam');
const attnEl = document.getElementById('attn');
const explainEl = document.getElementById('explainBox');
const dropzone = document.getElementById('dropzone');
const audioPlayer = document.getElementById('audioPlayer');
const downloadBtn = document.getElementById('downloadBtn');
const reportState = document.getElementById('reportState');
const loaderWrap = document.getElementById('loaderWrap');
let audioObjectUrl = null;
let downloadUrl = null;
pickBtn.onclick = () => input.click();
input.onchange = () => {
  if (!input.files.length){ statusEl.textContent='Awaiting upload...'; audioPlayer.style.display='none'; return; }
  const file = input.files[0];
  statusEl.textContent = `Selected: ${file.name}`;
  if(audioObjectUrl) URL.revokeObjectURL(audioObjectUrl);
  audioObjectUrl = URL.createObjectURL(file);
  audioPlayer.src = audioObjectUrl;
  audioPlayer.style.display = 'block';
};
['dragenter','dragover'].forEach(evt => dropzone.addEventListener(evt, e => {e.preventDefault(); e.stopPropagation(); dropzone.classList.add('drag');}));
['dragleave','drop'].forEach(evt => dropzone.addEventListener(evt, e => {e.preventDefault(); e.stopPropagation(); dropzone.classList.remove('drag');}));
dropzone.addEventListener('drop', e => { if (e.dataTransfer.files.length){ input.files = e.dataTransfer.files; input.dispatchEvent(new Event('change')); } });
runBtn.onclick = async () => {
  if (!input.files.length) { statusEl.textContent = 'Please choose one audio file first.'; return; }
  const fd = new FormData(); fd.append('audio', input.files[0]);
  statusEl.textContent = 'Running deepfake analysis... please wait.'; runBtn.disabled = true; pickBtn.disabled = true; loaderWrap.style.display = 'flex';
  try {
    const res = await fetch('/api/analyze', { method: 'POST', body: fd });
    const data = await res.json(); if (!res.ok) throw new Error(data.error || 'Unknown failure');
    predEl.textContent = data.prediction; predEl.className = 'v ' + (data.prediction === 'FAKE' ? 'bad' : 'good');
    probEl.textContent = Number(data.fake_probability).toFixed(3); tsEl.textContent = data.timestamp_utc;
    explainEl.textContent = data.explanation;
    const cacheBust = `?t=${Date.now()}`;
    gradEl.src = data.gradcam_url + cacheBust; attnEl.src = data.attention_url + cacheBust;
    statusEl.textContent = 'Analysis completed successfully.';
    downloadUrl = data.report_download_url;
    downloadBtn.disabled = !downloadUrl;
    reportState.textContent = downloadUrl ? 'Ready to download' : 'Not available';
  } catch (err) { statusEl.textContent = 'Failed: ' + err.message; }
  finally { runBtn.disabled = false; pickBtn.disabled = false; loaderWrap.style.display = 'none'; }
};
downloadBtn.onclick = () => { if(downloadUrl){ window.location.href = downloadUrl; } };
</script>
"""
    return render_template_string(_shell("Deepfake Voice Detection - Analyzer", content))


@app.get("/overcome")
def overcome_page():
    content = """
<section class="panel" style="padding:24px;margin-top:18px">
  <h1 style="font-size:42px;margin:0 0 8px">🛡 Overcome Deepfake Voice Risks</h1>
  <p style="color:var(--muted)">Synthetic voice fraud is used in social engineering, executive impersonation, voice-phishing, misinformation, and fabricated legal narratives. Modern defense must combine technology, operations, and governance.</p>
  <img src="https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=1400&q=80" alt="overcome deepfake risks" style="margin-top:12px;width:100%;height:250px;object-fit:cover;border-radius:14px;border:1px solid var(--line)"/>
  <p style="color:var(--muted);margin-top:10px">This module is focused on practical response: detect suspicious voice quickly, explain model behavior clearly, and package evidence in a format suitable for organizational review.</p>
  <p style="color:var(--muted)">Key operational objective: reduce false trust in voice clips by forcing multi-layer verification (technical evidence + human verification + process controls).</p>
</section>
<section class="panel" style="padding:20px;margin-top:12px">
  <h2>⚠ Problem statement</h2>
  <ul style="color:var(--muted);line-height:1.7">
    <li><strong>Business fraud:</strong> executive voice impersonation for urgent transfer requests.</li>
    <li><strong>Legal confusion:</strong> fabricated clips that appear authentic to non-experts.</li>
    <li><strong>Identity abuse:</strong> bypassing weak voice-authentication channels.</li>
    <li><strong>Public trust erosion:</strong> misinformation campaigns using cloned voices.</li>
  </ul>
  <img src="https://images.unsplash.com/photo-1516321497487-e288fb19713f?auto=format&fit=crop&w=1400&q=80" alt="problem statement visual" style="margin-top:10px;width:100%;height:230px;object-fit:cover;border-radius:14px;border:1px solid var(--line)"/>
  <p style="color:var(--muted);margin-top:8px">Problem severity increases when fake audio spreads quickly and verification teams lack transparent technical evidence. A modern solution must be fast, explainable, and report-ready.</p>
  <p style="color:var(--muted)">Without structured detection processes, institutions can face financial losses, legal disputes, and long-term reputational damage from manipulated audio incidents.</p>
</section>
<section class="panel" style="padding:20px;margin-top:12px;display:grid;grid-template-columns:repeat(3,1fr);gap:14px">
  <div class="module"><strong>✅ Early Screening</strong><br/><small>Rapidly triages suspicious clips before escalation and reduces analyst overload.</small></div>
  <div class="module"><strong>✅ Evidence Visibility</strong><br/><small>Grad-CAM + attention maps provide transparent forensic reasoning support.</small></div>
  <div class="module"><strong>✅ Documented Proof</strong><br/><small>Downloadable report preserves technical findings for investigation trail.</small></div>
</section>
<section class="panel" style="padding:20px;margin-top:12px">
  <h2 style="text-align:center">Who Needs Deepfake Detection?</h2>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px">
    <div style="background:#fff;color:#1a2340;border-radius:14px;overflow:hidden;border:1px solid #d8e2f3">
      <img src="https://images.unsplash.com/photo-1563986768609-322da13575f3?auto=format&fit=crop&w=1200&q=80" style="width:100%;height:170px;object-fit:cover"/>
      <div style="padding:14px"><div style="font-weight:800">● Security Operations (SecOps)</div><p style="margin:8px 0 0;color:#4b5879">Check internal and external voice communications for impersonation risks.</p></div>
    </div>
    <div style="background:#fff;color:#1a2340;border-radius:14px;overflow:hidden;border:1px solid #d8e2f3">
      <img src="https://images.unsplash.com/photo-1450101499163-c8848c66ca85?auto=format&fit=crop&w=1200&q=80" style="width:100%;height:170px;object-fit:cover"/>
      <div style="padding:14px"><div style="font-weight:800">● Legal & Compliance</div><p style="margin:8px 0 0;color:#4b5879">Validate submitted audio evidence and suspicious voice records.</p></div>
    </div>
    <div style="background:#fff;color:#1a2340;border-radius:14px;overflow:hidden;border:1px solid #d8e2f3">
      <img src="https://images.unsplash.com/photo-1529078155058-5d716f45d604?auto=format&fit=crop&w=1200&q=80" style="width:100%;height:170px;object-fit:cover"/>
      <div style="padding:14px"><div style="font-weight:800">● Editorial & Media Teams</div><p style="margin:8px 0 0;color:#4b5879">Verify authenticity before publishing interviews and voice statements.</p></div>
    </div>
  </div>
</section>
<section class="panel" style="padding:20px;margin-top:12px">
  <h2>📸 Real-world impact visuals</h2>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:10px">
    <img src="https://images.unsplash.com/photo-1550751827-4bd374c3f58b?auto=format&fit=crop&w=1200&q=80" alt="cyber risk visual" style="width:100%;height:260px;object-fit:cover;border-radius:14px;border:1px solid var(--line)"/>
    <img src="https://images.unsplash.com/photo-1510511459019-5dda7724fd87?auto=format&fit=crop&w=1200&q=80" alt="audio investigation visual" style="width:100%;height:260px;object-fit:cover;border-radius:14px;border:1px solid var(--line)"/>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
    <img src="https://images.unsplash.com/photo-1563986768609-322da13575f3?auto=format&fit=crop&w=1200&q=80" alt="security operations visual" style="width:100%;height:260px;object-fit:cover;border-radius:14px;border:1px solid var(--line)"/>
    <img src="https://images.unsplash.com/photo-1520607162513-77705c0f0d4a?auto=format&fit=crop&w=1200&q=80" alt="forensics report visual" style="width:100%;height:260px;object-fit:cover;border-radius:14px;border:1px solid var(--line)"/>
  </div>
</section>
<section class="panel" style="padding:20px;margin-top:12px">
  <h2>🧭 Overcome solution strategy</h2>
  <p style="color:var(--muted)">Use this with incident response controls: retain raw audio, capture metadata, require analyst review, and avoid automated punitive decisions from a single model output.</p>
  <p style="color:var(--muted)">Go to the Analyzer page, process audio, then download the aligned forensic report containing prediction, AI explanation, Grad-CAM, and attention graph as proof artifact.</p>
  <p style="color:var(--muted)">For stronger protection: combine this detector with caller verification workflows, challenge-response prompts, and post-call anomaly checks in telecom / enterprise environments.</p>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px">
    <div class="module">🧾 <strong>Governance:</strong><br/><small>Policy + reviewer sign-off before legal use.</small></div>
    <div class="module">🛠 <strong>Technical:</strong><br/><small>Model evidence + metadata + tamper checks.</small></div>
    <div class="module">👥 <strong>Human loop:</strong><br/><small>Analyst confirmation to avoid blind trust in automation.</small></div>
  </div>
  <div class="cta-wrap"><a href="/analyze" class="cta-btn">🚀 Open Analyzer</a></div>
</section>
"""
    return render_template_string(_shell("Deepfake Voice Detection - Overcome", content))


@app.get("/xai/<path:filename>")
def serve_xai(filename: str):
    return send_from_directory(XAI_DIR, filename)


def _to_data_uri(image_path: Path) -> str:
    data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    ext = image_path.suffix.lower().replace(".", "") or "png"
    mime = "jpeg" if ext in {"jpg", "jpeg"} else ext
    return f"data:image/{mime};base64,{data}"


def _write_report_doc(report_data: dict) -> Path:
    report_path = XAI_DIR / "forensic_report.doc"
    grad_uri = _to_data_uri(Path(report_data["gradcam_path"]))
    attn_uri = _to_data_uri(Path(report_data["attention_path"]))
    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><title>Forensic Deepfake Voice Report</title></head>
<body>
<h1>Digital Forensics Voice Authenticity Report</h1>
<p>Generated at: {report_data['timestamp_utc']} | Audio file: {report_data['audio_filename']}</p>
<p>Prediction: {report_data['prediction']} | Fake probability: {report_data['fake_probability']:.4f} | Threshold: {report_data.get('threshold', 0.5):.2f}</p>
<pre>{report_data['explanation']}</pre>
<img src="{grad_uri}" alt="gradcam"/>
<img src="{attn_uri}" alt="attention"/>
</body></html>"""
    report_path.write_text(html_doc, encoding="utf-8")
    return report_path


def _write_report_pdf(report_data: dict) -> Path:
    pdf_path = XAI_DIR / "forensic_report.pdf"
    grad_path = Path(report_data["gradcam_path"])
    attn_path = Path(report_data["attention_path"])

    with PdfPages(pdf_path) as pdf:
        header_lines = [
            "Digital Forensics Voice Authenticity Report",
            "",
            f"Generated at: {report_data['timestamp_utc']}",
            f"Audio file: {report_data['audio_filename']}",
            f"Prediction: {report_data['prediction']}",
            f"Fake probability: {report_data['fake_probability']:.4f}",
            f"Threshold: {report_data.get('threshold', 0.5):.2f}",
            "",
            "AI Explanation:",
        ]
        wrapped_explain: list[str] = []
        for para in str(report_data["explanation"]).splitlines() or [""]:
            wrapped_explain.extend(textwrap.wrap(para, width=110) or [""])

        all_lines = header_lines + wrapped_explain
        lines_per_page = 45
        for start in range(0, len(all_lines), lines_per_page):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor("white")
            plt.axis("off")
            page_lines = all_lines[start : start + lines_per_page]
            y = 0.97
            for line in page_lines:
                size = 12 if line == "Digital Forensics Voice Authenticity Report" else 9.5
                color = "#111111" if start == 0 and y > 0.75 else "#243b53"
                plt.text(0.05, y, line, fontsize=size, va="top", ha="left", color=color)
                y -= 0.02
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if grad_path.exists():
            fig = plt.figure(figsize=(11.69, 8.27))
            plt.imshow(mpimg.imread(grad_path))
            plt.title("Grad-CAM Evidence Overlay")
            plt.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if attn_path.exists():
            fig = plt.figure(figsize=(11.69, 8.27))
            plt.imshow(mpimg.imread(attn_path))
            plt.title("Attention Dynamics")
            plt.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return pdf_path


@app.post("/api/analyze")
def analyze_audio():
    if "audio" not in request.files:
        return jsonify({"error": "Audio file is required."}), 400

    file = request.files["audio"]
    if not file.filename:
        return jsonify({"error": "No selected file."}), 400

    if not _is_allowed_file(file.filename):
        return jsonify({"error": "Unsupported audio format."}), 400

    uploads_dir = XAI_DIR / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename).suffix.lower()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    safe_name = _slugify_name(file.filename)
    saved_path = uploads_dir / f"{safe_name}_{run_id}{suffix}"
    file.save(saved_path)

    try:
        result = run_single_audio_pipeline(
            audio_path=str(saved_path),
            model_path=DEFAULT_MODEL_PATH,
            gemini_model=DEFAULT_GEMINI_MODEL,
            #gemini_api_key=os.getenv("GEMINI_API_KEY", DEFAULT_GEMINI_API_KEY),
        )
    except Exception as exc:
        return jsonify({"error": f"Inference failed: {exc}"}), 500

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    report_data = {
        "audio_filename": file.filename,
        "prediction": result["prediction"],
        "fake_probability": result["fake_probability"],
        "threshold": result["threshold"],
        "explanation": result["explanation"],
        "gradcam_path": result["gradcam_path"],
        "attention_path": result["attention_path"],
        "timestamp_utc": timestamp_utc,
    }
    report_json = XAI_DIR / "latest_report.json"
    report_json.write_text(json.dumps(report_data, ensure_ascii=False), encoding="utf-8")

    return jsonify(
        {
            "prediction": result["prediction"],
            "fake_probability": result["fake_probability"],
            "threshold": result["threshold"],
            "explanation": result["explanation"],
            "gradcam_url": "/xai/gradcam_overlay.png",
            "attention_url": "/xai/attention_weights.png",
            "timestamp_utc": timestamp_utc,
            "report_download_url": "/download/report/pdf",
            "debug_summary": result["debug"],
        }
    )


@app.get("/download/report")
def download_report_doc():
    report_json = XAI_DIR / "latest_report.json"
    if not report_json.exists():
        return jsonify({"error": "No analysis report available yet. Run analysis first."}), 404

    report_data = json.loads(report_json.read_text(encoding="utf-8"))
    report_doc_path = _write_report_doc(report_data)
    return send_file(
        report_doc_path,
        mimetype="application/msword",
        as_attachment=True,
        download_name=f"forensic_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.doc",
    )


@app.get("/download/report/pdf")
def download_report_pdf():
    report_json = XAI_DIR / "latest_report.json"
    if not report_json.exists():
        return jsonify({"error": "No analysis report available yet. Run analysis first."}), 404

    report_data = json.loads(report_json.read_text(encoding="utf-8"))
    report_pdf_path = _write_report_pdf(report_data)
    return send_file(
        report_pdf_path,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"forensic_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pdf",
    )


if __name__ == "__main__":
    XAI_DIR.mkdir(parents=True, exist_ok=True)
    print("Starting Audio Authenticity web app on http://127.0.0.1:7860")
    app.run(host="127.0.0.1", port=7860, debug=False)
