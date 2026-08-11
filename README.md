# SparkIntel — AI-Powered Detection of Synthetic Media & Phishing Threats in Securities Markets

Built for the **SEBI Securities Market TechSprint** (Problem Statement: *AI-Driven Detection of
Synthetic Media and Phishing Attacks in Securities Markets*).

SparkIntel detects deepfakes, voice clones, phishing emails, fake circulars, and synthetic social
posts targeting investors — and separately lets investors verify whether a communication genuinely
came from SEBI, a broker, or an exchange. Unlike single-message classifiers, it also models an
investor's full interaction cycle (phishing email → link click → fake broker site → voice call →
WhatsApp tip) as a sequence, catching multi-stage campaigns that look individually mild but are
unambiguous fraud as a chain.

**Team:** Gourab Mahato, Ansh Aryan, Anubhav Sarangi — BIT Mesra

## What's in this demo

| Tab | What it does | Status |
|---|---|---|
| 📧 Text / Circular Check | SBERT + classifier phishing score, with human-legible phrase-level explanations | Needs `01_phishing_text_detector.ipynb`'s output files |
| 🔗 Trust Verification | Digital-signature issue/verify + QR codes, catches forged IDs, tampered content, and revoked notices | **Works out of the box** |
| 🔀 Fraud Journey Sequence | GRU/BiGRU sequence model scoring a multi-stage investor interaction | **Works out of the box** |
| 🎭 Deepfake / Voice Check | Pretrained image/audio models for deepfake and voice-clone detection | Needs live Hugging Face Hub access |

---

## Free deployment (recommended): Streamlit Community Cloud

This repository now ships a **lean hosted profile** in `requirements.txt` and pins Streamlit's
runtime to Python 3.11 through `runtime.txt`. It deliberately excludes TensorFlow, PyTorch and
Hugging Face inference dependencies: on free tiers their combined download and memory footprint is
a common cause of installation failures. The Trust Verification workflow and the transparent
Fraud Journey risk engine remain fully usable in this profile.

For a laptop, Render paid instance, or another server with at least 4 GB RAM, install
`requirements-full.txt` to enable the TensorFlow sequence model and the optional transformer-based
text/image/audio models.

Hugging Face Gradio/Docker Spaces now require a **PRO subscription** for CPU hosting.  
**Streamlit Community Cloud is free** for public apps.

### Steps

1. **Push this folder to a public GitHub repo** (must include these files):
   - `app.py`
   - `requirements.txt`
   - `trust_verification.py`
   - `gru_sequence_model.keras`
   - `.streamlit/config.toml`

2. Go to **[share.streamlit.io](https://share.streamlit.io)** → sign in with GitHub

3. Click **Create app** → select your repo → set main file to **`app.py`** → **Deploy**

4. Your live URL will be: `https://<app-name>.streamlit.app`

First build takes **5–15 minutes** (downloads torch + tensorflow). The app sleeps after ~7 days
of inactivity but wakes on the next visit.

### Files to commit to GitHub

```
app.py
requirements.txt
trust_verification.py
gru_sequence_model.keras
.streamlit/config.toml
README.md
```

Do **not** need to upload notebooks for deployment.

---

## Other free options

| Platform | Cost | Notes |
|---|---|---|
| **[Streamlit Community Cloud](https://share.streamlit.io)** | Free | **Best choice** — use `app.py` as-is |
| **HF ZeroGPU Spaces** | Free (limited) | Account must be 30+ days old; max 2 Spaces; use `app_gradio.py` |
| **Render.com** | Free tier | Spins down after 15 min idle; cold starts are slow |
| **HF Static Spaces** | Free | Static HTML only — won't run Python ML models |

---

## Paid option: Hugging Face PRO ($9/mo)

If you subscribe to HF PRO, deploy with Gradio:
- Rename `requirements-gradio.txt` → `requirements.txt`
- Use `app_gradio.py` as the entry file (or copy to `app.py`)
- Create Space with SDK: **Gradio → Blank**

---

## Setup (optional tabs)

Two tabs work immediately after deploy. To enable the rest:

1. Run `01_phishing_text_detector.ipynb` in Colab
2. Add the 3 `.joblib` files to your GitHub repo
3. Streamlit auto-redeploys on push

Built on the SBERT + GRU/BiGRU pipeline from [LogIntel-AI](https://github.com/Gourab357/LogIntel-AI),
extended with trust verification and multimodal synthetic-media detectors.
