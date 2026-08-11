"""
SparkIntel — Streamlit demo app.

Free deployment: Streamlit Community Cloud (https://share.streamlit.io)
  → push to GitHub, deploy app.py + requirements.txt + trust_verification.py
     + gru_sequence_model.keras

Paid/alternative: Hugging Face Spaces with Gradio → use app_gradio.py + requirements-gradio.txt
"""
import io
import os
import re

import numpy as np
import streamlit as st
from PIL import Image

from trust_verification import (
    Issuer, VerificationRegistry,
    make_verification_qr, embed_watermark, extract_watermark,
)

st.set_page_config(page_title="SparkIntel", page_icon="🛡️", layout="wide")

ARTIFACT_DIR = os.path.dirname(os.path.abspath(__file__))

st.markdown("""
<style>
    :root { --spark-bg: #f5f8fc; --spark-card: #ffffff; --spark-text: #102a43;
            --spark-muted: #52677d; --spark-border: #d7e2ee; --spark-accent: #00a6a6; }
    .stApp { background: var(--spark-bg); color: var(--spark-text); }
    .block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem; }
    [data-testid="stMetric"] { background: var(--spark-card); border: 1px solid var(--spark-border);
        border-radius: 14px; padding: 14px; }
    [data-testid="stSidebar"] { border-right: 1px solid var(--spark-border); }
    .spark-hero { background: linear-gradient(120deg, #073b5c, #007c86); color: #fff; padding: 1.5rem;
        border-radius: 18px; margin-bottom: 1.2rem; box-shadow: 0 10px 28px rgba(4, 72, 94, .18); }
    .spark-hero h1 { color: #fff !important; font-size: 2.35rem; margin: 0; }
    .spark-hero p { color: #e7f8fa !important; margin: .35rem 0 0; }
    .spark-kicker { color: #99eff1 !important; font-size: .75rem; font-weight: 700; letter-spacing: .12em; }
    .spark-card { background: var(--spark-card); border: 1px solid var(--spark-border); border-radius: 14px;
        padding: 1rem 1.15rem; margin: .5rem 0 1rem; }
    .spark-card h3 { margin-top: 0; }
    .spark-muted { color: var(--spark-muted); }
    @media (prefers-color-scheme: dark) {
        :root { --spark-bg: #101923; --spark-card: #182534; --spark-text: #edf6ff;
                --spark-muted: #b4c4d4; --spark-border: #314558; --spark-accent: #4ddbd8; }
        .stApp, [data-testid="stAppViewContainer"] { background: var(--spark-bg) !important; color: var(--spark-text) !important; }
        [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stSidebar"] { background: #111c28 !important; }
        h1, h2, h3, p, label, [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] { color: var(--spark-text) !important; }
        [data-baseweb="input"] > div, [data-baseweb="textarea"] > div { background: #213246 !important; color: #fff !important; border-color: #465d73 !important; }
        [data-baseweb="input"] input, textarea { color: #fff !important; }
        [data-testid="stFileUploaderDropzone"] { background: #1b2a3a !important; border-color: #486278 !important; }
    }
</style>
""", unsafe_allow_html=True)


def artifact_path(name: str) -> str:
    return os.path.join(ARTIFACT_DIR, name)


def fallback_text_analysis(text: str) -> tuple[float, list[str], list[str]]:
    """Explainable safety screen that needs no downloaded model."""
    normalized = text.lower()
    risk_rules = {
        "urgent|immediately|within \\d+ hours": "Artificial urgency",
        "kyc|account.{0,20}(freeze|block|suspend)": "Account-threat language",
        "click|bit\\.ly|tinyurl|https?://": "Link or click-through request",
        "otp|pin|password|cvv": "Request for sensitive credentials",
        "guaranteed|assured|risk-free": "Unrealistic investment promise",
        "whatsapp|telegram": "Unverified social-media channel",
    }
    safe_rules = {
        "do not share.*otp": "Warns against sharing credentials",
        "official website|investor grievance": "Uses an official support route",
    }
    risks = [label for pattern, label in risk_rules.items() if re.search(pattern, normalized)]
    safeguards = [label for pattern, label in safe_rules.items() if re.search(pattern, normalized)]
    score = min(0.95, 0.10 + 0.16 * len(risks) - 0.05 * len(safeguards))
    return score, risks, safeguards


def render_text_fallback(note: str) -> None:
    st.markdown('<div class="spark-card"><h3>Instant message screening</h3><p class="spark-muted">'
                'Explainable, rule-based triage for the hosted demo. It flags common fraud signals; '
                'use Trust Verification to confirm an issuer.</p></div>', unsafe_allow_html=True)
    text = st.text_area("Message text", value=("URGENT: Your KYC has expired. Update immediately "
                                               "or your trading account will be frozen within 24 hours."),
                        height=150, key="fallback_text_input")
    if st.button("Screen message", type="primary", key="fallback_text_button"):
        score, risks, safeguards = fallback_text_analysis(text)
        label = "HIGH RISK" if score >= 0.55 else "REVIEW CAREFULLY" if score >= 0.30 else "LOWER RISK"
        left, right = st.columns([1, 2])
        with left:
            st.metric("Screening risk", f"{score:.0%}")
            if score >= 0.55:
                st.error(label)
            elif score >= 0.30:
                st.warning(label)
            else:
                st.success(label)
        with right:
            st.write("**Signals found**")
            for signal in risks or ["No common phishing pattern detected"]:
                st.write(f"- {signal}")
            if safeguards:
                st.caption("Safeguards: " + ", ".join(safeguards))
        st.info("Recommended action: do not use links or share credentials. Verify the communication ID in the Trust Verification tab.")
    if note:
        st.caption(note)


def render_image_triage(img: Image.Image) -> None:
    pixels = np.asarray(img.convert("RGB"), dtype=np.float32)
    variation = float(pixels.std())
    metadata_present = bool(img.getexif())
    st.markdown('<div class="spark-card"><h3>Image intake complete</h3><p class="spark-muted">'
                'This hosted demo records basic forensic context. It is not a deepfake verdict; '
                'the full vision model is available only in the dedicated compute profile.</p></div>',
                unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric("Resolution", f"{img.width} x {img.height}")
    b.metric("Pixel variation", f"{variation:.0f}")
    c.metric("EXIF metadata", "Present" if metadata_present else "Not present")
    st.warning("Recommended action: treat unverified market-media as suspicious and verify the underlying issuer communication.")


def render_audio_triage(uploaded_audio) -> None:
    size_kb = len(uploaded_audio.getvalue()) / 1024
    st.markdown('<div class="spark-card"><h3>Audio intake complete</h3><p class="spark-muted">'
                'The hosted demo accepts the clip for review. A voice-clone probability requires '
                'the optional audio model on dedicated compute.</p></div>', unsafe_allow_html=True)
    st.metric("Uploaded audio", f"{size_kb:.0f} KB")
    st.warning("Recommended action: do not act on trading instructions delivered by an unverified voice call.")


st.title("🛡️ SparkIntel")
st.caption("AI-Powered Detection of Synthetic Media & Phishing Threats in Securities Markets — "
           "SEBI Securities Market TechSprint prototype")

st.markdown("""
<div class="spark-hero">
  <div class="spark-kicker">SEBI TECHSPRINT • INVESTOR PROTECTION</div>
  <h1>Spot the signal. Verify the source. Protect the investor.</h1>
  <p>One workspace for screening suspicious market communications, authenticating issuers, and tracing multi-stage fraud.</p>
</div>
""", unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Screening", "Text & media")
col_b.metric("Verification", "Cryptographic")
col_c.metric("Journey view", "Multi-stage")

tab_text, tab_trust, tab_sequence, tab_media = st.tabs([
    "📧 Text / Circular Check",
    "🔗 Trust Verification",
    "🔀 Fraud Journey Sequence",
    "🎭 Deepfake / Voice Check",
])

with st.sidebar:
    st.header("Investor safety desk")
    st.caption("A SEBI TechSprint prototype for triage, verification, and investor action.")
    st.markdown("**Recommended demo flow**")
    st.markdown("1. Screen a suspicious message\n2. Verify its issuer\n3. Map the fraud journey")
    st.success("Core modules online")
    st.caption("Hosted mode uses transparent fallbacks when dedicated ML models are unavailable.")

# =============================================================================
# TAB 1 — Text / Circular Check  (needs notebook 1's saved .joblib artifacts)
# =============================================================================
with tab_text:
    st.subheader("Phishing & Fake-Circular Text Detector")
    st.caption("Paste an email, WhatsApp forward, or circular. Scores it with SBERT embeddings "
               "+ a trained classifier, and separately highlights the phrases driving the score.")

    text_artifacts_ok = os.path.exists(artifact_path("phishing_rf_model.joblib"))

    if not text_artifacts_ok:
        render_text_fallback("Optional SBERT model artifacts were not uploaded; hosted screening is active.")
    else:
        @st.cache_resource(show_spinner="Loading text models...")
        def load_text_artifacts():
            import joblib
            from sentence_transformers import SentenceTransformer
            rf = joblib.load(artifact_path("phishing_rf_model.joblib"))
            tfidf = joblib.load(artifact_path("phishing_tfidf_vectorizer.joblib"))
            kw_clf = joblib.load(artifact_path("phishing_keyword_model.joblib"))
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            return rf, tfidf, kw_clf, embedder

        try:
            rf, tfidf, kw_clf, embedder = load_text_artifacts()

            def explain_keywords(text, top_k=6):
                vec = tfidf.transform([text])
                idx = vec.nonzero()[1]
                contrib = [(tfidf.get_feature_names_out()[i], float(vec[0, i] * kw_clf.coef_[0, i])) for i in idx]
                contrib.sort(key=lambda t: -abs(t[1]))
                return contrib[:top_k]

            default_example = ("URGENT: Your KYC has expired. Update immediately at this link "
                                "or your trading account will be frozen within 24 hours.")
            user_text = st.text_area("Message text", value=default_example, height=140)

            if st.button("Check this message", type="primary", key="check_text"):
                emb = embedder.encode([user_text])
                score = float(rf.predict_proba(emb)[0, 1])
                verdict = ("LIKELY PHISHING" if score > 0.6 else
                           "SUSPICIOUS" if score > 0.35 else "LIKELY LEGITIMATE")

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Phishing score", f"{score:.2f}")
                    if verdict == "LIKELY PHISHING":
                        st.error(verdict)
                    elif verdict == "SUSPICIOUS":
                        st.warning(verdict)
                    else:
                        st.success(verdict)
                with col2:
                    st.write("**Phrases driving this score:**")
                    for phrase, contrib in explain_keywords(user_text):
                        direction = "toward phishing" if contrib > 0 else "toward legitimate"
                        st.write(f"- `{phrase}` — {direction} ({contrib:+.3f})")
        except Exception:
            render_text_fallback("Dedicated text model is unavailable in this hosted profile; explainable screening is active.")

# =============================================================================
# TAB 2 — Trust Verification
# =============================================================================
with tab_trust:
    st.subheader("Authenticity / Trust Verification")
    st.caption("Layer 2: independent of whether content *looks* synthetic — is this provably, "
               "cryptographically the thing SEBI / a broker / an exchange actually sent?")

    if "registry" not in st.session_state:
        st.session_state.registry = VerificationRegistry()
        st.session_state.issuers = {name: Issuer(name) for name in ("SEBI", "Sample Broker Ltd")}
        st.session_state.public_keys = {n: i.public_key for n, i in st.session_state.issuers.items()}

    registry = st.session_state.registry
    issuers = st.session_state.issuers
    public_keys = st.session_state.public_keys

    sub_issue, sub_verify = st.tabs(["Issue a signed communication", "Verify a message"])

    with sub_issue:
        issuer_name = st.selectbox("Issuing as", list(issuers.keys()))
        message = st.text_area("Communication text",
                                value="Investors are advised that trading hours on the upcoming "
                                      "holiday remain unchanged.", key="issue_text")
        if st.button("Sign & register", key="sign_btn"):
            rec = registry.register(issuers[issuer_name], message, channel="email")
            st.session_state.last_record_id = rec.record_id
            st.success(f"Registered. Verification ID: `{rec.record_id}`")
            qr_img = make_verification_qr(rec)
            st.image(qr_img, width=180, caption="QR code that would ship in the message footer")

    with sub_verify:
        st.write("Paste a verification ID and the message text an investor actually received — "
                 "exactly as they'd do after scanning the QR code or typing in the ID.")
        check_id = st.text_input("Verification ID", value=st.session_state.get("last_record_id", ""))
        check_text = st.text_area("Message text received", key="verify_text")
        if st.button("Verify", key="verify_btn"):
            result = registry.verify_message(check_id, check_text, public_keys)
            if result["authentic"]:
                st.success(f"Verified: {result['status']} - {result['detail']}")
            else:
                st.error(f"Not verified: {result['status']} - {result['detail']}")

        st.divider()
        st.caption("Try it: register a message above, then paste the *exact* text back here to see "
                   "VERIFIED — or paste the same ID with slightly different/extra text (e.g. an "
                   "added link) to see CONTENT_MISMATCH: a fraudster reusing a real reference number "
                   "under tampered content.")

# =============================================================================
# TAB 3 — Fraud Journey Sequence
# =============================================================================
with tab_sequence:
    st.subheader("Multi-Stage Fraud Journey Detector")
    st.caption("Build an investor's event sequence and score it as a whole — catches campaigns "
               "a single-message classifier would miss.")

    FRAUD_EVENTS = ["phishing_email", "suspicious_link_click", "fake_broker_site_visit",
                     "spoofed_voice_call", "whatsapp_tip", "unauthorized_fund_transfer"]
    BENIGN_EVENTS = ["legit_broker_email", "portal_login", "statement_download",
                      "support_call", "app_notification", "authorized_fund_transfer"]
    EVENT_TYPES = FRAUD_EVENTS + BENIGN_EVENTS
    CHANNELS = ["email", "sms", "web", "call", "whatsapp", "app"]
    EVENT_CHANNEL = {
        "phishing_email": "email", "suspicious_link_click": "web",
        "fake_broker_site_visit": "web", "spoofed_voice_call": "call",
        "whatsapp_tip": "whatsapp", "unauthorized_fund_transfer": "app",
        "legit_broker_email": "email", "portal_login": "web",
        "statement_download": "web", "support_call": "call",
        "app_notification": "app", "authorized_fund_transfer": "app",
    }
    SCORE_PROFILE = {
        "phishing_email":             (0.70, 0.18, 0.10, 0.08, 0.60, 0.20),
        "suspicious_link_click":      (0.30, 0.15, 0.10, 0.08, 0.65, 0.18),
        "fake_broker_site_visit":     (0.25, 0.15, 0.10, 0.08, 0.70, 0.18),
        "spoofed_voice_call":         (0.15, 0.10, 0.65, 0.18, 0.15, 0.10),
        "whatsapp_tip":               (0.50, 0.20, 0.10, 0.08, 0.45, 0.20),
        "unauthorized_fund_transfer": (0.20, 0.12, 0.20, 0.12, 0.30, 0.15),
        "legit_broker_email":         (0.15, 0.10, 0.08, 0.06, 0.12, 0.08),
        "portal_login":               (0.08, 0.06, 0.08, 0.05, 0.08, 0.05),
        "statement_download":         (0.08, 0.06, 0.08, 0.05, 0.08, 0.05),
        "support_call":               (0.08, 0.06, 0.15, 0.08, 0.08, 0.05),
        "app_notification":           (0.08, 0.06, 0.08, 0.05, 0.08, 0.05),
        "authorized_fund_transfer":   (0.08, 0.06, 0.08, 0.05, 0.08, 0.05),
    }
    EVENT2IDX = {e: i for i, e in enumerate(EVENT_TYPES)}
    CHANNEL2IDX = {c: i for i, c in enumerate(CHANNELS)}
    N_EVENT, N_CHANNEL = len(EVENT_TYPES), len(CHANNELS)
    FEAT_DIM = N_EVENT + N_CHANNEL + 3 + 1
    MAX_LEN = 12
    PAD_VALUE = 0.0

    def event_vector(event_type, hours_since_prev, rng):
        vec = np.zeros(FEAT_DIM, dtype=np.float32)
        vec[EVENT2IDX[event_type]] = 1.0
        vec[N_EVENT + CHANNEL2IDX[EVENT_CHANNEL[event_type]]] = 1.0
        pm, ps, vm, vs, um, us = SCORE_PROFILE[event_type]
        scores = rng.normal([pm, vm, um], [ps, vs, us])
        vec[N_EVENT + N_CHANNEL: N_EVENT + N_CHANNEL + 3] = np.clip(scores, 0, 1)
        vec[-1] = np.log1p(hours_since_prev)
        return vec

    def logit(p, eps=1e-6):
        p = min(max(p, eps), 1 - eps)
        return float(np.log(p / (1 - p)))

    def render_hosted_journey():
        """Dependency-free fallback used by the free hosted demo."""
        st.info("Hosted demo mode: transparent risk scoring is active. Install "
                "requirements-full.txt on a dedicated server to use the TensorFlow model.")
        if "journey" not in st.session_state:
            st.session_state.journey = []
        left, middle, right = st.columns([2, 1, 1])
        with left:
            event = st.selectbox("Event type", EVENT_TYPES, key="hosted_event")
        with middle:
            gap = st.number_input("Hours since previous", min_value=0.0, value=1.0,
                                  step=0.5, key="hosted_gap")
        with right:
            st.write("")
            st.write("")
            if st.button("Add to journey", key="hosted_add",
                         disabled=len(st.session_state.journey) >= MAX_LEN):
                st.session_state.journey.append((event, gap))
                st.rerun()
        if not st.session_state.journey:
            return
        st.write("**Current journey:** " + " -> ".join(
            f"{event} (+{gap:.1f}h)" for event, gap in st.session_state.journey))
        clear, score = st.columns(2)
        with clear:
            if st.button("Clear journey", key="hosted_clear"):
                st.session_state.journey = []
                st.rerun()
        with score:
            score_clicked = st.button("Score this journey", type="primary", key="hosted_score")
        if score_clicked:
            fraud_events = [event for event, _ in st.session_state.journey if event in FRAUD_EVENTS]
            escalation = sum(event in {"suspicious_link_click", "fake_broker_site_visit",
                                        "unauthorized_fund_transfer"} for event in fraud_events)
            risk = min(0.98, 0.08 + 0.18 * len(fraud_events) + 0.10 * escalation)
            st.metric("Fraud-campaign score", f"{risk:.3f}")
            if risk > 0.5:
                st.error("LIKELY FRAUD CAMPAIGN")
            else:
                st.success("LIKELY BENIGN ACTIVITY")
            st.write("**Risk drivers:** " + (", ".join(fraud_events) if fraud_events
                     else "no high-risk stages detected"))

    sequence_model_ok = os.path.exists(artifact_path("gru_sequence_model.keras"))

    if not sequence_model_ok:
        st.info(
            "**Not set up yet.** Run `04_fraud_sequence_model.ipynb` in Colab, copy "
            "`gru_sequence_model.keras` into this folder, and redeploy."
        )
    else:
        @st.cache_resource(show_spinner="Loading sequence model...")
        def load_sequence_model():
            import tensorflow as tf
            return tf.keras.models.load_model(artifact_path("gru_sequence_model.keras"))

        try:
            seq_model = load_sequence_model()
            if "journey" not in st.session_state:
                st.session_state.journey = []

            st.write("Add events in the order they happened (max 12):")
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                new_event = st.selectbox("Event type", EVENT_TYPES, key="new_event_select")
            with col2:
                gap_hours = st.number_input("Hours since previous", min_value=0.0, value=1.0, step=0.5)
            with col3:
                st.write("")
                st.write("")
                add_disabled = len(st.session_state.journey) >= MAX_LEN
                if st.button("Add to journey", disabled=add_disabled):
                    st.session_state.journey.append((new_event, gap_hours))
                    st.rerun()

            if st.session_state.journey:
                st.write("**Current journey:** " +
                          " → ".join(f"{e} (+{g:.1f}h)" for e, g in st.session_state.journey))
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Clear journey"):
                        st.session_state.journey = []
                        st.rerun()
                with col_b:
                    score_clicked = st.button("Score this journey", type="primary")

                if score_clicked:
                    rng = np.random.default_rng(0)
                    events = [e for e, _ in st.session_state.journey]
                    gaps = [g for _, g in st.session_state.journey]
                    vecs = [event_vector(e, g, rng) for e, g in zip(events, gaps)]
                    padded = np.full((MAX_LEN, FEAT_DIM), PAD_VALUE, dtype=np.float32)
                    padded[: len(vecs)] = vecs
                    prob = float(seq_model.predict(padded[None, ...], verbose=0)[0, 0])
                    verdict = "LIKELY FRAUD CAMPAIGN" if prob > 0.5 else "LIKELY BENIGN ACTIVITY"

                    st.metric("Fraud-campaign score", f"{prob:.3f}")
                    if prob > 0.5:
                        st.error(verdict)
                    else:
                        st.success(verdict)

                    base = logit(prob)
                    impacts = []
                    for i, ev in enumerate(events):
                        occluded = padded.copy()
                        occluded[i] = PAD_VALUE
                        s = logit(float(seq_model.predict(occluded[None, ...], verbose=0)[0, 0]))
                        impacts.append((ev, base - s))
                    impacts.sort(key=lambda t: -t[1])

                    st.write("**Which stages drove this score:**")
                    for ev, impact in impacts:
                        st.write(f"- `{ev}`  Δ={impact:+.2f}")
        except (ImportError, ModuleNotFoundError):
            render_hosted_journey()
        except Exception as e:
            st.warning(f"Sequence model unavailable: {e}")
            render_hosted_journey()

# =============================================================================
# TAB 4 — Deepfake / Voice Check
# =============================================================================
with tab_media:
    st.subheader("Deepfake Image & Voice Clone Check")
    st.caption("Loads pretrained Hugging Face models directly — no local artifacts needed.")

    st.info("Upload media for a quick intake review. The optional full model produces a deepfake or voice-clone score; "
            "the free hosted profile shows transparent triage instead of a misleading model error.")
    media_kind = st.radio("Check a:", ["Image", "Audio clip"], horizontal=True)

    if media_kind == "Image":
        DEEPFAKE_MODEL_IDS = ["prithivMLmods/Deep-Fake-Detector-v2-Model",
                               "dima806/deepfake_vs_real_image_detection"]

        @st.cache_resource(show_spinner="Loading deepfake-image model...")
        def load_deepfake_pipe():
            from transformers import pipeline
            last_err = None
            for model_id in DEEPFAKE_MODEL_IDS:
                try:
                    return pipeline("image-classification", model=model_id), model_id
                except Exception as e:
                    last_err = e
            raise RuntimeError(f"No candidate model loaded. Last error: {last_err}")

        uploaded = st.file_uploader("Upload an image (e.g. a frame pulled from a video)",
                                     type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, width=300)
            try:
                pipe, used_model = load_deepfake_pipe()
                out = pipe(img)
                fake_score = sum(p["score"] for p in out if p["label"].lower()
                                  in ("fake", "deepfake", "ai-generated", "synthetic"))
                st.metric("Fake score", f"{fake_score:.2f}")
                st.caption(f"Model: {used_model}")
                st.json(out)
            except Exception as e:
                render_image_triage(img)

    else:
        VOICE_MODEL_IDS = ["MelodyMachine/Deepfake-audio-detection-V2",
                            "motheecreator/Deepfake-audio-detection"]

        @st.cache_resource(show_spinner="Loading voice-clone model...")
        def load_voice_pipe():
            from transformers import pipeline
            last_err = None
            for model_id in VOICE_MODEL_IDS:
                try:
                    return pipeline("audio-classification", model=model_id), model_id
                except Exception as e:
                    last_err = e
            raise RuntimeError(f"No candidate model loaded. Last error: {last_err}")

        uploaded_audio = st.file_uploader("Upload an audio clip", type=["wav", "mp3", "m4a"])
        if uploaded_audio is not None:
            st.audio(uploaded_audio)
            try:
                import soundfile as sf
                audio, sr = sf.read(io.BytesIO(uploaded_audio.getvalue()), dtype="float32", always_2d=False)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                pipe, used_model = load_voice_pipe()
                out = pipe({"array": audio, "sampling_rate": sr})
                clone_score = sum(p["score"] for p in out if p["label"].lower()
                                   in ("fake", "spoof", "cloned", "synthetic", "ai-generated"))
                st.metric("Clone score", f"{clone_score:.2f}")
                st.caption(f"Model: {used_model}")
                st.json(out)
            except Exception as e:
                render_audio_triage(uploaded_audio)

st.divider()
st.caption("SparkIntel — SEBI Securities Market TechSprint prototype.")
