import os
import re
import json
import base64
import urllib.request
import urllib.parse
from dotenv import load_dotenv
from google import genai
from google.genai import types
import streamlit as st
import streamlit.components.v1 as components

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize Gemini Client
client = genai.Client(api_key=api_key)

# Page configuration
st.set_page_config(
    page_title="Brahmāstra — SDG Educational AI",
    page_icon="🏹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Asset paths resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GOLDEN_LOTUS_PATH = os.path.join(BASE_DIR, "golden lotus.png")
BRAND_IMAGE_PATH = GOLDEN_LOTUS_PATH
BRAND_IMAGE_URL = (
    "https://thumbs.dreamstime.com/b/glowing-golden-lotus-flower-emitting-radiant-light-"
    "positioned-serene-natural-setting-radiant-golden-lotus-flower-glowing-353279508.jpg"
)


def get_image_base64(filepath: str) -> str:
    """Encodes a local image to base64 string for direct inline rendering."""
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""


lotus_b64 = get_image_base64(GOLDEN_LOTUS_PATH)
brand_b64 = lotus_b64


def _sanitize_error_message(err_str: str) -> str:
    """Sanitize error messages to prevent exposing API keys, internal URLs, or tokens."""
    if not err_str:
        return "An unexpected error occurred while generating the image."
    err_lower = err_str.lower()
    if "429" in err_lower or "resource_exhausted" in err_lower or "quota" in err_lower:
        return "Gemini image generation quota limit reached. Please check your Gemini API plan or try again shortly."
    if "safety" in err_lower or "blocked" in err_lower:
        return "The requested image could not be generated due to Gemini safety policies. Please adjust your prompt."
    if "api_key" in err_lower or "unauthorized" in err_lower or "401" in err_lower:
        return "Authentication error communicating with Gemini. Please check your API key configuration."
    cleaned = re.sub(r"key=[A-Za-z0-9_\-]+", "key=REDACTED", err_str)
    cleaned = re.sub(r"AIza[0-9A-Za-z-_]{35}", "REDACTED", cleaned)
    if len(cleaned) > 200:
        return "Gemini image service encountered an issue processing this request. Please try another prompt."
    return cleaned


# ==================== GEMINI 3.1 FLASH IMAGE GENERATION ====================

class GeminiImageGenerator:
    """Gemini image generation provider using gemini-3.1-flash-image-preview."""
    def __init__(self, gemini_client):
        self.client = gemini_client
        self.model = "gemini-3.1-flash-image-preview"

    def generate(self, prompt: str, previous_image: dict | None = None) -> dict:
        try:
            config = types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="16:9"),
            )

            if previous_image and previous_image.get("b64"):
                try:
                    image_bytes = base64.b64decode(previous_image["b64"])
                    contents = [
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_text(
                                    text=(
                                        f"Use the previous image as reference and apply this refinement: {prompt}"
                                    )
                                ),
                                types.Part.from_bytes(data=image_bytes, mime_type=previous_image.get("mime", "image/png")),
                            ],
                        )
                    ]
                    resp = self.client.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=config,
                    )
                except Exception:
                    resp = self.client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=config,
                    )
            else:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )

            for candidate in getattr(resp, "candidates", []) or []:
                for part in getattr(candidate.content, "parts", []) or []:
                    if getattr(part, "inline_data", None):
                        data = part.inline_data.data
                        b64 = base64.b64encode(data).decode("utf-8") if isinstance(data, bytes) else data
                        mime = part.inline_data.mime_type or "image/png"
                        return {"success": True, "image_b64": b64, "mime_type": mime, "error": None}

            return {
                "success": False,
                "image_b64": None,
                "mime_type": None,
                "error": "No image data was returned by the Gemini image model.",
            }
        except Exception as e:
            return {
                "success": False,
                "image_b64": None,
                "mime_type": None,
                "error": _sanitize_error_message(str(e)),
            }


image_generator = GeminiImageGenerator(client)


def _message_text(msg) -> str:
    """Extract the plain text from a Streamlit Gemini content object or dict."""
    try:
        if hasattr(msg, "parts"):
            for part in msg.parts:
                if getattr(part, "text", None):
                    return part.text
        if isinstance(msg, dict):
            return msg.get("text", "") or msg.get("content", "") or ""
    except Exception:
        pass
    return ""


def classify_intent(prompt: str, has_last_image: bool = False) -> tuple[str, str | None, str | None]:
    """
    Classifies user prompt into one of:
      - ("text", text_prompt, None)
      - ("image", None, image_prompt)
      - ("combined", text_prompt, image_prompt)
    """
    text = re.sub(r"\s+", " ", prompt.strip())
    if not text:
        return ("text", prompt, None)

    lowered = text.lower()

    if any(q in lowered for q in ["what is image generation", "what is an image", "how to generate an image"]):
        return ("text", prompt, None)

    IMAGE_VERBS = r"(?:create|generate|draw|design|visualize|make|produce|render|paint|sketch|illustrate)"
    IMAGE_TARGETS = r"(?:image|picture|photo|poster|illustration|diagram|infographic|visual|artwork|graphic|drawing|sketch|wallpaper|cinematic|figure)"

    has_image_intent = False

    if re.search(rf"\b{IMAGE_VERBS}\b(?:\s+\w+){{0,5}}\s+\b(an?\s+)?{IMAGE_TARGETS}\b", lowered):
        has_image_intent = True
    elif re.search(rf"\bvisualize\b", lowered):
        has_image_intent = True
    elif re.search(rf"\bshow\s+(?:me\s+)?(?:an?\s+)?{IMAGE_TARGETS}\b", lowered):
        has_image_intent = True
    elif re.search(rf"\bdraw\b\s+(?:a\s+|an\s+|the\s+)?\w+", lowered) and not re.search(r"\bdraw\s+(?:conclusions?|a\s+blank)\b", lowered):
        has_image_intent = True
    elif has_last_image and any(s in lowered for s in ["make it", "add detail", "more realistic", "change the background", "refine", "make it brighter", "make it darker", "add solar"]):
        has_image_intent = True

    if not has_image_intent:
        return ("text", prompt, None)

    # Check if combined request (informational inquiry + visual creation)
    is_combined = False
    expl_start = re.match(r"^(?:explain|tell me about|what is|what are|describe|discuss|summarize|how does|why is)\b", lowered)
    if expl_start and re.search(rf"\b(?:and|also|along with|as well as|plus)\b.*?\b{IMAGE_VERBS}\b", lowered):
        is_combined = True
    elif re.search(r"[\.\?!]\s+", prompt):
        parts = re.split(r"[\.\?!]\s+", prompt)
        has_text_part = any(re.search(r"\b(explain|tell me about|what is|what are|describe|discuss|summarize|how does|why is)\b", p.lower()) for p in parts)
        if has_text_part and has_image_intent:
            is_combined = True

    if is_combined:
        return ("combined", prompt, prompt)
    return ("image", None, prompt)


def extract_image_subprompt(prompt: str) -> str:
    """For combined requests, extract or isolate the visual portion."""
    match = re.search(r"\b(?:and|also|along with|as well as|plus)\s+(.*)", prompt, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    parts = re.split(r"[\.\?!]\s+", prompt)
    if len(parts) > 1:
        IMAGE_VERBS = r"(?:create|generate|draw|design|visualize|make|produce|render|paint|sketch|illustrate)"
        for p in parts:
            if re.search(rf"\b{IMAGE_VERBS}\b", p, re.IGNORECASE):
                return p.strip()
    return prompt


def detect_image_intent(prompt: str, recent_messages: list | None = None) -> bool:
    """Classifier returning True if prompt requests an image (pure or combined)."""
    has_last_img = bool(st.session_state.get("last_generated_image"))
    intent, _, _ = classify_intent(prompt, has_last_image=has_last_img)
    return intent in {"image", "combined"}


def build_image_prompt(prompt: str, recent_messages: list | None = None) -> str:
    """Construct a clean, focused prompt from the current request and recent chat context."""
    history_text = []
    recent_limit = 5
    if recent_messages:
        for msg in list(reversed(recent_messages))[-recent_limit:]:
            text = _message_text(msg).strip()
            if text and len(text) > 3 and not text.lower().startswith("generated image"):
                history_text.append(text)

    history_text = list(reversed(history_text))
    prior_context = "; ".join(history_text[-3:]) if history_text else ""

    prev_prompt = ""
    if st.session_state.get("last_generated_image"):
        prev_prompt = str(st.session_state["last_generated_image"].get("prompt", "")).strip()

    user_request = prompt.strip()
    if prev_prompt and re.search(r"\b(make it|make the|add|update|refine|change|remake|redo|nighttime|more realistic|more cinematic|brighter|darker|solar|holographic)\b", user_request.lower()):
        base = f"Refine the previous image concept: {prev_prompt}. Apply the new instruction: {user_request}."
    else:
        base = f"Create a high-quality visual based on: {user_request}."

    sdg_context = f" Context: the user's active SDG context is {st.session_state.get('selectedSDG', 'General')} ."
    if prior_context:
        base += f" Relevant recent context: {prior_context}."
    if prev_prompt:
        base += f" Previous image prompt reference: {prev_prompt}."

    base += (
        " Keep the request faithful to the user's intent, preserve important subject matter, and produce a polished, "
        "clearly composed, visually compelling image. Use appropriate lighting, atmosphere, colors, and camera composition. "
        "Do not invent contradictory styles unless the user requests them."
    )

    if st.session_state.get("currentMode") in {"sdg", "general"}:
        base += sdg_context

    return base


def generate_image_with_gemini(prompt: str, previous_image: dict | None = None) -> dict:
    """Delegates to GeminiImageGenerator with gemini-3.1-flash-image-preview."""
    return image_generator.generate(prompt, previous_image=previous_image)


# 2. Geolocation & Reverse Geocoding Service
class ReverseGeocoder:
    """Abstract interface for reverse geocoding services."""
    def reverse_geocode(self, lat: float, lon: float) -> str:
        raise NotImplementedError


class NominatimReverseGeocoder(ReverseGeocoder):
    """Reverse geocoding provider using OpenStreetMap Nominatim with safety timeout."""
    def reverse_geocode(self, lat: float, lon: float) -> str:
        try:
            url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
            req = urllib.request.Request(url, headers={"User-Agent": "Brahmastra-SDG-Assistant/1.0"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                addr = data.get("address", {})
                parts = []
                city = addr.get("city") or addr.get("town") or addr.get("village") or addr.get("suburb") or addr.get("state_district")
                state = addr.get("state")
                country = addr.get("country")
                if city:
                    parts.append(city)
                if state and state != city:
                    parts.append(state)
                if country:
                    parts.append(country)
                return ", ".join(parts) if parts else f"{lat:.4f}, {lon:.4f}"
        except Exception:
            return f"Latitude: {lat:.4f}, Longitude: {lon:.4f}"


class GeolocationService:
    def __init__(self, geocoder: ReverseGeocoder):
        self.geocoder = geocoder

    def get_location_details(self, lat: float, lon: float, accuracy: float) -> dict:
        display = self.geocoder.reverse_geocode(lat, lon)
        return {
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "accuracy": round(accuracy, 1),
            "display": display,
        }


# Instantiate modular geolocation service
geo_service = GeolocationService(NominatimReverseGeocoder())


# ==================== ROYAL BLUE ATMOSPHERIC DESIGN SYSTEM ====================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    /* Multi-layered dark royal blue atmospheric background */
    .stApp {
        background-color: #030712 !important;
        background-image: 
            radial-gradient(circle at 50% 0%, rgba(29, 78, 216, 0.32) 0%, transparent 48%),
            radial-gradient(circle at 8% 28%, rgba(30, 58, 138, 0.25) 0%, transparent 42%),
            radial-gradient(circle at 92% 52%, rgba(37, 99, 235, 0.20) 0%, transparent 45%),
            radial-gradient(circle at 50% 90%, rgba(15, 23, 42, 0.85) 0%, #030712 75%) !important;
        background-attachment: fixed !important;
        color: #f1f5f9 !important;
    }

    /* Sidebar styled in deep midnight navy glass */
    section[data-testid="stSidebar"] {
        background: rgba(6, 12, 28, 0.85) !important;
        backdrop-filter: blur(16px) !important;
        border-right: 1px solid rgba(59, 130, 246, 0.22) !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(59, 130, 246, 0.2) !important;
    }

    /* Brahmastra Branding Header Area */
    .branding-header {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 12px 0 6px 0;
        margin-bottom: 4px;
        gap: 8px;
    }

    .branding-logo-row {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-wrap: wrap;
        gap: 18px;
        text-align: center;
    }

    .branding-img {
        display: block;
        width: clamp(52px, 7vw, 110px);
        max-width: 100%;
        height: auto;
        aspect-ratio: 1 / 1;
        object-fit: contain;
        border-radius: 20px;
        box-shadow: 
            0 0 25px rgba(245, 158, 11, 0.42),
            0 0 50px rgba(37, 99, 235, 0.35);
        border: 1px solid rgba(251, 191, 36, 0.4);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        flex-shrink: 0;
    }

    .branding-img:hover {
        transform: scale(1.03);
        box-shadow: 
            0 0 35px rgba(245, 158, 11, 0.6),
            0 0 70px rgba(37, 99, 235, 0.5);
    }

    .branding-title {
        font-size: clamp(1.8rem, 4vw, 2.6rem) !important;
        font-weight: 900 !important;
        letter-spacing: 5px !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.1 !important;
        background: linear-gradient(135deg, #ffffff 20%, #93c5fd 60%, #f59e0b 95%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        text-shadow: 0 0 30px rgba(37, 99, 235, 0.4);
    }

    .branding-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin: 0 0 16px 0;
        font-weight: 500;
    }

    @media (max-width: 640px) {
        .branding-logo-row {
            gap: 10px;
        }

        .branding-title {
            letter-spacing: 2px !important;
        }
    }

    /* Top Mode Buttons Styling */
    div[data-testid="stHorizontalBlock"] button {
        border-radius: 12px !important;
        padding: 10px 14px !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        transition: all 0.25s ease !important;
        backdrop-filter: blur(10px) !important;
    }

    div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
        background: rgba(12, 22, 50, 0.65) !important;
        border: 1px solid rgba(59, 130, 246, 0.25) !important;
        color: #cbd5e1 !important;
    }

    div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
        background: rgba(26, 45, 95, 0.85) !important;
        border-color: rgba(96, 165, 250, 0.6) !important;
        color: #ffffff !important;
        box-shadow: 0 0 16px rgba(37, 99, 235, 0.4) !important;
        transform: translateY(-2px);
    }

    div[data-testid="stHorizontalBlock"] button[kind="primary"] {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 60%, #3b82f6 100%) !important;
        border: 1px solid #93c5fd !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.6), 0 0 35px rgba(29, 78, 216, 0.35) !important;
        transform: translateY(-1px);
    }

    /* Context Ribbon Bar */
    .context-ribbon {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(12, 22, 50, 0.5);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(59, 130, 246, 0.22);
        border-radius: 12px;
        padding: 10px 20px;
        margin: 12px 0 22px 0;
        font-size: 0.9rem;
    }

    .ribbon-item {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .ribbon-label {
        color: #94a3b8;
    }

    .ribbon-value {
        color: #f1f5f9;
        font-weight: 600;
    }

    .ribbon-value.highlight {
        color: #93c5fd;
        background: rgba(37, 99, 235, 0.28);
        padding: 3px 12px;
        border-radius: 8px;
        border: 1px solid rgba(96, 165, 250, 0.35);
        box-shadow: 0 0 10px rgba(37, 99, 235, 0.25);
    }

    /* Chat Messages Glass Containers */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(10, 20, 48, 0.58) !important;
        backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(59, 130, 246, 0.22) !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3), 0 0 15px rgba(29, 78, 216, 0.12) !important;
        padding: 18px 22px !important;
        margin-bottom: 14px !important;
        transition: all 0.25s ease !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(96, 165, 250, 0.45) !important;
        box-shadow: 0 6px 26px rgba(0, 0, 0, 0.4), 0 0 22px rgba(37, 99, 235, 0.25) !important;
    }

    /* Premium Illuminated Input Box */
    [data-testid="stChatInput"] {
        background: transparent !important;
    }

    [data-testid="stChatInput"] > div {
        background: rgba(10, 20, 48, 0.88) !important;
        backdrop-filter: blur(18px) !important;
        border: 1px solid rgba(59, 130, 246, 0.35) !important;
        border-radius: 18px !important;
        box-shadow: 0 4px 25px rgba(0, 0, 0, 0.45), 0 0 18px rgba(29, 78, 216, 0.2) !important;
        transition: all 0.3s ease !important;
        padding: 4px 10px !important;
    }

    [data-testid="stChatInput"] > div:focus-within {
        border-color: rgba(96, 165, 250, 0.75) !important;
        box-shadow: 0 6px 30px rgba(0, 0, 0, 0.55), 0 0 28px rgba(37, 99, 235, 0.4) !important;
    }

    [data-testid="stChatInput"] textarea {
        color: #f8fafc !important;
        font-size: 0.98rem !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #64748b !important;
    }

    [data-testid="stChatInput"] button {
        color: #60a5fa !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stChatInput"] button:hover {
        color: #93c5fd !important;
        transform: scale(1.1) !important;
    }

    /* Image Generating Futuristic Animation */
    .image-generating-anim {
        text-align: center;
        padding: 30px 20px;
        background: rgba(10, 20, 48, 0.75);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(59, 130, 246, 0.35);
        border-radius: 20px;
        margin: 16px 0;
        box-shadow: 0 0 35px rgba(29, 78, 216, 0.35);
    }

    .pulsing-emblem {
        font-size: 2.8rem;
        animation: pulseGlow 1.8s infinite ease-in-out;
        margin-bottom: 12px;
    }

    @keyframes pulseGlow {
        0% { transform: scale(1); filter: drop-shadow(0 0 6px rgba(245, 158, 11, 0.5)); }
        50% { transform: scale(1.12); filter: drop-shadow(0 0 26px rgba(37, 99, 235, 0.9)); }
        100% { transform: scale(1); filter: drop-shadow(0 0 6px rgba(245, 158, 11, 0.5)); }
    }

    .generating-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #93c5fd;
        letter-spacing: 1px;
        margin-bottom: 6px;
        text-shadow: 0 0 16px rgba(37, 99, 235, 0.6);
    }

    .generating-desc {
        font-size: 0.88rem;
        color: #94a3b8;
        margin-bottom: 20px;
    }

    .quantum-loader {
        width: 250px;
        height: 6px;
        background: rgba(30, 58, 138, 0.4);
        border-radius: 10px;
        overflow: hidden;
        margin: 0 auto;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    .quantum-bar {
        width: 40%;
        height: 100%;
        background: linear-gradient(90deg, #1d4ed8, #60a5fa, #f59e0b);
        border-radius: 10px;
        animation: quantumSlide 1.5s infinite ease-in-out;
    }

    @keyframes quantumSlide {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(300%); }
    }

    /* Chat Generated Image Card */
    .chat-image-container {
        margin-top: 14px;
        margin-bottom: 8px;
        width: 100%;
        max-width: 680px;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }

    .chat-image-header {
        width: 100%;
        margin-bottom: 10px;
        padding: 8px 14px;
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 10px;
        backdrop-filter: blur(8px);
        box-sizing: border-box;
    }

    .chat-image-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #f59e0b;
        margin-bottom: 4px;
    }

    .chat-image-request {
        font-size: 0.95rem;
        color: #e2e8f0;
        font-weight: 500;
        line-height: 1.4;
        word-break: break-word;
    }

    .chat-image-frame {
        width: 100%;
        text-align: center;
    }

    .chat-image-wrapper {
        margin-top: 14px;
        text-align: center;
        width: 100%;
    }

    .chat-generated-image {
        max-width: 680px;
        width: 100%;
        height: auto;
        object-fit: contain;
        border-radius: 18px;
        border: 1px solid rgba(59, 130, 246, 0.45);
        box-shadow: 0 10px 32px rgba(0, 0, 0, 0.55), 0 0 28px rgba(37, 99, 235, 0.35);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeIn 0.45s ease;
        box-sizing: border-box;
    }

    .chat-generated-image:hover {
        transform: scale(1.012);
        box-shadow: 0 14px 40px rgba(0, 0, 0, 0.65), 0 0 38px rgba(245, 158, 11, 0.45);
    }

    .chat-image-caption {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 8px;
        font-style: italic;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 17 UN Sustainable Development Goals
SDG_LIST = [
    "SDG 1 — No Poverty",
    "SDG 2 — Zero Hunger",
    "SDG 3 — Good Health and Well-being",
    "SDG 4 — Quality Education",
    "SDG 5 — Gender Equality",
    "SDG 6 — Clean Water and Sanitation",
    "SDG 7 — Affordable and Clean Energy",
    "SDG 8 — Decent Work and Economic Growth",
    "SDG 9 — Industry, Innovation and Infrastructure",
    "SDG 10 — Reduced Inequalities",
    "SDG 11 — Sustainable Cities and Communities",
    "SDG 12 — Responsible Consumption and Production",
    "SDG 13 — Climate Action",
    "SDG 14 — Life Below Water",
    "SDG 15 — Life on Land",
    "SDG 16 — Peace, Justice and Strong Institutions",
    "SDG 17 — Partnerships for the Goals",
]

DEFAULT_SDG = "SDG 1 — No Poverty"

# Conversation Modes Configuration
MODE_CONFIG = [
    ("general", "🧠 General", "General-purpose AI assistant (default)"),
    ("sdg", "🌍 SDG Focus", "Primary focus on the selected SDG"),
    ("study", "📚 Study", "Structured step-by-step tutoring"),
    ("quiz", "📝 Quiz", "Interactive 3-question MCQ quiz"),
    ("research", "🔬 Research", "Academic research and methodology"),
    ("project", "💡 Project", "Practical student project development"),
]
MODE_KEYS = [m[0] for m in MODE_CONFIG]
MODE_LABELS = {m[0]: m[1] for m in MODE_CONFIG}

# SDG Metadata for SDG Focus Mode detection
SDG_METADATA = {
    1: {"name": "No Poverty", "keywords": ["no poverty", "poverty"]},
    2: {"name": "Zero Hunger", "keywords": ["zero hunger", "hunger"]},
    3: {"name": "Good Health and Well-being", "keywords": ["good health", "well-being", "wellbeing"]},
    4: {"name": "Quality Education", "keywords": ["quality education", "education"]},
    5: {"name": "Gender Equality", "keywords": ["gender equality"]},
    6: {"name": "Clean Water and Sanitation", "keywords": ["clean water", "sanitation"]},
    7: {"name": "Affordable and Clean Energy", "keywords": ["affordable and clean energy", "clean energy", "renewable energy"]},
    8: {"name": "Decent Work and Economic Growth", "keywords": ["decent work", "economic growth"]},
    9: {"name": "Industry, Innovation and Infrastructure", "keywords": ["industry, innovation and infrastructure", "infrastructure"]},
    10: {"name": "Reduced Inequalities", "keywords": ["reduced inequalities", "inequality", "inequalities"]},
    11: {"name": "Sustainable Cities and Communities", "keywords": ["sustainable cities", "cities and communities"]},
    12: {"name": "Responsible Consumption and Production", "keywords": ["responsible consumption", "consumption and production"]},
    13: {"name": "Climate Action", "keywords": ["climate action", "climate change"]},
    14: {"name": "Life Below Water", "keywords": ["life below water", "marine life", "oceans"]},
    15: {"name": "Life on Land", "keywords": ["life on land", "biodiversity on land", "terrestrial ecosystems"]},
    16: {"name": "Peace, Justice and Strong Institutions", "keywords": ["peace, justice and strong institutions", "peace and justice", "strong institutions"]},
    17: {"name": "Partnerships for the Goals", "keywords": ["partnerships for the goals", "partnerships"]},
}

# Exact required refusal for adult content
SAFETY_REFUSAL_MESSAGE = "I am not allowed to talk about this topic"

# Initialize Session State
if "selectedSDG" not in st.session_state:
    st.session_state.selectedSDG = DEFAULT_SDG

if "selected_sdg" not in st.session_state:
    st.session_state.selected_sdg = st.session_state.selectedSDG

if "currentMode" not in st.session_state:
    st.session_state.currentMode = "general"

if "conversationContext" not in st.session_state:
    st.session_state.conversationContext = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

if "message_attachments" not in st.session_state:
    st.session_state.message_attachments = {}

if "input_mode" not in st.session_state:
    st.session_state.input_mode = "chat"

if "user_location" not in st.session_state:
    st.session_state.user_location = None

if "geo_error" not in st.session_state:
    st.session_state.geo_error = None

if "request_location" not in st.session_state:
    st.session_state.request_location = False

if "last_generated_image" not in st.session_state:
    st.session_state.last_generated_image = None

if "sdg_conversations" not in st.session_state:
    st.session_state.sdg_conversations = {sdg: [] for sdg in SDG_LIST}

if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False

if "confirm_reset" not in st.session_state:
    st.session_state.confirm_reset = False

# Process browser Geolocation Query Params returned via navigator.geolocation
geo_status = st.query_params.get("geo_status")
if geo_status:
    if geo_status == "success":
        try:
            lat = float(st.query_params.get("geo_lat", 0))
            lon = float(st.query_params.get("geo_lon", 0))
            acc = float(st.query_params.get("geo_acc", 0))
            st.session_state.user_location = geo_service.get_location_details(lat, lon, acc)
            st.session_state.geo_error = None
        except Exception as e:
            st.session_state.geo_error = f"Error processing coordinates: {e}"
    elif geo_status == "denied":
        st.session_state.geo_error = "Location permission was denied. You can enable it from your browser's site permissions."
    elif geo_status == "unavailable":
        st.session_state.geo_error = "Your location is currently unavailable. Please try again."
    elif geo_status == "timeout":
        st.session_state.geo_error = "Location request timed out. Please try again."
    elif geo_status == "unsupported":
        st.session_state.geo_error = "Geolocation isn't supported by this browser."

    st.session_state.request_location = False
    st.query_params.clear()


def check_18_plus_content(user_text: str) -> bool:
    """Identifies clearly adult/18+ content without falsely triggering on educational/academic topics."""
    text = user_text.lower().strip()

    educational_safe_patterns = [
        r"\bsex\s+ratio\b",
        r"\bsex\s+education\b",
        r"\bbiological\s+sex\b",
        r"\breproductive\s+health\b",
        r"\breproductive\s+rights\b",
        r"\bhuman\s+reproduction\b",
        r"\bplant\s+reproduction\b",
        r"\banimal\s+reproduction\b",
        r"\bmaternal\s+health\b",
        r"\bhuman\s+development\b",
        r"\bwater\s+body\b",
        r"\bgoverning\s+body\b",
        r"\bstudent\s+body\b",
        r"\bhealthy\s+relationships?\b",
        r"\binterpersonal\s+relationships?\b",
        r"\binternational\s+relationships?\b",
        r"\bcell(ular)?\s+biology\b",
        r"\bmarine\s+biology\b",
        r"\bmolecular\s+biology\b",
        r"\bhuman\s+anatomy\b",
        r"\bstd\s+prevention\b",
        r"\bsexually\s+transmitted\b",
        r"\bfamily\s+planning\b",
    ]

    adult_patterns = [
        r"\bporn(o|ography)?\b",
        r"\bxxx\b",
        r"\bhentai\b",
        r"\berotic(a)?\b",
        r"\bnsfw\b",
        r"\bnude(s)?\b",
        r"\bnaked\b",
        r"\bcamgirl\b",
        r"\bonlyfans\b",
        r"\bstripper(s)?\b",
        r"\bblowjob(s)?\b",
        r"\bhandjob(s)?\b",
        r"\bdildo(s)?\b",
        r"\bvibrator(s)?\b",
        r"\bmasturbat(e|ing|ion)\b",
        r"\borgasm(s)?\b",
        r"\bhorny\b",
        r"\bpenis\b",
        r"\bvagina\b",
        r"\bboobs?\b",
        r"\btits?\b",
        r"\bpussy\b",
        r"\bdick\b",
        r"\bcock\b",
        r"\bcunnilingus\b",
        r"\bfellatio\b",
        r"\bbdsm\b",
        r"\bfetish(es)?\b",
        r"\bgangbang\b",
        r"\bincest\b",
        r"\bmilf\b",
        r"\bdeepthroat\b",
        r"\bthreesome\b",
        r"\bsex\s+video\b",
        r"\bsex\s+movie\b",
        r"\bsex\s+scene\b",
        r"\bsex\s+positions?\b",
        r"\bkama\s+sutra\b",
        r"\berotic\s+stor(y|ies)\b",
        r"\bcybersex\b",
        r"\bsext(ing)?\b",
        r"\bhaving\s+sex\b",
        r"\bmake\s+love\b",
        r"\bhooker(s)?\b",
        r"\bprostitut(e|ion|es)\b",
        r"\b18\+\s*(content|video|pic|image|story|stuff)\b",
        r"\badult\s*(content|video|film|movie|game|site|story)\b",
        r"\bdirty\s+story\b",
        r"\bsexual\s+acts?\b",
    ]

    for pattern in adult_patterns:
        if re.search(pattern, text):
            is_edu = False
            for edu_pat in educational_safe_patterns:
                m_edu = re.search(edu_pat, text)
                if m_edu:
                    m_adult = re.search(pattern, text)
                    if m_adult and m_edu.start() <= m_adult.start() and m_edu.end() >= m_adult.end():
                        is_edu = True
                        break
            if not is_edu:
                return True

    return False


def get_sdg_num(sdg_str: str) -> int:
    m = re.search(r"SDG\s*(\d+)", sdg_str, re.IGNORECASE)
    return int(m.group(1)) if m else 1


def detect_other_sdg_request(prompt: str, current_sdg: str) -> bool:
    """When currentMode is 'sdg', checks if the user is explicitly asking about a different SDG."""
    text = prompt.lower().strip()
    current_num = get_sdg_num(current_sdg)

    sdg_num_matches = re.findall(r"\b(?:sdg|goal)\s*(\d{1,2})\b", text)
    if sdg_num_matches:
        for num_str in sdg_num_matches:
            num = int(num_str)
            if 1 <= num <= 17 and num != current_num:
                return True
        if any(int(n) == current_num for n in sdg_num_matches):
            return False

    current_keywords = SDG_METADATA.get(current_num, {}).get("keywords", [])
    for kw in current_keywords:
        if kw in text:
            return False

    query_intent_verbs = r"(?:what is|explain|tell me about|describe|discuss|how about|switch to|look at|info on|teach me about)\s+"
    for num, meta in SDG_METADATA.items():
        if num != current_num:
            for kw in meta["keywords"]:
                pattern = query_intent_verbs + re.escape(kw)
                if re.search(pattern, text):
                    return True
                if text == kw or text == f"sdg {num}" or text == f"goal {num}":
                    return True

    return False


def is_location_query(text: str) -> bool:
    """Checks if the user prompt is asking about their location or accuracy."""
    t = text.lower().strip()
    patterns = [
        r"\bwhere am i\b",
        r"\bwhat is my (current )?location\b",
        r"\bwhat's my (current )?location\b",
        r"\bmy (current )?location\b",
        r"\bwhere i am\b",
        r"\bhow accurate is that\b",
        r"\bhow accurate is my location\b",
        r"\bwhat is my city\b",
    ]
    return any(re.search(p, t) for p in patterns)


def assemble_system_prompt(sdg: str, current_mode: str, user_location: dict = None) -> str:
    """Dynamically assembles the master system instruction for Brahmastra based on currentMode and location."""
    sdg_title = sdg.split("—")[-1].strip() if "—" in sdg else sdg

    # Location context section
    if user_location:
        location_section = f"""
======================================================================
USER LOCATION CONTEXT (OBTAINED WITH EXPLICIT BROWSER PERMISSION):
======================================================================
The user has granted permission to access their browser geographic location.
- Approximate Location: {user_location['display']}
- Coordinates: Latitude {user_location['lat']}, Longitude {user_location['lon']}
- Accuracy: approximately {user_location['accuracy']} meters

RULES FOR LOCATION:
1. If the user asks "Where am I?", "What's my location?", or about their current location, reply:
   "Your approximate location is {user_location['display']}."
2. If the user asks "How accurate is that?", reply:
   "The browser reports an accuracy of approximately {user_location['accuracy']} meters."
3. Never fabricate or extrapolate coordinates beyond what the browser provided.
"""
    else:
        location_section = """
======================================================================
USER LOCATION STATUS: NOT GRANTED
======================================================================
The user has NOT granted location access.
RULES FOR LOCATION:
1. If the user asks "Where am I?", "What's my location?", or requests their location, reply:
   "Location access hasn't been granted. Would you like to allow location access? You can click the 📍 Location button to grant access."
2. NEVER pretend to know the user's location when permission has not been granted.
"""

    base_prompt = f"""You are Brahmastra, an educational AI chatbot designed to help students and learners understand, study, research, and develop projects related to the United Nations Sustainable Development Goals (SDGs).

CURRENT PREFERRED / SELECTED SDG:
{sdg}

CURRENT CONVERSATION MODE:
{current_mode}

{location_section}

======================================================================
CRITICAL PRIORITY 1: 18+ / ADULT CONTENT SAFETY FILTER (HIGHEST PRIORITY)
======================================================================
If the user asks about clearly adult, sexually explicit, pornographic, or 18+ content:
You MUST respond with EXACTLY and ONLY the following string:
I am not allowed to talk about this topic
Do NOT add any explanation, greeting, apology, alternative suggestion, link, or extra words.
Do NOT trigger this filter for legitimate academic or educational topics (such as reproductive health under SDG 3, census sex ratios, human biology, anatomy, human development). Only trigger for clearly adult/18+ content.

======================================================================
ROLE & DEFAULT PERSONALITY:
======================================================================
Brahmastra is a friendly SDG tutor, educational assistant, research assistant, project assistant, and quiz instructor.
Help students understand concepts rather than simply providing raw answers.
Be friendly, patient, clear, encouraging, educational, practical, neutral, and student-friendly.
Use simple English. When technical terms are needed, explain them simply.

ACCURACY RULE:
Never knowingly fabricate statistics, research papers, authors, organizations, government programs, scientific findings, case studies, SDG targets, dates, citations, or sources.
If uncertain, say: "I'm not sure about that."
"""

    if current_mode == "general":
        mode_instructions = f"""
======================================================================
CURRENT MODE: GENERAL (GENERAL-PURPOSE AI ASSISTANT)
======================================================================
Brahmastra behaves as a normal, helpful, general-purpose AI assistant.
- The selected SDG ({sdg}) remains stored separately but is only optional background context.
- Do NOT force the selected SDG into unrelated questions.
- Answer questions on any educational topic, science, technology, sustainability, or any of the 17 SDGs freely and thoroughly.
- For normal educational questions, structure the answer clearly:
  ### Answer
  Clear explanation addressing the concept.
  ### Example
  One practical or real-world example when appropriate.
  ### Why it matters
  Explain why the topic matters to development, society, or everyday life.
  End with a short check question when appropriate.
"""

    elif current_mode == "sdg":
        mode_instructions = f"""
======================================================================
CURRENT MODE: SDG FOCUS (PRIMARY FOCUS ON {sdg})
======================================================================
The currently selected SDG ({sdg}) is your primary conversational focus.
RULES:
1. SAME-SDG QUESTIONS:
   Questions directly or indirectly related to {sdg} (concepts, definitions, targets, challenges, solutions, examples) must ALWAYS be answered normally and thoroughly.
   CRITICAL: NEVER say "I am locked", "The system has locked me", or mention technical restrictions.
2. GENERAL & TECHNOLOGY QUESTIONS:
   If the user asks general educational, scientific, or technology questions (e.g. "How can machine learning help?", "What is IoT?", "Give me a project idea"), interpret the question in the context of {sdg} and explain how it applies to {sdg_title}.
3. QUESTIONS ABOUT ANOTHER SDG:
   When the user explicitly asks about a different SDG, naturally redirect the conversation toward {sdg}.
   Never say "I am locked." Never mention technical restrictions.
   Respond naturally with:
   "I'd be happy to help with the current focus: {sdg}. Ask me anything about it!"
"""

    elif current_mode == "study":
        mode_instructions = f"""
======================================================================
CURRENT MODE: STUDY (STRUCTURED TUTORING)
======================================================================
You are operating in Study Mode. Behave as a dedicated personal tutor.
Explain concepts step-by-step, starting from foundations.
You MUST format your tutoring response using the following structured flow:
### Concept
Identify and name the specific concept clearly.
### Simple Explanation
Explain the concept in simple, accessible, student-friendly terms.
### Example
Provide an illustrative, practical, or real-world example.
### Key Point
State the essential core takeaway in 1-2 memorable sentences.
### Check Question
Provide a thoughtful check question to verify the student's understanding.
"""

    elif current_mode == "quiz":
        mode_instructions = f"""
======================================================================
CURRENT MODE: QUIZ (INTERACTIVE 3-QUESTION QUIZ)
======================================================================
You are operating in Quiz Mode. Run an interactive quiz interaction.
Rules:
- Ask exactly 3 multiple-choice questions (MCQs) in total.
- Present ONE question at a time.
- Each question must contain exactly 4 options labeled A, B, C, and D.
- Wait for the user's answer before presenting the next question.
- After the user answers:
  - State whether the answer is correct or incorrect.
  - Briefly explain the reason.
  - Present the next question if fewer than 3 questions have been asked.
  - After the 3rd question is answered, state the final score out of 3 (e.g. "Final Score: 3/3") with encouraging feedback.
- If the selected SDG ({sdg}) is relevant to the user's request, use it as context.
"""

    elif current_mode == "research":
        mode_instructions = f"""
======================================================================
CURRENT MODE: RESEARCH (ACADEMIC RESEARCH ASSISTANT)
======================================================================
You are operating in Research Mode. Assist students and scholars with rigorous academic research.
Provide comprehensive and structured assistance with:
- Research topics and domains
- Research questions and problem statements
- Research objectives and scope
- Formulating testable hypotheses
- Literature review themes, theoretical frameworks, and search strategies
- Research methodology (qualitative, quantitative, mixed-methods)
- Independent, dependent, and control variables
- Surveys, interviews, questionnaires, and data collection protocols
- Case studies and comparative analysis
- Identifying research gaps and limitations
- Future research directions
- Structuring research papers, reports, and presentations
CRITICAL RULE: Never fabricate citations, authors, journals, dates, or data. If specific literature is requested, suggest verified databases (Google Scholar, UN iLibrary, IEEE) and appropriate search queries.
"""

    elif current_mode == "project":
        mode_instructions = f"""
======================================================================
CURRENT MODE: PROJECT (PRACTICAL STUDENT PROJECT DESIGN)
======================================================================
You are operating in Project Mode. Help students design, develop, and structure practical, realistic technology and educational projects.
Provide detailed, structured assistance covering:
- Practical project ideas and concepts
- Problem statements and targeted challenges
- Project objectives and success criteria
- Core features and capabilities
- Recommended technologies, frameworks, and tools
- High-level system architecture and data flows
- Step-by-step implementation plan
- AI/ML integration (models, libraries, pipelines)
- Data requirements and suggested open data sources
- Evaluation metrics and testing strategy
- Expected real-world outcomes and impact
- Future improvements and scalability
"""

    else:
        mode_instructions = ""

    return base_prompt + "\n\n" + mode_instructions


# ==================== SIDEBAR ====================
with st.sidebar:
    if lotus_b64:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                <img src="data:image/png;base64,{lotus_b64}" style="width: 44px; height: 44px; border-radius: 12px; object-fit: contain; box-shadow: 0 0 16px rgba(245, 158, 11, 0.45); border: 1px solid rgba(251, 191, 36, 0.4);" alt="Golden Lotus" />
                <div>
                    <h3 style="margin: 0; padding: 0; color: #f8fafc; font-size: 1.2rem; font-weight: 800; letter-spacing: 1.5px;">BRAHMĀSTRA</h3>
                    <p style="margin: 0; padding: 0; color: #94a3b8; font-size: 0.72rem; letter-spacing: 0.5px;">UN SDG Educational Assistant</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown("## 🏹 **BRAHMĀSTRA**")
        st.caption("🌍 **UN SDG Educational Assistant**")
    st.markdown("---")

    # 🌍 SDG Selector (Determines which SDG is selected)
    st.markdown("### 🌍 SDG Selector")
    current_index = SDG_LIST.index(st.session_state.selectedSDG) if st.session_state.selectedSDG in SDG_LIST else 0
    chosen_sdg = st.selectbox(
        "Choose Preferred Goal:",
        options=SDG_LIST,
        index=current_index,
        help="Select any of the 17 UN Sustainable Development Goals.",
    )

    if chosen_sdg != st.session_state.selectedSDG:
        st.session_state.selectedSDG = chosen_sdg
        st.session_state.selected_sdg = chosen_sdg
        st.rerun()

    st.markdown("---")

    # Quick summary of active state in sidebar
    active_label = MODE_LABELS.get(st.session_state.currentMode, "🧠 General")
    st.markdown(f"**Selected SDG:**  \n`{st.session_state.selectedSDG}`")
    st.markdown(f"**Current Mode:**  \n`{active_label}`")

    if st.session_state.user_location:
        st.markdown(f"**📍 Location:**  \n`{st.session_state.user_location['display']}`")

    st.markdown("---")

    # Action Buttons: Clear Conversation & Reset Settings
    col_clear, col_reset = st.columns(2)

    with col_clear:
        if st.button("🗑 Clear Chat", use_container_width=True, help="Clear conversation history"):
            st.session_state.confirm_clear = True
            st.session_state.confirm_reset = False

    with col_reset:
        if st.button("⚙ Reset", use_container_width=True, help="Reset SDG and conversation mode to defaults"):
            st.session_state.confirm_reset = True
            st.session_state.confirm_clear = False

    # Confirmation Prompts
    if st.session_state.confirm_clear:
        st.warning("Clear the current conversation?")
        c1, c2 = st.columns(2)
        if c1.button("Yes, Clear", type="primary", use_container_width=True):
            st.session_state.messages = []
            st.session_state.message_attachments = {}
            st.session_state.last_generated_image = None
            st.session_state.sdg_conversations = {sdg: [] for sdg in SDG_LIST}
            st.session_state.confirm_clear = False
            st.rerun()
        if c2.button("Cancel", key="cancel_clear", use_container_width=True):
            st.session_state.confirm_clear = False
            st.rerun()

    if st.session_state.confirm_reset:
        st.warning("Reset SDG and conversation mode to default?")
        r1, r2 = st.columns(2)
        if r1.button("Yes, Reset", type="primary", use_container_width=True):
            st.session_state.selectedSDG = DEFAULT_SDG
            st.session_state.selected_sdg = DEFAULT_SDG
            st.session_state.currentMode = "general"
            st.session_state.conversationContext = ""
            st.session_state.user_location = None
            st.session_state.geo_error = None
            st.session_state.input_mode = "chat"
            st.session_state.last_generated_image = None
            st.session_state.confirm_reset = False
            st.rerun()
        if r2.button("Cancel", key="cancel_reset", use_container_width=True):
            st.session_state.confirm_reset = False
            st.rerun()


# ==================== MAIN INTERFACE ====================

# 1. BRAHMĀSTRA BRANDING HEADER (GOLDEN LOTUS AS HEADING)
if lotus_b64:
    brand_img_tag = f'<img src="data:image/png;base64,{lotus_b64}" class="branding-img" alt="Golden Lotus — Brahmāstra Logo" />'
elif brand_b64:
    brand_img_tag = f'<img src="data:image/jpeg;base64,{brand_b64}" class="branding-img" alt="Brahmāstra Logo" />'
else:
    brand_img_tag = f'<img src="{BRAND_IMAGE_URL}" class="branding-img" alt="Brahmāstra Logo" />'

st.markdown(
    f"""
    <div class="branding-header">
        <div class="branding-logo-row">
            {brand_img_tag}
            <h1 class="branding-title">BRAHMĀSTRA</h1>
        </div>
        <p class="branding-subtitle">UN Sustainable Development Goals • AI Educational Assistant</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 2. TOP CONVERSATION MODE SELECTOR
mode_cols = st.columns(len(MODE_CONFIG))
for i, (mode_key, mode_label, mode_desc) in enumerate(MODE_CONFIG):
    is_active = (st.session_state.currentMode == mode_key)
    btn_type = "primary" if is_active else "secondary"
    with mode_cols[i]:
        if st.button(
            mode_label,
            key=f"top_mode_{mode_key}",
            type=btn_type,
            use_container_width=True,
            help=mode_desc,
        ):
            if st.session_state.currentMode != mode_key:
                st.session_state.currentMode = mode_key
                st.rerun()

# 3. SLEEK CONTEXT RIBBON & LOCATION STATUS
active_label = MODE_LABELS.get(st.session_state.currentMode, "🧠 General")
loc_display_str = st.session_state.user_location["display"] if st.session_state.user_location else "Not granted"

st.markdown(
    f"""
    <div class="context-ribbon">
        <div class="ribbon-item">
            <span class="ribbon-icon">🌍</span>
            <span class="ribbon-label">Selected SDG:</span>
            <span class="ribbon-value">{st.session_state.selectedSDG}</span>
        </div>
        <div class="ribbon-item">
            <span class="ribbon-icon">🧭</span>
            <span class="ribbon-label">Current Mode:</span>
            <span class="ribbon-value highlight">{active_label}</span>
        </div>
        <div class="ribbon-item">
            <span class="ribbon-icon">📍</span>
            <span class="ribbon-label">Location:</span>
            <span class="ribbon-value">{'🌐 ' + loc_display_str if st.session_state.user_location else '⚪ Not granted'}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Geolocation error notification if any
if st.session_state.geo_error:
    st.info(f"📍 {st.session_state.geo_error}")

# Client-side Geolocation API invocation component when requested
if st.session_state.request_location:
    components.html(
        """
        <script>
        if (!navigator.geolocation) {
            const p = new URLSearchParams(window.parent.location.search);
            p.set('geo_status', 'unsupported');
            window.parent.location.search = p.toString();
        } else {
            navigator.geolocation.getCurrentPosition(
                function(pos) {
                    const p = new URLSearchParams(window.parent.location.search);
                    p.set('geo_lat', pos.coords.latitude);
                    p.set('geo_lon', pos.coords.longitude);
                    p.set('geo_acc', Math.round(pos.coords.accuracy));
                    p.set('geo_status', 'success');
                    window.parent.location.search = p.toString();
                },
                function(err) {
                    let s = 'error';
                    if (err.code === 1) s = 'denied';
                    else if (err.code === 2) s = 'unavailable';
                    else if (err.code === 3) s = 'timeout';
                    const p = new URLSearchParams(window.parent.location.search);
                    p.set('geo_status', s);
                    window.parent.location.search = p.toString();
                },
                { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 }
            );
        }
        </script>
        """,
        height=0,
    )

# 4. WELCOME MESSAGE IF NO MESSAGES YET
if not st.session_state.messages:
    welcome_text = (
        f"👋 **Greetings! I am Brahmāstra.**\n\n"
        f"I am your dedicated educational AI companion for the **United Nations Sustainable Development Goals**.\n\n"
        f"🌍 **Current Focus Goal:** **{st.session_state.selectedSDG}**\n\n"
        f"🧭 **Selected Mode:** **{active_label}**\n\n"
        f"✨ You can ask any question, explore SDG research, or simply ask me to **create an image, draw a diagram, or make an infographic** — I will automatically generate it directly in our conversation."
    )
    with st.container(border=True):
        st.markdown(f"**🏹 Brahmāstra**\n\n{welcome_text}")

# 5. REPLAY CONVERSATION HISTORY (WITH GENERATED IMAGES)
for idx, msg in enumerate(st.session_state.messages):
    with st.container(border=True):
        if msg.role == "user":
            st.markdown(f"**🧑 You**\n\n{msg.parts[0].text}")
        else:
            st.markdown(f"**🏹 Brahmāstra**\n\n{msg.parts[0].text}")
            # Render attached generated image if present with prompt above image
            if idx in st.session_state.message_attachments:
                att = st.session_state.message_attachments[idx]
                if att.get("type") == "image":
                    st.markdown(
                        f"""
                        <div class="chat-image-container">
                            <div class="chat-image-header">
                                <span class="chat-image-badge">🎨 Original Request</span>
                                <div class="chat-image-request">"{att['prompt']}"</div>
                            </div>
                            <div class="chat-image-frame">
                                <img src="data:{att['mime']};base64,{att['b64']}" 
                                     class="chat-generated-image" 
                                     alt="{att['prompt']}" />
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ==================== UNIFIED CONVERSATIONAL INTERFACE ====================
prompt = st.chat_input("Ask Brahmāstra anything or describe an image to create...")
if prompt:
    user_content = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    st.session_state.messages.append(user_content)
    with st.container(border=True):
        st.markdown(f"**🧑 You**\n\n{prompt}")

    recent_messages = st.session_state.messages[-8:]

    if check_18_plus_content(prompt):
        safety_reply = SAFETY_REFUSAL_MESSAGE
        with st.container(border=True):
            st.markdown(f"**🏹 Brahmāstra**\n\n{safety_reply}")
        model_content = types.Content(role="model", parts=[types.Part.from_text(text=safety_reply)])
        st.session_state.messages.append(model_content)

    elif is_location_query(prompt):
        if st.session_state.user_location:
            if "accurate" in prompt.lower():
                loc_reply = f"The browser reports an accuracy of approximately {st.session_state.user_location['accuracy']} meters."
            else:
                loc_reply = f"Your approximate location is {st.session_state.user_location['display']}."
        else:
            loc_reply = "Location access hasn't been granted. Brahmāstra is ready to assist you with any SDG or sustainability inquiry worldwide or for any region you specify."

        with st.container(border=True):
            st.markdown(f"**🏹 Brahmāstra**\n\n{loc_reply}")
        model_content = types.Content(role="model", parts=[types.Part.from_text(text=loc_reply)])
        st.session_state.messages.append(model_content)

    elif st.session_state.currentMode == "sdg" and detect_other_sdg_request(prompt, st.session_state.selectedSDG):
        redirect_reply = f"I'd be happy to help with the current focus: {st.session_state.selectedSDG}. Ask me anything about it!"
        with st.container(border=True):
            st.markdown(f"**🏹 Brahmāstra**\n\n{redirect_reply}")
        model_content = types.Content(role="model", parts=[types.Part.from_text(text=redirect_reply)])
        st.session_state.messages.append(model_content)

    else:
        has_last_img = bool(st.session_state.get("last_generated_image"))
        intent_type, text_req, img_req = classify_intent(prompt, has_last_image=has_last_img)

        # ----------------------------------------------------
        # CASE 1: COMBINED REQUEST (TEXT EXPLANATION + IMAGE)
        # ----------------------------------------------------
        if intent_type == "combined":
            system_instruction = assemble_system_prompt(
                sdg=st.session_state.selectedSDG,
                current_mode=st.session_state.currentMode,
                user_location=st.session_state.user_location,
            )

            with st.spinner("Brahmāstra is generating your explanation & visual..."):
                try:
                    config = types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.3,
                    )
                    combined_text_prompt = (
                        f"The user requested: '{prompt}'. "
                        "Provide a concise, informative, high-quality response addressing the conceptual and informational part of their request. "
                        "Do not include instructions about how to generate an image because Gemini image generation is automatically rendering the visual."
                    )
                    response = client.models.generate_content(
                        model="gemini-3.5-flash-lite",
                        contents=combined_text_prompt,
                        config=config,
                    )
                    explanation_text = response.text or "Here is the explanation for your request."
                except Exception as e:
                    explanation_text = "Here is the explanation for your request."

                sub_prompt = extract_image_subprompt(prompt)
                image_prompt = build_image_prompt(sub_prompt, recent_messages)
                previous_image = st.session_state.get("last_generated_image")
                result = image_generator.generate(image_prompt, previous_image=previous_image)

            model_content = types.Content(role="model", parts=[types.Part.from_text(text=explanation_text)])
            st.session_state.messages.append(model_content)
            assistant_idx = len(st.session_state.messages) - 1

            if result.get("success"):
                img_b64 = result["image_b64"]
                mime_type = result.get("mime_type", "image/png")
                st.session_state.message_attachments[assistant_idx] = {
                    "type": "image",
                    "b64": img_b64,
                    "mime": mime_type,
                    "prompt": prompt,
                }
                st.session_state.last_generated_image = {
                    "b64": img_b64,
                    "mime": mime_type,
                    "prompt": prompt,
                }

                with st.container(border=True):
                    st.markdown(f"**🏹 Brahmāstra**\n\n{explanation_text}")
                    st.markdown(
                        f"""
                        <div class="chat-image-container">
                            <div class="chat-image-header">
                                <span class="chat-image-badge">🎨 Original Request</span>
                                <div class="chat-image-request">"{prompt}"</div>
                            </div>
                            <div class="chat-image-frame">
                                <img src="data:{mime_type};base64,{img_b64}" class="chat-generated-image" alt="{prompt}" />
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                err_text = result.get("error", "Image generation could not be completed.")
                with st.container(border=True):
                    st.markdown(f"**🏹 Brahmāstra**\n\n{explanation_text}")
                    st.warning(f"⚠️ {err_text}")

        # ----------------------------------------------------
        # CASE 2: DIRECT IMAGE GENERATION (GEMINI 3.1 FLASH IMAGE)
        # ----------------------------------------------------
        elif intent_type == "image":
            image_prompt = build_image_prompt(prompt, recent_messages)
            with st.container(border=True):
                st.markdown("**🏹 Brahmāstra**\n\n✦ Brahmāstra is visualizing...")
                anim_placeholder = st.empty()
                anim_placeholder.markdown(
                    """
                    <div class="image-generating-anim">
                        <div class="pulsing-emblem">🏹</div>
                        <div class="generating-title">BRAHMĀSTRA is creating your image...</div>
                        <div class="generating-desc">Synthesizing visual concepts with Gemini 3.1 Flash Image Preview</div>
                        <div class="quantum-loader"><div class="quantum-bar"></div></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            previous_image = st.session_state.get("last_generated_image")
            result = image_generator.generate(image_prompt, previous_image=previous_image)
            anim_placeholder.empty()

            if result.get("success"):
                img_b64 = result["image_b64"]
                mime_type = result.get("mime_type", "image/png")
                model_text = "✨ Visual generated directly for your request."
                model_content = types.Content(role="model", parts=[types.Part.from_text(text=model_text)])
                st.session_state.messages.append(model_content)

                assistant_idx = len(st.session_state.messages) - 1
                st.session_state.message_attachments[assistant_idx] = {
                    "type": "image",
                    "b64": img_b64,
                    "mime": mime_type,
                    "prompt": prompt,
                }
                st.session_state.last_generated_image = {
                    "b64": img_b64,
                    "mime": mime_type,
                    "prompt": prompt,
                }

                with st.container(border=True):
                    st.markdown(f"**🏹 Brahmāstra**\n\n{model_text}")
                    st.markdown(
                        f"""
                        <div class="chat-image-container">
                            <div class="chat-image-header">
                                <span class="chat-image-badge">🎨 Original Request</span>
                                <div class="chat-image-request">"{prompt}"</div>
                            </div>
                            <div class="chat-image-frame">
                                <img src="data:{mime_type};base64,{img_b64}" class="chat-generated-image" alt="{prompt}" />
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                err_text = result.get("error", "Image generation could not be completed.")
                fallback_msg = f"⚠️ {err_text} You can try another image prompt or ask any SDG question using the chat input."
                st.warning(fallback_msg)
                model_content = types.Content(role="model", parts=[types.Part.from_text(text=fallback_msg)])
                st.session_state.messages.append(model_content)

        # ----------------------------------------------------
        # CASE 3: STANDARD TEXT CONVERSATION
        # ----------------------------------------------------
        else:
            system_instruction = assemble_system_prompt(
                sdg=st.session_state.selectedSDG,
                current_mode=st.session_state.currentMode,
                user_location=st.session_state.user_location,
            )

            with st.spinner("Brahmāstra is thinking..."):
                try:
                    config = types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.3,
                    )
                    response = client.models.generate_content(
                        model="gemini-3.5-flash-lite",
                        contents=st.session_state.messages,
                        config=config,
                    )
                    response_text = response.text or "I'm not sure about that."

                    if "i am not allowed to talk about this topic" in response_text.lower():
                        response_text = SAFETY_REFUSAL_MESSAGE

                    with st.container(border=True):
                        st.markdown(f"**🏹 Brahmāstra**\n\n{response_text}")

                    model_content = types.Content(role="model", parts=[types.Part.from_text(text=response_text)])
                    st.session_state.messages.append(model_content)
                except Exception as e:
                    clean_err = _sanitize_error_message(str(e))
                    error_msg = f"⚠️ An error occurred while communicating with Gemini: {clean_err}"
                    st.error(error_msg)
                    model_content = types.Content(role="model", parts=[types.Part.from_text(text="I hit an error while responding. Please try again.")])
                    st.session_state.messages.append(model_content)

    st.rerun()
