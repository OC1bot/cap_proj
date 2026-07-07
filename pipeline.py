"""
pipeline.py - LCEL pipeline for the image description app.

This module contains all the LangChain logic. The Streamlit file (app.py)
only handles the user interface; every AI-related piece lives here.

Chain shape (initial description):
    RunnableLambda(preprocess) | ChatPromptTemplate | ChatOllama | RunnableLambda(robust_parse)

Chain shape (follow-up questions):
    RunnableWithMessageHistory( ChatPromptTemplate | ChatOllama | StrOutputParser )
"""

# --- Standard library imports ----------------------------------------------
import base64            # converts binary image bytes into text the model API accepts
import io                # lets Pillow read image bytes from memory instead of a file on disk
import json              # parses the JSON returned by the model and by the HTTP endpoints
import urllib.request    # plain HTTP client used to list models (no extra dependency needed)
from typing import List  # type hints for readability

# --- Third party imports ----------------------------------------------------
from PIL import Image                    # Pillow: opens, converts, and resizes images
from pydantic import BaseModel, Field    # defines the structured output schema

# --- LangChain imports -------------------------------------------------------
from langchain_core.chat_history import InMemoryChatMessageHistory   # the memory object that stores past turns
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser  # turn model text into objects/strings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder       # prompt templates with variables
from langchain_core.runnables import RunnableLambda                  # wraps a plain Python function as a chain step
from langchain_core.runnables.history import RunnableWithMessageHistory  # adds conversational memory around a chain
from langchain_ollama import ChatOllama                              # chat interface to a local Ollama server

# --- Configuration constants -------------------------------------------------
DEFAULT_MODEL = "llava"                      # preselected in the dropdown when installed (matched by exact name or by the part before the colon)
DEFAULT_BASE_URL = "http://localhost:11434"  # default address of a local Ollama server
MAX_IMAGE_SIDE = 1024                        # images are downscaled so no side exceeds this many pixels


# ---------------------------------------------------------------------------
# Model discovery over HTTP
#
# These two functions do NOT run any AI. They only ask a server which models
# it offers, so the app can fill the dropdown list. Both return an empty
# list on any failure, and the app treats an empty list as "cannot connect".
# ---------------------------------------------------------------------------

def list_ollama_models(base_url: str = DEFAULT_BASE_URL, timeout: float = 3.0) -> List[str]:
    """
    Ask a LOCAL Ollama server which models are installed.

    Ollama exposes GET /api/tags, which returns JSON like:
        {"models": [{"name": "llama3.2-vision:11b", ...}, ...]}
    We extract every "name" and return them sorted. The equivalent of
    running `ollama list` on the command line.
    """
    url = base_url.rstrip("/") + "/api/tags"   # rstrip avoids a double slash if the user typed a trailing /
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return sorted(m.get("name", "") for m in data.get("models", []) if m.get("name"))
    except Exception:
        # Server down, wrong address, firewall, malformed reply: all become []
        return []


def check_ollama_vision(model: str, base_url: str = DEFAULT_BASE_URL, timeout: float = 5.0):
    """
    Ask a local Ollama server whether a model can accept images.

    POST /api/show with {"model": name} returns metadata about the model.
    Two signals are checked, from strongest to weakest:
      1. A "capabilities" list (newer Ollama versions) that contains
         "vision" for multimodal models.
      2. The "details.families" list (older versions), where families such
         as "clip" or "mllama" indicate an attached vision encoder.

    Returns True (vision capable), False (text only), or None when the
    server cannot be reached or the metadata is inconclusive, so the
    caller can warn instead of blocking.
    """
    url = base_url.rstrip("/") + "/api/show"
    payload = json.dumps({"model": model}).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None

    # Signal 1: explicit capabilities list.
    caps = data.get("capabilities") or []
    if caps:
        return "vision" in caps

    # Signal 2: model families that imply a vision encoder.
    families = (data.get("details") or {}).get("families") or []
    if any(f in ("clip", "mllama") for f in families):
        return True

    # Metadata present but no vision signal either way: inconclusive.
    return None


_TINY_JPEG_B64 = None


def _tiny_image_b64() -> str:
    """A valid 1x1 white JPEG, generated once and reused, for the remote probe."""
    global _TINY_JPEG_B64
    if _TINY_JPEG_B64 is None:
        buf = io.BytesIO()
        Image.new("RGB", (1, 1), (255, 255, 255)).save(buf, format="JPEG")
        _TINY_JPEG_B64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return _TINY_JPEG_B64


def check_remote_vision(model: str, base_url: str, api_key: str, timeout: float = 20.0):
    """
    Functional vision probe for REMOTE OpenAI compatible endpoints.

    Remote providers expose no standard capability metadata, so the only
    reliable check is to actually try: send a chat completion containing a
    1x1 test image with max_tokens=1 and see whether the endpoint accepts
    it. This costs at most a token or two and works the same way on
    OpenAI, Ollama Cloud, OpenRouter and the Hugging Face router.

    Returns True when the request is accepted (vision capable), False when
    the endpoint rejects it with a message indicating images are not
    supported, and None for anything inconclusive (auth failures, rate
    limits, timeouts), so the caller warns instead of wrongly blocking.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "max_tokens": 1,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{_tiny_image_b64()}"
                        },
                    },
                ],
            }
        ],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            return True  # the endpoint accepted an image for this model
    except urllib.error.HTTPError as e:
        try:
            msg = json.dumps(json.loads(e.read().decode("utf-8"))).lower()
        except Exception:
            msg = ""
        # A 400/422 whose error text mentions images is a clear "text only"
        # verdict, e.g. "Multimodal data provided, but model does not
        # support multimodal requests."
        if e.code in (400, 422) and any(
            k in msg for k in ("multimodal", "image", "vision")
        ):
            return False
        return None  # auth error, rate limit, or unrelated 4xx/5xx
    except Exception:
        return None  # unreachable, timeout, malformed reply

#----------------------------------------------------------------------------
# The project requires the initial description to come back as structured
# data, not free text. Pydantic defines the shape, and PydanticOutputParser
# does two jobs: it generates the "format instructions" text that tells the
# model exactly what JSON to produce, and it converts the model's reply
# into a real Python object with typed fields.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 1. Structured output schema (Pydantic) + parser
#
# The project requires the initial description to come back as structured
# data, not free text. Pydantic defines the shape, and PydanticOutputParser
# does two jobs: it generates the "format instructions" text that tells the
# model exactly what JSON to produce, and it converts the model's reply
# into a real Python object with typed fields.
# ---------------------------------------------------------------------------

class ImageDescription(BaseModel):
    """Structured result for the initial image analysis."""
    description: str = Field(description="Two to four sentence description of the image")
    objects: List[str] = Field(description="List of the main objects visible in the image")
    scene_type: str = Field(description="One or two word scene category, e.g. 'airport', 'kitchen', 'outdoor'")


# One shared parser instance, used both to build the prompt and to parse replies.
parser = PydanticOutputParser(pydantic_object=ImageDescription)


# ---------------------------------------------------------------------------
# 2. Preprocess step (the required RunnableLambda)
#
# RunnableLambda wraps an ordinary Python function so it can sit inside an
# LCEL chain and be composed with the | operator. This one is the FIRST
# step of the describe chain: it receives the raw upload and outputs the
# variables the prompt template needs.
# ---------------------------------------------------------------------------

def preprocess(inputs: dict) -> dict:
    """
    Takes raw uploaded bytes, normalises the image, and returns a base64 string.

    - Opens JPG, PNG, WEBP, AVIF, etc. via Pillow
    - Converts to RGB (handles PNG alpha channels, which JPEG cannot store)
    - Downscales to a maximum side of 1024 px so the model gets a
      consistent, reasonably sized input regardless of the original size
    - Encodes as base64 JPEG, because the chat API carries images as text
    """
    raw: bytes = inputs["image_bytes"]          # the chain is invoked with {"image_bytes": <bytes>}
    img = Image.open(io.BytesIO(raw))           # BytesIO makes the bytes look like a file for Pillow
    if img.mode != "RGB":
        img = img.convert("RGB")                # e.g. RGBA (transparent PNG) or P (palette) -> RGB
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))  # resizes in place, keeps aspect ratio, never enlarges
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)    # re-encode everything as JPEG at high quality
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")  # bytes -> base64 text
    return {
        # These two keys match the {image_b64} and {format_instructions}
        # placeholders inside describe_prompt below.
        "image_b64": b64,
        "format_instructions": parser.get_format_instructions(),
    }


def encode_image_bytes(raw: bytes) -> str:
    """
    Standalone helper for the app: after the first analysis, app.py caches
    the base64 image in session state so follow-up questions can re-attach
    it without the user uploading the file again.
    """
    return preprocess({"image_bytes": raw})["image_b64"]


# ---------------------------------------------------------------------------
# 3. Robust output parsing step (the second RunnableLambda)
#
# This is the LAST step of the describe chain. Local vision models often
# ignore "JSON only" instructions and wrap the JSON in prose or markdown
# fences, so parsing happens in three layers, from strict to forgiving.
# ---------------------------------------------------------------------------

def robust_parse(ai_message) -> ImageDescription:
    """
    Layer 1: strict Pydantic parse of the whole reply.
    Layer 2: cut out the outermost {...} block and parse that, which
             recovers replies wrapped in ```json fences or extra sentences.
    Layer 3: give up on JSON and return the raw text as the description,
             with scene_type set to "unknown", so the app never crashes.
    """
    # The model reply is an AIMessage object; .content holds the text.
    text = ai_message.content if hasattr(ai_message, "content") else str(ai_message)
    if isinstance(text, list):
        # Some providers return content as a list of blocks; join the text parts.
        text = " ".join(part.get("text", "") for part in text if isinstance(part, dict))

    # Layer 1: the happy path when the model followed instructions exactly.
    try:
        return parser.parse(text)
    except Exception:
        pass

    # Layer 2: find the first { and the last } and try to parse what is between.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            return ImageDescription(
                description=str(data.get("description", "")).strip(),
                objects=[str(o) for o in data.get("objects", []) if str(o).strip()],
                scene_type=str(data.get("scene_type", "unknown")).strip() or "unknown",
            )
        except Exception:
            pass

    # Layer 3: graceful degradation, never an exception.
    return ImageDescription(description=text.strip(), objects=[], scene_type="unknown")


# ---------------------------------------------------------------------------
# 4. Prompts (ChatPromptTemplate)
#
# A ChatPromptTemplate is a list of messages with {placeholders}. At run
# time LangChain fills the placeholders with the values flowing through
# the chain. The human message here is MULTIMODAL: it contains a text
# block and an image block, which is how vision models receive pictures.
# ---------------------------------------------------------------------------

describe_prompt = ChatPromptTemplate.from_messages([
    (
        # The system message sets the model's behaviour and output rules.
        "system",
        "You are a precise image analyst. You describe images factually. "
        "You never invent objects that are not visible. "
        "You respond with valid JSON only, with no markdown fences and no extra text.",
    ),
    (
        # The human message carries the actual request plus the image.
        "human",
        [
            {
                "type": "text",
                "text": (
                    # {format_instructions} is replaced with the JSON schema
                    # text generated by the Pydantic parser above.
                    "Describe this image.\n\n{format_instructions}\n\n"
                    "Return the JSON object only."
                ),
            },
            {
                # The image travels as a data URL: a base64 JPEG embedded
                # directly in the message. {image_b64} comes from preprocess().
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,{image_b64}"},
            },
        ],
    ),
])

followup_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are answering follow-up questions about a single image the user "
        "uploaded earlier. Use the conversation history for context. "
        "Answer briefly and factually. If something is not visible in the "
        "image, say so instead of guessing.",
    ),
    # MessagesPlaceholder is an empty slot. RunnableWithMessageHistory
    # fills it with all previous question/answer turns on every call,
    # which is what makes follow-up questions like "how about the
    # airplane?" understandable after "what brand is the car?".
    MessagesPlaceholder("history"),
    (
        "human",
        [
            {"type": "text", "text": "{question}"},          # the new question typed by the user
            {
                # The same cached image is re-attached on every turn, because
                # the model needs to see the pixels again to answer new
                # visual questions. The user never re-uploads anything.
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,{image_b64}"},
            },
        ],
    ),
])


# ---------------------------------------------------------------------------
# 5. Model factories and chain builders
#
# The app supports two backends behind one interface: a local Ollama
# server, and any remote OpenAI compatible endpoint. Both produce a
# LangChain chat model object, so the chains below work with either.
# ---------------------------------------------------------------------------

def make_llm(model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL) -> ChatOllama:
    """LOCAL backend: chat model that talks to an Ollama server.
    temperature=0.2 keeps answers factual with little randomness."""
    return ChatOllama(model=model, base_url=base_url, temperature=0.2)


def list_openai_models(base_url: str, api_key: str, timeout: float = 8.0) -> List[str]:
    """
    REMOTE discovery: list models from an OpenAI compatible endpoint
    (GET {base}/models with an Authorization: Bearer <key> header).
    Works with OpenAI (https://api.openai.com/v1), Ollama Cloud (https://ollama.com/v1),
    OpenRouter (https://openrouter.ai/api/v1), Hugging Face (https://router.huggingface.co/v1), 
    and similar providers. The reply looks like:
        {"data": [{"id": "gpt-4o", ...}, ...]}
    Returns an empty list on any failure, which the app shows as a connection error.    
    """
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return sorted(m.get("id", "") for m in data.get("data", []) if m.get("id"))
    except Exception:
        return []


def make_remote_llm(model: str, base_url: str, api_key: str):
    """
    REMOTE backend: chat model for an OpenAI compatible endpoint.
    The import happens inside the function (lazily) so the app still runs
    for local-only use even if langchain-openai is not installed.
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, base_url=base_url, api_key=api_key, temperature=0.2)


def build_describe_chain(llm: ChatOllama):
    """
    The full LCEL pipeline for the initial description, composed with the
    | operator, exactly as the project scope requires:

        preprocess -> prompt -> model -> output parser

    Data flow when .invoke({"image_bytes": raw}) is called:
      1. preprocess turns the bytes into {"image_b64", "format_instructions"}
      2. describe_prompt fills its placeholders with those values
      3. llm sends the messages to the model and returns an AIMessage
      4. robust_parse turns the reply into an ImageDescription object
    """
    return (
        RunnableLambda(preprocess)
        | describe_prompt
        | llm
        | RunnableLambda(robust_parse)
    )


def build_followup_chain(llm: ChatOllama, get_history):
    """
    Memory-backed follow-up chain. `get_history` is a callable that maps a
    session_id to a ChatMessageHistory object; the app supplies one that
    returns the history stored in Streamlit session state.

    RunnableWithMessageHistory does the memory bookkeeping automatically:
    BEFORE each call it injects all stored turns into the "history"
    placeholder, and AFTER each call it appends the new question
    (input_messages_key) and the new answer to the history. This is real
    LangChain memory, not a hardcoded variable.
    """
    base = followup_prompt | llm | StrOutputParser()   # StrOutputParser: AIMessage -> plain string
    return RunnableWithMessageHistory(
        base,
        get_history,
        input_messages_key="question",     # which input field is "the user's new message"
        history_messages_key="history",    # which prompt placeholder receives past turns
    )


# Names that app.py is allowed to import from this module.
__all__ = [
    "DEFAULT_MODEL",
    "ImageDescription",
    "InMemoryChatMessageHistory",
    "build_describe_chain",
    "build_followup_chain",
    "encode_image_bytes",
    "check_ollama_vision",
    "check_remote_vision",
    "list_ollama_models",
    "list_openai_models",
    "make_remote_llm",
    "make_llm",
]