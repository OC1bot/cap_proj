"""
app.py - Streamlit front end for the "What's In This Image?" project.

This file only handles the user interface: the sidebar for connection
settings, the image upload with validation, and the follow-up chat panel.
All AI logic (chains, prompts, parsing, memory) lives in pipeline.py.

Run with:
    streamlit run app.py

Important Streamlit concept used throughout this file: Streamlit reruns
this ENTIRE script from top to bottom on every user interaction (every
click, every keystroke committed). Ordinary variables are therefore lost
between interactions; anything that must survive is kept in
st.session_state, which persists for the whole browser session.
"""

import io                      # wraps raw bytes so Pillow can open them like a file

import streamlit as st         # the web UI framework
from PIL import Image          # used here only to validate and preview the upload

# Everything AI-related is imported from the pipeline module.
from pipeline import (
    DEFAULT_BASE_URL,          # http://localhost:11434, prefilled in Local mode
    DEFAULT_MODEL,             # preselected in the dropdown when available
    InMemoryChatMessageHistory,  # the LangChain memory object for follow-ups
    build_describe_chain,      # LCEL chain: preprocess | prompt | model | parser
    build_followup_chain,      # memory-backed chain for follow-up questions
    encode_image_bytes,        # bytes -> base64, cached for the no-re-upload feature
    check_ollama_vision,       # asks Ollama whether a model can accept images
    check_remote_vision,       # probes a remote endpoint with a 1x1 test image
    list_ollama_models,        # asks a local Ollama server what is installed
    list_openai_models,        # asks a remote OpenAI compatible endpoint the same
    make_llm,                  # builds a ChatOllama model (Local mode)
    make_remote_llm,           # builds a ChatOpenAI model (Remote mode)
)

# Browser tab title, icon, and a wide page so the two columns fit nicely.
st.set_page_config(page_title="What's In This Image?", page_icon="🖼", layout="wide")

# ---------------------------------------------------------------------------
# Cached model discovery
#
# Because Streamlit reruns the whole script constantly, calling the model
# listing endpoints directly would hit the server on every keystroke.
# st.cache_data stores the result keyed by the arguments (URL, key) and
# reuses it until the ttl (time to live, in seconds) expires or the cache
# is cleared by the Refresh button.
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30, show_spinner=False)
def cached_local_models(url: str) -> list:
    """Cache the Ollama /api/tags call between Streamlit reruns."""
    return list_ollama_models(url)


@st.cache_data(ttl=60, show_spinner=False)
def cached_remote_models(url: str, key: str) -> list:
    """Cache the remote /models call between Streamlit reruns."""
    return list_openai_models(url, key)


@st.cache_data(ttl=300, show_spinner=False)
def cached_vision_check(model_name: str, url: str):
    """Cache the /api/show vision capability probe per model."""
    return check_ollama_vision(model_name, url)


@st.cache_data(ttl=300, show_spinner=False)
def cached_remote_vision_check(model_name: str, url: str, key: str):
    """Cache the remote 1x1 test image probe per model."""
    return check_remote_vision(model_name, url, key)


# Defaults for this rerun. They stay None until the user completes each
# sidebar step, and the code after the sidebar checks them to decide
# whether the app is ready to run.
model = None
mode = None
base_url = None
api_key = None
vision_blocked = False   # set True when the selected model fails the vision check

# ---------------------------------------------------------------------------
# Sidebar: connection settings, revealed step by step
#
# Step 1: choose Local or Remote.
# Step 2: enter the connection details for that choice.
#         (Remote also requires pressing Connect.)
# Step 3: pick a model from the list fetched from that server.
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    # Step 1: pick the connection type. index=None means nothing is
    # preselected, so only the placeholder shows until the user chooses,
    # and the rest of the sidebar stays hidden.
    mode = st.selectbox(
        "Connection type",
        ["Local", "Remote"],
        index=None,
        placeholder="Select Local or Remote",
    )

    # Step 2 (Local): one field only, prefilled with the default address.
    if mode == "Local":
        base_url = st.text_input("Local Ollama URL", value=DEFAULT_BASE_URL)

    # Step 2 (Remote): URL + API key + Connect button.
    elif mode == "Remote":
        base_url = st.text_input(
            "Remote API URL",
            placeholder="https://api.openai.com/v1",
            help="Any OpenAI compatible endpoint: " 
                "OpenAI (https://api.openai.com/v1), "
                "Ollama Cloud (https://ollama.com/v1), "
                "OpenRouter (https://openrouter.ai/api/v1), " 
                "Hugging Face (https://router.huggingface.co/v1) and similar.",
        )
        # type="password" masks the key with dots as the user types.
        api_key = st.text_input("API key", type="password")

        # Security/consistency guard: if the user edits either field after
        # connecting, the old connection must not silently keep being used.
        # We remember the credentials that were connected with, and any
        # difference resets the connected flag, which brings the Connect
        # button back.
        creds = (base_url or "", api_key or "")
        if st.session_state.get("remote_creds") != creds:
            st.session_state.remote_connected = False

        # The Connect button is always visible in Remote mode. It stays
        # disabled (greyed out) until both fields are filled in, and it
        # disappears once connected.
        if not st.session_state.get("remote_connected"):
            connect_disabled = not (base_url and api_key)
            if st.button(
                "Connect",
                type="primary",              # orange primary styling
                use_container_width=True,    # full sidebar width
                disabled=connect_disabled,   # greyed out until fields are filled
            ):
                # Connecting = trying to list models. A wrong key or a dead
                # endpoint fails here, at Connect time, not later during
                # analysis, which gives the user immediate feedback.
                with st.spinner("Connecting..."):
                    found = cached_remote_models(base_url, api_key)
                if found:
                    st.session_state.remote_connected = True
                    st.session_state.remote_creds = creds
                    st.rerun()   # rerun immediately so the UI switches to the connected view
                else:
                    st.error(
                        f"Could not connect to {base_url}. Check the URL and "
                        "the API key, then press Connect again."
                    )
            if connect_disabled:
                st.caption("Enter the URL and API key to enable Connect.")

        # Green banner confirming which endpoint is connected.
        if st.session_state.get("remote_connected"):
            st.success(f"Connected to {base_url}")

    # Step 3 gate: the model list only loads when the connection is ready.
    # Local is ready as soon as a URL exists; Remote is ready only after a
    # successful Connect.
    if mode == "Local":
        details_ready = bool(base_url)
    elif mode == "Remote":
        details_ready = bool(st.session_state.get("remote_connected"))
    else:
        details_ready = False

    if details_ready:
        # Clearing both caches forces a fresh call to the server, e.g.
        # after pulling a new model with `ollama pull`.
        if st.button("Refresh model list"):
            cached_local_models.clear()
            cached_remote_models.clear()

        # Fetch the model names from whichever backend is active.
        if mode == "Local":
            available = cached_local_models(base_url)
        else:
            available = cached_remote_models(base_url, api_key)

        if not available:
            # Empty list means the server could not be reached or returned
            # nothing; tell the user what to check.
            st.error(
                f"Could not list models from {base_url}. Check that the "
                "server is reachable"
                + (" and the API key is valid" if mode == "Remote" else "")
                + ", then press Refresh model list."
            )
        else:
            # Preselect the project default model if the server has it.
            # Matching is by exact name first, then by the part before the
            # colon, so DEFAULT_MODEL = "llava" also matches "llava:latest".
            default_index = 0
            for i, name in enumerate(available):
                if name == DEFAULT_MODEL or name.split(":")[0] == DEFAULT_MODEL:
                    default_index = i
                    break
            model = st.selectbox("Model", available, index=default_index)
            st.caption(f"{len(available)} models found on the server.")

            # Vision validation happens HERE, at selection time, so a text
            # only model is caught before the user ever presses Analyze
            # Image, instead of failing later with a 400 multimodal error.
            if mode == "Local" and model:
                verdict = cached_vision_check(model, base_url)
                if verdict is False:
                    st.error(
                        f"'{model}' is a text only model and cannot analyse "
                        "images. Select a vision capable model such as "
                        "llava or llama3.2-vision."
                    )
                    vision_blocked = True
                    model = None
                elif verdict is True:
                    st.caption("Vision capability confirmed for this model.")
                else:
                    st.warning(
                        "Could not verify whether this model supports "
                        "images. If analysis fails with a multimodal "
                        "error, pick a vision capable model instead."
                    )
            elif mode == "Remote" and model:
                # Remote providers expose no capability metadata, so the
                # probe actually sends a 1x1 test image with max_tokens=1
                # to this model and checks whether it is accepted. This
                # may consume a token or two on the provider account.
                with st.spinner("Checking vision capability..."):
                    verdict = cached_remote_vision_check(model, base_url, api_key)
                if verdict is False:
                    st.error(
                        f"'{model}' rejected image input, so it cannot "
                        "analyse images. Select a vision capable model "
                        "such as gpt-4o."
                    )
                    vision_blocked = True
                    model = None
                elif verdict is True:
                    st.caption(
                        "Vision capability confirmed with a tiny test request."
                    )
                else:
                    st.warning(
                        "Could not verify whether this model supports "
                        "images (the test request was inconclusive). If "
                        "analysis fails with a multimodal error, pick a "
                        "vision capable model instead."
                    )

    # Wipes the analysis, the cached image, and the LangChain memory so
    # the user can start over without restarting the app.
    if st.button("Reset conversation"):
        for key in ("history", "result", "image_b64", "chat_log", "image_name"):
            st.session_state.pop(key, None)
        st.rerun()

# ---------------------------------------------------------------------------
# Guidance screen
#
# Until a model is selected (model is still None), the main page shows a
# message telling the user which sidebar step is missing, then st.stop()
# ends the script so none of the app below renders half-configured.
# ---------------------------------------------------------------------------

if model is None:
    st.title("What's In This Image?")
    if mode is None:
        st.info("Start in the sidebar: select Local or Remote as the connection type.")
    elif mode == "Remote" and not (base_url and api_key):
        st.info("Enter the remote API URL and API key in the sidebar, then press Connect.")
    elif mode == "Remote" and not st.session_state.get("remote_connected"):
        st.info("Press Connect in the sidebar to load the model list.")
    elif vision_blocked:
        st.info(
            "The selected model does not support images. Pick a vision "
            "capable model in the sidebar to continue."
        )
    else:
        st.info("Complete the connection details in the sidebar to load the model list.")
    st.stop()

# ---------------------------------------------------------------------------
# Session state initialisation
#
# This is where the "follow-up questions without re-upload" requirement is
# satisfied. The base64 image and the LangChain message history both live
# in st.session_state, so they survive every Streamlit rerun for the whole
# browser session.
# ---------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = InMemoryChatMessageHistory()   # LangChain memory object
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []   # (role, text) pairs used only to redraw the chat UI


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    RunnableWithMessageHistory calls this on every invocation to fetch the
    memory object for the given session id. This app has one browser
    session, so it always returns the single history from session state.
    """
    return st.session_state.history


# Build the chat model for whichever backend the user configured, then
# build both chains around it. The chains are identical for Local and
# Remote because both model objects speak the same LangChain interface.
if mode == "Local":
    llm = make_llm(model=model, base_url=base_url)
else:
    llm = make_remote_llm(model, base_url, api_key)
describe_chain = build_describe_chain(llm)
followup_chain = build_followup_chain(llm, get_history)

# ---------------------------------------------------------------------------
# Main layout: title, then two equal columns.
# Left column  = upload, validation, Analyze button, structured result.
# Right column = follow-up chat about the analysed image.
# ---------------------------------------------------------------------------

st.title("What's In This Image?")
st.write("Upload an image, get an AI description, then ask follow-up questions.")

left, right = st.columns([1, 1])

# The two allowed lists used by the three validation layers below.
# ALLOWED_EXTENSIONS checks the file NAME; ALLOWED_FORMATS checks what the
# file CONTENT actually is once Pillow has decoded it.
ALLOWED_EXTENSIONS = ("jpg", "jpeg", "png", "webp", "avif")
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "AVIF"}

with left:
    uploaded = st.file_uploader(
        "Upload an image",
        type=list(ALLOWED_EXTENSIONS),   # the file picker filters to these extensions
        help="JPG and PNG are fully supported; WEBP and AVIF also work.",
    )

    if uploaded is not None:
        # Validation step 1: the file extension must be in the allowed list.
        # The picker already filters, but drag and drop or a renamed file
        # can bypass it, so check again here.
        ext = uploaded.name.rsplit(".", 1)[-1].lower() if "." in uploaded.name else ""
        if ext not in ALLOWED_EXTENSIONS:
            st.error(
                f"'{uploaded.name}' has an invalid file type. Please upload "
                "a correct format: JPG, JPEG, PNG, WEBP or AVIF."
            )
            st.stop()

        # Validation step 2: the file must actually decode as an image.
        # This catches corrupt files, non images renamed with a valid
        # extension, and missing codecs (e.g. AVIF without libavif).
        # preview.load() forces a full decode now, instead of letting
        # Pillow decode lazily and crash later inside st.image().
        raw = uploaded.getvalue()
        try:
            preview = Image.open(io.BytesIO(raw))
            preview.load()
        except Exception:
            st.error(
                f"'{uploaded.name}' is not a valid image or cannot be "
                "decoded. Please upload a correct format: JPG, JPEG, PNG, "
                "WEBP or AVIF. If this is a real AVIF file, your Pillow "
                "installation may lack the codec; try: pip install --upgrade pillow"
            )
            st.stop()

        # Validation step 3: the real decoded format must match the allowed
        # list, so a BMP or GIF renamed to .png is still rejected.
        detected = (preview.format or "").upper()
        if detected not in ALLOWED_FORMATS:
            st.error(
                f"'{uploaded.name}' is actually a {detected or 'unknown'} "
                "file, which is not supported. Please upload a correct "
                "format: JPG, JPEG, PNG, WEBP or AVIF."
            )
            st.stop()

        # All three checks passed: show the preview.
        st.image(preview, use_container_width=True)

        # New file selected: clear the previous analysis and memory, so
        # answers about the old image never leak into the new one.
        if st.session_state.get("image_name") != uploaded.name:
            st.session_state.image_name = uploaded.name
            st.session_state.pop("result", None)
            st.session_state.pop("image_b64", None)
            st.session_state.history = InMemoryChatMessageHistory()
            st.session_state.chat_log = []

        if st.button("Analyze Image", type="primary", use_container_width=True):
            # This is the single call that runs the whole LCEL pipeline:
            # preprocess -> prompt -> vision model -> structured parser.
            with st.spinner("Analysing image with the vision model..."):
                try:
                    result = describe_chain.invoke({"image_bytes": raw})
                except Exception as exc:
                    st.error(
                        f"Could not reach the model: {exc}. "
                        "Check that Ollama is running and the model is pulled."
                    )
                    st.stop()

            # Cache everything the follow-up chat needs. image_b64 is the
            # key piece: it lets every follow-up re-attach the image
            # without the user uploading it again.
            st.session_state.result = result
            st.session_state.image_b64 = encode_image_bytes(raw)

            # Seed the LangChain memory with the first exchange, so the
            # model knows what it already said when follow-ups arrive.
            st.session_state.history = InMemoryChatMessageHistory()
            st.session_state.history.add_user_message(
                "Please describe the image I uploaded."
            )
            st.session_state.history.add_ai_message(result.description)
            st.session_state.chat_log = []

    # Render the structured result (the ImageDescription Pydantic object)
    # whenever one exists in session state, so it survives reruns.
    if "result" in st.session_state:
        r = st.session_state.result
        st.markdown(f"**Description:** {r.description}")
        st.markdown(f"**Objects:** {', '.join(r.objects) if r.objects else 'none detected'}")
        st.markdown(f"**Scene type:** {r.scene_type}")

with right:
    st.subheader("Ask follow-up questions about this image")

    # No analysed image yet: show guidance instead of an empty chat box.
    if "image_b64" not in st.session_state:
        st.info("Upload an image and press Analyze Image first.")
    else:
        # Redraw all previous turns. Necessary because Streamlit reruns
        # the script on every interaction and would otherwise show an
        # empty chat each time.
        for role, text in st.session_state.chat_log:
            with st.chat_message(role):
                st.write(text)

        # chat_input renders the text box pinned at the bottom and returns
        # the typed question once, on the rerun where the user submits it.
        question = st.chat_input("e.g. Is there a person in this image?")
        if question:
            # Show the user's bubble immediately.
            st.session_state.chat_log.append(("user", question))
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # The memory-backed chain: past turns are injected
                        # automatically, the cached image is re-attached,
                        # and this new question + answer are appended to
                        # the history afterwards. session_id tells
                        # get_history() which conversation this belongs to.
                        answer = followup_chain.invoke(
                            {
                                "question": question,
                                "image_b64": st.session_state.image_b64,
                            },
                            config={"configurable": {"session_id": "streamlit"}},
                        )
                    except Exception as exc:
                        # A model error becomes a chat message rather than
                        # a crash, so the conversation can continue.
                        answer = f"Error talking to the model: {exc}"
                st.write(answer)

            # Store the answer so it is redrawn on future reruns.
            st.session_state.chat_log.append(("assistant", answer))