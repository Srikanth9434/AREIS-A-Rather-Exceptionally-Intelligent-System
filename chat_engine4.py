"""
chat_engine4.py
All AI logic lives here. Instead of loading the model directly via
llama-cpp-python (which needs a C++ compiler to build), this version talks
to a locally running llama-server.exe (CUDA build) over HTTP on
127.0.0.1 -- fully offline, no internet involved, GPU-accelerated.

main.py is responsible for starting llama-server.exe before this module's
functions are used. This file just sends requests to it.

Requires TWO model files in the models/ folder (same as before):
  - the main Qwen3.5 GGUF
  - the matching mmproj GGUF (the vision projector), loaded by main.py
    when it launches llama-server.exe with --mmproj
"""

import os
import re
import sys
import json
import base64
import cv2
import requests

if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
HEALTH_URL = "http://127.0.0.1:8080/health"

COMMAND = (
    "You are my butler who will assist me in studies and make my life easier. "
    "You are running inside a simple GUI chat window, not a markdown-rendering app. "
    "Never use latex (no $$ or \\begin{...}), never use markdown headers or tables. "
    "For math, write it in plain text (e.g. 'lambda' instead of the Greek letter, "
    "'A - lambda I' as plain characters, matrices as simple bracketed rows "
    "like [2, 1; 4, 3]). "
    "Be concise. Only go step-by-step if the user explicitly asks for a full explanation. "
    "You can also see images the user attaches from their webcam -- describe and "
    "analyze them accurately when present."
)

MODE_DETECTION = re.compile(
    r"\b(code|function|algorithm|debug|write a|explain why|prove|solve|"
    r"calculate|compare|analyze|design|optimi[sz]e|step by step|math)\b"
)

PROMPT_ENHANCER = """
You are a prompt enhancer.

Rewrite the user's request into a clearer, more precise instruction for another AI.

Rules:
- Preserve the user's original intent exactly.
- Do not answer the request.
- Do not add unnecessary assumptions.
- Fix ambiguity, grammar, and missing context when possible.
- Keep it concise.
- If the prompt is already clear, return it almost unchanged.
- Return ONLY the enhanced prompt, with no explanation.

User request:
"""

# Conversation memory
Storage = [
    {"role": "system", "content": COMMAND}
]


def is_model_loaded() -> bool:
    """Checks whether llama-server.exe is up and responding."""
    try:
        r = requests.get(HEALTH_URL, timeout=2)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def maybe_detection(prompt: str) -> bool:
    return bool(MODE_DETECTION.search(prompt.lower())) or len(prompt.split()) > 25


def _call_server(messages, stream=False, max_tokens=2048, think=True):
    """Low-level call to llama-server.exe's OpenAI-compatible endpoint.

    max_tokens raised 400 -> 1024 -> 2048: this model reasons before every
    answer (766 reasoning tokens were spent on a plain "say hello" test),
    and that reasoning counts against max_tokens. -c (context size) in
    main4.py's llama-server launch args must be comfortably larger than
    this number plus your prompt/history length, or generation gets cut
    off mid-reasoning with no answer at all. If blank replies persist on
    longer conversations, raise -c further before raising this.

    think: previously this was controlled by prepending the literal text
    "/think" or "/no_think" to the prompt, but that doesn't actually work
    for this model/server combo -- the model was seen analyzing "/no_think"
    as if it were part of the user's message instead of obeying it. The
    real mechanism llama-server uses for Qwen3-style hybrid-thinking models
    is the chat_template_kwargs.enable_thinking request field, sent below.
    """
    return requests.post(
        SERVER_URL,
        json={
            "messages": messages,
            "stream": stream,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": think},
        },
        stream=stream,
        timeout=120,
    )


def enhance_prompt(prompt: str) -> str:
    """Rewrites the user's prompt to be clearer before sending it to the main model.
    Skipped entirely when an image is attached -- the enhancer is text-only."""
    try:
        response = _call_server(
            messages=[
                {"role": "system", "content": PROMPT_ENHANCER},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        data = response.json()
        enhanced = data["choices"][0]["message"]["content"].strip()
        if enhanced:
            return enhanced
    except Exception as e:
        print(f"[Prompt enhancer unavailable: {e}]")

    return prompt


def clear_memory():
    global Storage
    Storage = [{"role": "system", "content": COMMAND}]


# --- Webcam capture helpers ---

def capture_webcam_image(save_path=None, cam_index=0):
    """Grabs a single frame from the default webcam and saves it as a JPEG.
    Returns the path to the saved image."""
    if save_path is None:
        save_path = os.path.join(BASE_DIR, "snapshot.jpg")

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check it isn't in use by another app.")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Failed to capture frame from webcam.")

    cv2.imwrite(save_path, frame)
    return save_path


def _image_to_data_uri(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower().lstrip(".") or "jpeg"
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{b64}"


def stream_reply(prompt: str, force_think=None, image_path: str = None):
    """
    Generator that yields pieces of the AI's reply as they arrive.
    Yields tuples: ("thinking_start", None) | ("token", text) | ("done", full_reply) | ("error", msg)

    image_path: optional path to an image (e.g. from capture_webcam_image()) to
    attach to this message. When present, the prompt enhancer is skipped and
    the message is sent as multimodal content (text + image) to llama-server,
    which routes it through the mmproj vision projector it was started with.
    """
    global Storage

    think = force_think if force_think is not None else maybe_detection(prompt)

    final_text = prompt

    if image_path:
        data_uri = _image_to_data_uri(image_path)
        user_content = [
            {"type": "text", "text": final_text},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]
    else:
        user_content = final_text

    Storage.append({"role": "user", "content": user_content})

    try:
        response = _call_server(messages=Storage, stream=True, think=think)
        if response.status_code != 200:
            Storage.pop()
            yield ("error", f"Server returned status {response.status_code}: {response.text}")
            return
    except requests.exceptions.RequestException as e:
        Storage.pop()
        yield ("error", f"Could not reach local model server: {e}")
        return

    full_reply = ""
    said_thinking = False
    saw_any_token = False

    # NOTE: this llama-server build does NOT emit <think>...</think> tags
    # inside the "content" field. Instead it streams reasoning separately
    # in each chunk's delta.reasoning_content, and the final answer in
    # delta.content. The old code only ever read delta["content"], so
    # during the entire reasoning phase (which can be hundreds of tokens
    # for even a trivial prompt) it saw nothing -- and if reasoning alone
    # used up the whole max_tokens budget, content never started and the
    # reply came back completely blank. Both fields are now read directly.
    try:
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload.strip() == "[DONE]":
                break

            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue

            delta = chunk["choices"][0].get("delta", {})
            reasoning_token = delta.get("reasoning_content", "") or ""
            content_token = delta.get("content", "") or ""

            if reasoning_token:
                saw_any_token = True
                if not said_thinking:
                    said_thinking = True
                    yield ("thinking_start", None)
                # Reasoning is shown too (not appended to full_reply, since
                # it isn't part of the actual answer) so you always see
                # something instead of a blank wait.
                yield ("token", reasoning_token)

            if content_token:
                saw_any_token = True
                full_reply += content_token
                yield ("token", content_token)
    except Exception as e:
        yield ("error", str(e))

    if not saw_any_token and not full_reply:
        # The server responded but the stream produced literally nothing --
        # surface this as a visible error instead of a silent blank bubble.
        msg = "[No response from the model -- it may have hit its token limit or the server errored. Try again or lower -c/-ngl if this repeats.]"
        full_reply = msg
        yield ("token", msg)

    Storage.append({"role": "assistant", "content": full_reply})
    yield ("done", full_reply)