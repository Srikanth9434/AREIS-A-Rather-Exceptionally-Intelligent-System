"""
voice_io.py
Local talk I/O for AREIS.

Listen: records from the default microphone (sounddevice) and transcribes
with SpeechRecognition. Prefers the Windows recognizer when available,
then Google's free endpoint if the PC is online.

Speak: Windows SAPI via pyttsx3 -- fully offline.
"""

import io
import re
import threading
import wave

LISTEN_RATE = 16000
LISTEN_CHANNELS = 1


class VoiceError(RuntimeError):
    pass


_tts_lock = threading.Lock()
_tts_engine = None
_stop_speak = threading.Event()


def _get_tts():
    global _tts_engine
    if _tts_engine is None:
        try:
            import pyttsx3
        except ImportError as e:
            raise VoiceError(
                "Voice replies need pyttsx3. In void_env run: pip install pyttsx3"
            ) from e
        engine = pyttsx3.init()
        engine.setProperty("rate", 185)
        _tts_engine = engine
    return _tts_engine


def stop_speaking():
    _stop_speak.set()
    try:
        engine = _get_tts()
        engine.stop()
    except Exception:
        pass


def _speakable(text: str) -> str:
    """Drop code-heavy blocks so the voice does not read Python line by line."""
    if not text:
        return ""
    cleaned = re.sub(r"```.*?```", " (code is in the console) ", text, flags=re.S)
    lines = []
    codeish = 0
    for line in cleaned.splitlines():
        stripped = line.strip()
        if re.match(
            r"^(import |from |def |class |if |for |while |return |#|</|[{}\]])",
            stripped,
        ):
            codeish += 1
            continue
        if stripped:
            lines.append(stripped)
    spoken = " ".join(lines).strip()
    if codeish >= 6 and not spoken:
        return "I put the code in the console."
    if len(spoken) > 900:
        spoken = spoken[:900].rsplit(" ", 1)[0] + "."
    return spoken


def speak(text: str):
    spoken = _speakable(text)
    if not spoken:
        return
    _stop_speak.clear()
    with _tts_lock:
        engine = _get_tts()
        if _stop_speak.is_set():
            return
        engine.say(spoken)
        engine.runAndWait()


def _rms(chunk: bytes) -> float:
    import audioop
    if not chunk:
        return 0.0
    return audioop.rms(chunk, 2)


def listen_once(status_cb=None, cancel_event=None, max_seconds=12):
    """Record until a short silence (or max_seconds) and return transcript text."""
    try:
        import sounddevice as sd
        import speech_recognition as sr
    except ImportError as e:
        raise VoiceError(
            "Talk mode needs sounddevice and SpeechRecognition. "
            "In void_env run: pip install sounddevice SpeechRecognition"
        ) from e

    if status_cb:
        status_cb("Listening… speak now")

    frames = []
    started = False
    silent_chunks = 0
    chunk_ms = 80
    samples = int(LISTEN_RATE * chunk_ms / 1000)
    max_chunks = int(max_seconds * 1000 / chunk_ms)
    silence_needed = int(900 / chunk_ms)

    try:
        with sd.RawInputStream(
            samplerate=LISTEN_RATE,
            channels=LISTEN_CHANNELS,
            dtype="int16",
            blocksize=samples,
        ) as stream:
            for _ in range(max_chunks):
                if cancel_event is not None and cancel_event.is_set():
                    raise VoiceError("Listening cancelled")
                data, overflowed = stream.read(samples)
                chunk = bytes(data)
                energy = _rms(chunk)
                if not started:
                    if energy > 450:
                        started = True
                        frames.append(chunk)
                    continue
                frames.append(chunk)
                if energy < 350:
                    silent_chunks += 1
                    if silent_chunks >= silence_needed:
                        break
                else:
                    silent_chunks = 0
    except Exception as e:
        raise VoiceError(f"Microphone error: {e}") from e

    if not frames:
        raise VoiceError("I did not hear anything. Try again a bit closer to the mic.")

    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wav:
        wav.setnchannels(LISTEN_CHANNELS)
        wav.setsampwidth(2)
        wav.setframerate(LISTEN_RATE)
        wav.writeframes(b"".join(frames))
    wav_buf.seek(0)

    if status_cb:
        status_cb("Transcribing…")

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_buf) as source:
        audio = recognizer.record(source)

    errors = []
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        errors.append("unclear speech")
    except sr.RequestError as e:
        errors.append(str(e))
    except Exception as e:
        errors.append(str(e))

    try:
        return recognizer.recognize_sphinx(audio)
    except Exception as e:
        errors.append(str(e))

    raise VoiceError(
        "Could not transcribe that. Check the mic, speak clearly, "
        "and make sure you are online for speech-to-text. "
        + "; ".join(errors[:2])
    )
