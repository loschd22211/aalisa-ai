#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Алиса — локальный голосовой ассистент. Мозг офлайн, ноль рублей.

  python alice.py            голосовой режим: говоришь — отвечает голосом
  python alice.py --text     текстовый чат в консоли
  python alice.py --model qwen2.5-coder:7b     другая модель Ollama

Стек: Ollama (мозг) + faster-whisper (понимает речь) + Piper (озвучка).
Проверено на: RTX 3070 8 ГБ, Intel i5-12400, 16 ГБ DDR4, Windows.
"""

import argparse
import os
import sys
import time
import wave
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    import ollama
except ImportError:
    print("Нет пакета ollama. Выполни:  pip install ollama")
    sys.exit(1)

# Голосовой стек может отсутствовать — тогда текстовый режим всё равно работает.
try:
    import numpy as np
    import sounddevice as sd
    import winsound
    from faster_whisper import WhisperModel
    from piper import PiperVoice
    VOICE_OK = True
    VOICE_ERR = None
except Exception as _e:
    VOICE_OK = False
    VOICE_ERR = _e

# ==================== НАСТРОЙКИ ====================
MODEL = "qwen3.5:9b"                  # мозг; запасной вариант: "qwen2.5-coder:7b"
NUM_CTX = 8192                        # контекст диалога (фикс проблемы из гайда)
TEMPERATURE = 0.7
WHISPER_MODEL = "small"               # распознавание: small (быстро) | medium (точнее)
VOICE_NAME = "ru_RU-irina-medium"     # голос; мужской — ru_RU-ruslan-medium
SR = 16000                            # частота записи микрофона
SILENCE_S = 1.2                       # тишина 1.2 с = конец фразы
MAX_S = 20                            # максимум длительность одной фразы
WAIT_S = 20                           # сколько ждать начала речи
REPLY_WAV = Path(__file__).resolve().parent / "_reply.wav"

SYSTEM = (
    "Ты — Алиса, локальный голосовой ассистент на компьютере владельца. "
    "Отвечай только по-русски, коротко и живо: 1-3 предложения, как голосовой помощник. "
    "Без markdown, без списков, без кода — твой ответ произносится вслух. "
    "Если вопрос непонятен — задай один уточняющий вопрос. "
    "Если не знаешь — честно скажи, что не знаешь."
)


# ==================== МОЗГ (Ollama) ====================
def _field(obj, name, default=None):
    """Достаёт поле из dict или объекта (разные версии библиотеки ollama)."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def list_models():
    resp = ollama.list()
    names = []
    for m in _field(resp, "models", []) or []:
        n = _field(m, "model") or _field(m, "name")
        if n:
            names.append(n)
    return names


def ask(history, text):
    """Отправляет вопрос в Ollama, печатает ответ по мере поступления."""
    history.append({"role": "user", "content": text})
    answer = ""
    stream = ollama.chat(
        model=MODEL,
        messages=history,
        stream=True,
        options={"num_ctx": NUM_CTX, "temperature": TEMPERATURE},
    )
    for part in stream:
        msg = _field(part, "message", {})
        chunk = _field(msg, "content", "") or ""
        answer += chunk
        print(chunk, end="", flush=True)
    print()
    if answer:
        history.append({"role": "assistant", "content": answer})
    return history


def drop_dangling(history):
    """Убирает «висящий» вопрос из истории после ошибки, чтобы не ломать контекст."""
    if history and isinstance(history[-1], dict) and history[-1].get("role") == "user":
        history.pop()
    return history


# ==================== ТЕКСТОВЫЙ РЕЖИМ ====================
def main_text():
    global MODEL
    print(f"Алиса (текст) | модель: {MODEL} | контекст: {NUM_CTX}")
    print("Пишите вопрос. Команды: /модель, выход\n")
    history = [{"role": "system", "content": SYSTEM}]

    while True:
        try:
            text = input("Ты: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nПока!")
            break
        if not text:
            continue

        low = text.lower()
        if low in {"выход", "exit", "quit", "стоп"}:
            print("Пока!")
            break

        if low == "/модель" or low.startswith("/модель "):
            parts = text.split(maxsplit=1)
            if len(parts) == 1:
                print("Скачанные модели:")
                for name in list_models():
                    mark = "  <- текущая" if name.startswith(MODEL) else ""
                    print(f"  - {name}{mark}")
            else:
                MODEL = parts[1].strip()
                history = [{"role": "system", "content": SYSTEM}]
                print(f"Переключилась на {MODEL}, история очищена.")
            continue

        print("Алиса: ", end="", flush=True)
        try:
            history = ask(history, text)
        except Exception as e:
            print(f"[Ошибка: {e}]")
            print("Проверь, что Ollama запущена (значок в трее).")
            drop_dangling(history)
        print()


# ==================== ГОЛОСОВОЙ РЕЖИМ ====================
def find_voice(name):
    """Ищет файл голоса Piper: в папке скрипта, cwd, кэше и домашней папке."""
    fname = f"{name}.onnx"
    small_roots = [Path.cwd(), Path(__file__).resolve().parent]
    envd = os.environ.get("PIPER_VOICE_PATH")
    if envd:
        small_roots.append(Path(envd))
    small_roots.append(Path.home() / ".cache" / "piper")
    local = os.environ.get("LOCALAPPDATA")
    if local:
        small_roots.append(Path(local) / "piper")
    for root in small_roots:
        try:
            if not root or not root.exists():
                continue
            direct = root / fname
            if direct.exists():
                return direct
            for p in root.rglob(fname):
                return p
        except OSError:
            continue
    p = Path.home() / fname
    return p if p.exists() else None


def calibrate_threshold():
    """Меряет шум комнаты и выставляет порог срабатывания на речь."""
    print("Калибровка микрофона — помолчи полсекунды...")
    try:
        data = sd.rec(int(0.6 * SR), samplerate=SR, channels=1, dtype="float32")
        sd.wait()
        rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
    except Exception:
        rms = 0.0
    th = max(0.010, min(rms * 3.0, 0.05))
    print(f"Порог: {th:.4f}")
    return th


def listen(threshold):
    """Слушает микрофон. Возвращает речь (float32, 16 кГц) или None при тишине."""
    block = int(SR * 0.03)
    pre = []                 # запас до начала речи, чтобы не срезать первое слово
    buf = []
    started = False
    quiet = 0
    t0 = time.time()
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=block) as st:
        while True:
            data, _ = st.read(block)
            data = data.reshape(-1)
            rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
            now = time.time() - t0

            if not started:
                pre.append(data.copy())
                if len(pre) > 10:
                    pre.pop(0)
                if rms >= threshold:
                    started = True
                    buf.extend(pre)
                    quiet = 0
                elif now >= WAIT_S:
                    return None
                continue

            buf.append(data.copy())
            if rms >= threshold:
                quiet = 0
            else:
                quiet += 1
            if quiet * 0.03 >= SILENCE_S and len(buf) > 8:
                break
            if now >= MAX_S:
                break

    audio = np.concatenate(buf) if buf else None
    if audio is not None and audio.size < SR // 4:
        return None  # слишком короткий всплеск — это шум, а не речь
    return audio


def transcribe(model, audio):
    segments, _info = model.transcribe(
        audio,
        language="ru",
        vad_filter=True,
        beam_size=5,
        no_speech_threshold=0.5,
    )
    return "".join(s.text for s in segments).strip()


def say(voice, text):
    """Озвучивает текст голосом Piper и ждёт конца воспроизведения."""
    try:
        with wave.open(str(REPLY_WAV), "wb") as f:
            voice.synthesize_wav(text, f)
        winsound.PlaySound(str(REPLY_WAV), winsound.SND_FILENAME | winsound.SND_NODEFAULT)
    except Exception as e:
        print(f"[Не удалось озвучить: {e}]")


def main_voice():
    print("=== Голосовая Алиса ===")
    if not VOICE_OK:
        print(f"Голосовой стек не установлен: {VOICE_ERR}")
        print("Выполни:  pip install faster-whisper piper-tts sounddevice numpy")
        return 1

    try:
        ollama.list()
        print("Ollama: ок")
    except Exception as e:
        print(f"ВНИМАНИЕ: Ollama недоступна ({e}). Запусти её и перезапусти скрипт.")

    onnx = find_voice(VOICE_NAME)
    if onnx is None:
        print(f"Голос {VOICE_NAME} не найден. Выполни:")
        print(f"  python -m piper.download_voices {VOICE_NAME}")
        return 1
    print(f"Голос: {onnx}")

    print(f"Распознавание: whisper '{WHISPER_MODEL}' (первый раз — скачается ~460 МБ)...")
    stt = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    print("Синтез: Piper...")
    tts_voice = PiperVoice.load(str(onnx))

    threshold = calibrate_threshold()
    history = [{"role": "system", "content": SYSTEM}]

    say(tts_voice, "Алиса запущена. Скажи что-нибудь.")
    print("Говори спокойно, я отвечу голосом. Выход — сказать «выход».\n")

    while True:
        print("Слушаю...")
        audio = listen(threshold)
        if audio is None:
            continue

        heard = transcribe(stt, audio)
        if not heard:
            continue
        print(f"Ты: {heard}")

        if heard.lower().strip(" !?.…") in {"выход", "exit"}:
            say(tts_voice, "До встречи.")
            print("Пока!")
            break

        print("Алиса: ", end="", flush=True)
        try:
            history = ask(history, heard)
        except Exception as e:
            print(f"[Ошибка мозга: {e}]")
            print("Проверь, что Ollama запущена.")
            drop_dangling(history)
            continue

        answer = history[-1].get("content", "") if isinstance(history[-1], dict) else ""
        if answer:
            say(tts_voice, answer)


# ==================== ТОЧКА ВХОДА ====================
def main():
    global MODEL
    ap = argparse.ArgumentParser(description="Алиса — локальный голосовой ассистент")
    ap.add_argument("--text", action="store_true", help="текстовый чат вместо голоса")
    ap.add_argument("--model", default=None, help=f"модель Ollama (по умолчанию {MODEL})")
    args = ap.parse_args()
    if args.model:
        MODEL = args.model
    if args.text:
        main_text()
    else:
        main_voice()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПока!")
