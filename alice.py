#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Алиса — локальный голосовой ассистент. Мозг офлайн, ноль рублей.

  python alice.py            GUI-окно чата (по умолчанию, PySide6)
  python alice.py --text     текстовый чат в консоли
  python alice.py --voice    голосовой режим: говоришь — отвечает голосом
  python alice.py --model qwen2.5-coder:7b     другая модель Ollama

Ollama запускается автоматически при старте любого режима.
Стек: Ollama (мозг) + faster-whisper (понимает речь) + Piper (озвучка).
Проверено на: RTX 3070 8 ГБ, Intel i5-12400, 16 ГБ DDR4, Windows.
"""

import argparse
import os
import shutil
import subprocess
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
OLLAMA_START_TIMEOUT = 30              # сколько ждать автозапуск Ollama
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


# ==================== АВТОЗАПУСК OLLAMA ====================
def ollama_alive():
    """Проверяет, отвечает ли локальный сервер Ollama."""
    try:
        ollama.list()
        return True
    except Exception:
        return False


def ollama_env():
    """Env для запускаемого сервера: настройки передаём явно, не из профиля."""
    env = os.environ.copy()
    env["OLLAMA_CONTEXT_LENGTH"] = "8192"
    env["OLLAMA_FLASH_ATTENTION"] = "1"
    return env


def start_ollama_windows(env, status):
    """Windows: сначала tray-приложение, запасной вариант — скрытый ollama serve."""
    local = os.environ.get("LOCALAPPDATA")
    app = Path(local) / "Programs" / "Ollama" / "ollama app.exe" if local else None
    if app is not None and app.exists():
        try:
            subprocess.Popen([str(app)], env=env, close_fds=True)
            status("Запустила Ollama: ollama app.exe.")
            return True
        except Exception as e:
            status(f"Не удалось запустить ollama app.exe: {e}.")
    else:
        status("Не нашла %LOCALAPPDATA%\\Programs\\Ollama\\ollama app.exe.")

    serve = None
    if local:
        candidate = Path(local) / "Programs" / "Ollama" / "ollama.exe"
        if candidate.exists():
            serve = str(candidate)
    if serve is None:
        serve = shutil.which("ollama")
    if not serve:
        status("Не нашла ollama serve: нет ollama.exe в PATH и в папке Ollama.")
        return False

    try:
        startupinfo = None
        creationflags = 0
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            creationflags = subprocess.CREATE_NO_WINDOW
        subprocess.Popen(
            [serve, "serve"],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            startupinfo=startupinfo,
            creationflags=creationflags,
            close_fds=True,
        )
        status("Запустила скрытый процесс ollama serve.")
        return True
    except Exception as e:
        status(f"Не удалось запустить ollama serve: {e}.")
        return False


def wait_for_ollama(timeout=OLLAMA_START_TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ollama_alive():
            return True
        time.sleep(0.5)
    return ollama_alive()


def ensure_ollama(status=print, timeout=OLLAMA_START_TIMEOUT):
    """Проверяет Ollama и при необходимости поднимает её. Возвращает True/False."""
    if ollama_alive():
        status("Ollama: ок.")
        return True

    status("Ollama не отвечает. Запускаю её сама...")
    if os.name != "nt":
        status("Автозапуск настроен для Windows. Запусти Ollama вручную.")
        return False

    if not start_ollama_windows(ollama_env(), status):
        status("Не удалось запустить Ollama. Запусти её вручную и перезапусти Алису.")
        return False

    status(f"Жду готовности Ollama (до {timeout} секунд)...")
    if wait_for_ollama(timeout):
        status("Ollama: ок.")
        return True

    status("Ollama не ответила вовремя. Проверь, что она запущена, и перезапусти Алису.")
    return False


# ==================== ТЕКСТОВЫЙ РЕЖИМ ====================
def main_text():
    global MODEL
    print(f"Алиса (текст) | модель: {MODEL} | контекст: {NUM_CTX}")
    print("Пишите вопрос. Команды: /модель, выход\n")
    ensure_ollama(print)
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

    ensure_ollama(print)

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


# ==================== GUI-РЕЖИМ (PySide6) ====================
def main_gui():
    try:
        from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
        from PySide6.QtGui import QFont, QTextCursor
        from PySide6.QtWidgets import (
            QApplication, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
            QPushButton, QTextEdit, QVBoxLayout, QWidget,
        )
    except Exception as e:
        print(f"Не удалось загрузить PySide6: {e}")
        print("Выполни:  pip install PySide6")
        return 1

    class StartupWorker(QObject):
        status = Signal(str)
        completed = Signal(bool)

        def run(self):
            try:
                ok = ensure_ollama(self.status.emit, OLLAMA_START_TIMEOUT)
            except Exception as e:
                self.status.emit(f"Ошибка при запуске Ollama: {e}")
                ok = False
            self.completed.emit(ok)

    class ChatWorker(QObject):
        chunk = Signal(str)
        completed = Signal(str, bool)

        def __init__(self, history):
            super().__init__()
            self.history = list(history)

        def run(self):
            answer = ""
            try:
                stream = ollama.chat(
                    model=MODEL,
                    messages=self.history,
                    stream=True,
                    options={"num_ctx": NUM_CTX, "temperature": TEMPERATURE},
                )
                for part in stream:
                    msg = _field(part, "message", {})
                    piece = _field(msg, "content", "") or ""
                    if piece:
                        answer += piece
                        self.chunk.emit(piece)
                self.completed.emit(answer, True)
            except Exception as e:
                self.completed.emit(f"Ошибка Ollama: {e}", False)

    class ChatWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Алиса — локальный ассистент")
            self.resize(780, 580)
            self.setMinimumSize(520, 400)

            self.history = [{"role": "system", "content": SYSTEM}]
            self.ollama_ready = False
            self.busy = False
            self.assistant_open = False
            self.startup_thread = None
            self.startup_worker = None
            self.chat_thread = None
            self.chat_worker = None

            central = QWidget(self)
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(10)

            self.chat = QTextEdit(self)
            self.chat.setReadOnly(True)
            self.chat.setFont(QFont("Segoe UI", 11))
            layout.addWidget(self.chat, 1)

            row = QHBoxLayout()
            self.input = QLineEdit(self)
            self.input.setPlaceholderText("Напиши сообщение и нажми Enter...")
            self.input.setFont(QFont("Segoe UI", 11))
            self.input.returnPressed.connect(self.send_message)
            row.addWidget(self.input, 1)

            self.send_button = QPushButton("Отправить", self)
            self.send_button.clicked.connect(self.send_message)
            row.addWidget(self.send_button)
            layout.addLayout(row)

            self.status_label = QLabel("Проверяю Ollama...", self)
            layout.addWidget(self.status_label)

            self.append_status("Алиса запущена. Проверяю Ollama...")
            self.set_busy(True, "Проверяю Ollama...")
            self.input.setFocus()
            QTimer.singleShot(0, self.start_ollama_check)

        @Slot(str)
        def append_status(self, text):
            self.chat.append(f"Система: {text}")
            self.scroll_down()

        def append_user(self, text):
            self.chat.append(f"Ты: {text}")
            self.scroll_down()

        def begin_assistant(self):
            self.chat.append("Алиса: ")
            self.assistant_open = True
            self.scroll_down()

        def finish_assistant(self):
            if self.assistant_open:
                self.chat.append("")
                self.assistant_open = False
                self.scroll_down()

        @Slot(str)
        def append_chunk(self, chunk):
            self.chat.moveCursor(QTextCursor.End)
            self.chat.insertPlainText(chunk)
            self.scroll_down()

        def scroll_down(self):
            self.chat.moveCursor(QTextCursor.End)
            self.chat.ensureCursorVisible()
            bar = self.chat.verticalScrollBar()
            bar.setValue(bar.maximum())

        def set_busy(self, busy, label=None):
            self.busy = busy
            self.input.setEnabled(not busy)
            self.send_button.setEnabled(not busy)
            if label:
                self.status_label.setText(label)

        def start_ollama_check(self):
            self.set_busy(True, "Проверяю Ollama...")
            thread = QThread(self)
            worker = StartupWorker()
            worker.moveToThread(thread)
            worker.status.connect(self.append_status)
            worker.completed.connect(self.on_startup_done)
            worker.completed.connect(thread.quit)
            worker.completed.connect(worker.deleteLater)
            thread.started.connect(worker.run)
            thread.finished.connect(thread.deleteLater)
            self.startup_thread = thread
            self.startup_worker = worker
            thread.start()

        @Slot(bool)
        def on_startup_done(self, ok):
            self.ollama_ready = ok
            if ok:
                self.append_status("Ollama готова. Можно писать.")
                self.set_busy(False, "Ollama: ок")
            else:
                self.append_status("Ollama недоступна. Запусти её вручную и перезапусти Алису.")
                self.set_busy(False, "Ollama: ошибка")

        @Slot()
        def send_message(self):
            if self.busy:
                return
            text = self.input.text().strip()
            if not text:
                return
            self.input.clear()
            if not self.ollama_ready:
                self.append_status("Ollama недоступна. Запусти её и перезапусти Алису.")
                return

            self.append_user(text)
            self.begin_assistant()
            self.history.append({"role": "user", "content": text})
            self.set_busy(True, "Алиса печатает...")

            thread = QThread(self)
            worker = ChatWorker(self.history)
            worker.moveToThread(thread)
            worker.chunk.connect(self.append_chunk)
            worker.completed.connect(self.on_chat_completed)
            worker.completed.connect(thread.quit)
            worker.completed.connect(worker.deleteLater)
            thread.started.connect(worker.run)
            thread.finished.connect(thread.deleteLater)
            self.chat_thread = thread
            self.chat_worker = worker
            thread.start()

        @Slot(str, bool)
        def on_chat_completed(self, payload, ok):
            self.finish_assistant()
            if ok:
                if payload:
                    self.history.append({"role": "assistant", "content": payload})
                self.set_busy(False, "Готово")
            else:
                drop_dangling(self.history)
                self.chat.append(payload)
                self.scroll_down()
                self.set_busy(False, "Ошибка Ollama")
            self.input.setFocus()

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Алиса")
    app.setStyleSheet("""
        QWidget {
            background-color: #1e1e1e;
            color: #e8e8e8;
            font-family: 'Segoe UI', sans-serif;
            font-size: 14px;
        }
        QTextEdit {
            background-color: #252526;
            border: 1px solid #3a3a3a;
            border-radius: 8px;
            padding: 10px;
            selection-background-color: #0e639c;
        }
        QLineEdit {
            background-color: #2d2d30;
            border: 1px solid #3a3a3a;
            border-radius: 8px;
            padding: 9px;
        }
        QPushButton {
            background-color: #0e639c;
            color: #ffffff;
            border: none;
            border-radius: 8px;
            padding: 9px 18px;
        }
        QPushButton:hover { background-color: #1177bb; }
        QPushButton:disabled { background-color: #3a3a3a; color: #888888; }
        QLabel { color: #9cdcfe; }
    """)
    window = ChatWindow()
    window.show()
    return app.exec()


# ==================== ТОЧКА ВХОДА ====================
def main():
    global MODEL
    ap = argparse.ArgumentParser(description="Алиса — локальный голосовой ассистент")
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--text", action="store_true", help="текстовый чат в консоли")
    modes.add_argument("--voice", action="store_true", help="голосовой режим")
    ap.add_argument("--model", default=None, help=f"модель Ollama (по умолчанию {MODEL})")
    args = ap.parse_args()
    if args.model:
        MODEL = args.model
    if args.text:
        return main_text()
    elif args.voice:
        return main_voice()
    return main_gui()


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        print("\nПока!")
