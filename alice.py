#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Алиса v2 — локальный ассистент с GUI, голосом и бесплатными веб-сервисами.

  python alice.py            GUI-окно чата (по умолчанию, PySide6)
  python alice.py --text     текстовый чат в консоли
  python alice.py --voice    голосовой режим
  python alice.py --model qwen2.5-coder:7b     другая модель Ollama

Мозг — локальная Ollama. «Ум» — DeepSeek в отдельном профиле браузера.
Музыка — VK Музыка в том же профиле. Всё офлайн/бесплатно, 0 ₽.
"""

import argparse
import contextlib
import ctypes
import functools
import html
import importlib.util
import json
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import warnings
import urllib.error
import urllib.request
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

# Уже установленные на машине владельца библиотеки; при отсутствии
# используем встроенные fallback'ы, поэтому новых зависимостей не появляется.
try:
    import requests
except Exception:
    requests = None
try:
    import psutil
except Exception:
    psutil = None

# Голосовой стек загружается ТОЛЬКО по кнопке 🎙 или в --voice.
# Ранний import faster_whisper вызывал предупреждение HF Hub сразу при старте GUI.
VOICE_OK = None                 # None = ещё не загружали, True/False = результат
VOICE_ERR = None
VOICE_STACK = {}


def load_voice_stack(need_stt=False, need_tts=False, status=None):
    """Лениво подгружает numpy/sounddevice/faster_whisper/piper по требованию."""
    global VOICE_OK, VOICE_ERR
    if VOICE_OK is False:
        return False
    try:
        if "np" not in VOICE_STACK:
            import numpy as np
            import sounddevice as sd
            import winsound
            VOICE_STACK.update(np=np, sd=sd, winsound=winsound)
        if need_stt and "WhisperModel" not in VOICE_STACK:
            emit_status(status, "Грузим распознавание речи…")
            from faster_whisper import WhisperModel
            VOICE_STACK["WhisperModel"] = WhisperModel
        if need_tts and "PiperVoice" not in VOICE_STACK:
            from piper import PiperVoice
            VOICE_STACK["PiperVoice"] = PiperVoice
        VOICE_OK = True
        VOICE_ERR = None
        return True
    except Exception as e:
        VOICE_OK = False
        VOICE_ERR = e
        console_log(f"Голосовой стек недоступен: {e}", level="WARNING")
        return False


# ==================== НАСТРОЙКИ ====================
ALICE_VERSION = "2.0"
MODEL = "qwen3.5:9b"                  # мозг; запасной вариант: qwen2.5-coder:7b
NUM_CTX = 8192
TEMPERATURE = 0.7
WHISPER_MODEL = "small"
VOICE_NAME = "ru_RU-irina-medium"
SR = 16000
SILENCE_S = 1.2
MAX_S = 20
WAIT_S = 20
REPLY_WAV = Path(__file__).resolve().parent / "_reply.wav"

OLLAMA_START_TIMEOUT = 30             # ожидание автозапуска Ollama
OLLAMA_CALL_TIMEOUT = 5               # таймаут одиночного вызова ollama.list()
LLM_TIMEOUT = 120                     # таймаут ожидания ответа локальной модели
DEEPSEEK_LOGIN_TIMEOUT = 300          # 5 минут на ручной логин
DEEPSEEK_CAPTCHA_TIMEOUT = 180        # 3 минуты на капчу
DEEPSEEK_ANSWER_TIMEOUT = 60          # 1 минута на ответ DeepSeek
BROWSER_SETUP_TIMEOUT = 300           # 5 минут на настройку профиля
VK_LOGIN_TIMEOUT = 300
VK_ACTION_TIMEOUT = 60
HEALTH_INTERVAL_MS = 5000             # индикаторы обновляются раз в 5 секунд

OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
OLLAMA_API_TIMEOUT = 2                # прямой HTTP GET /api/version
OLLAMA_BUG_WAIT = 10                  # «трей жив, API мёртв» — сначала ждём
OLLAMA_POLL_INTERVAL = 0.5            # опрос HTTP при ожидании готовности
OLLAMA_LAUNCH_STEP_TIMEOUT = 5        # лимит на каждый способ запуска
OLLAMA_CLIENT_TIMEOUT = 10            # ollama.exe list сам ждёт ~5 с
LOG_MAX_BYTES = 1024 * 1024           # при открытии обрезаем лог до ~1 МБ
SINGLE_INSTANCE_NAME = r"Local\AliceSingleInstance"

# Включаются последовательно по блокам: 1 → B → C → E → D.
FEATURES = {
    "deepseek": True,
    "vk_music": True,
    "status_bar": True,
    "voice_gui": True,
    "autostart": True,                # /автозагрузка и README настроены
}

OLLAMA_LAST_ERROR = ""
OLLAMA_LAST_DETAILS = []
OLLAMA_TRIED_PATHS = []
OLLAMA_EXIT_CODES = []
OLLAMA_LOG_HANDLES = []
MODEL_MISSING = False
SINGLE_INSTANCE_HANDLE = None
SINGLE_INSTANCE_FD = None
GUI_WINDOW = None
CONSOLE_LAST = {}

# Ограничение показываемых ошибок: без этого сломанный вывод в ленту затапливает консоль и лог.
ERROR_STATE = {}
ERROR_STATE_LOCK = threading.RLock()
ERROR_CONSOLE_LIMIT = 20
ERROR_CONSOLE_INTERVAL = 5.0
ERROR_DISPLAY_DEPTH = 0
ERROR_DISPLAY_LOCK = threading.Lock()
ERROR_DISPLAY_METHODS = frozenset({"append_error", "on_global_error"})

# Архитектурные заготовки: НЕ реализованы в этой итерации.
FUTURE_HOOKS = {
    "wake_word": None,      # openWakeWord вместо постоянного прослушивания
    "long_memory": None,    # nomic-embed-text + заметки
    "auto_router": None,    # итерация 2: локально ↔ DeepSeek
    "yandex_music": None,   # второй музыкальный провайдер
}
AUTO_ROUTER = FUTURE_HOOKS["auto_router"]
MUSIC_PROVIDERS = {"vk": True, "yandex": False}

if os.name == "nt":
    BROWSER_PROFILE = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Alice" / "browser-profile"
else:
    BROWSER_PROFILE = Path.home() / ".alice" / "browser-profile"

SYSTEM = (
    "Ты — Алиса, локальный голосовой ассистент на компьютере владельца. "
    "Отвечай только по-русски, коротко и живо: 1-3 предложения, как голосовой помощник. "
    "Без markdown, без списков, без кода — твой ответ произносится вслух. "
    "Если вопрос непонятен — задай один уточняющий вопрос. "
    "Если не знаешь — честно скажи, что не знаешь."
)

EXIT_WORDS = {"выход", "exit", "quit", "стоп"}


# ==================== ВЕБ-СЕРВИСЫ: ИЗОЛИРОВАННЫЕ СЕЛЕКТОРЫ ====================
# Вся разметка сайтов живет здесь. При обновлении DeepSeek/VK правим только этот блок.
WEB_SELECTORS = {
    "deepseek": {
        "url": "https://chat.deepseek.com/",
        "logged_in": [
            "textarea",
            "[contenteditable='true']",
            "div[role='textbox']",
            "[data-testid='chat-input']",
        ],
        "input": [
            "textarea",
            "[contenteditable='true']",
            "div[role='textbox']",
            "[data-testid='chat-input']",
        ],
        "send": [
            "button[type='submit']",
            "button[aria-label*='Send']",
            "button[aria-label*='Отправ']",
            "[data-testid='send-button']",
        ],
        "answer": [
            "[data-message-author-role='assistant']",
            ".markdown",
            "[class*='markdown']",
            "[class*='answer']",
        ],
        "captcha": [
            "[class*='captcha']",
            "[id*='captcha']",
            "iframe[src*='captcha']",
            "text=Подтвердите",
            "text=Verify",
        ],
    },
    "vk_music": {
        "url": "https://vk.com/audio",
        "logged_in": [
            "[id='top_profile_link']",
            "[class*='TopNavBtn__profile']",
            "[aria-label*='Профиль']",
            "a[href*='/id']",
        ],
        "search_input": [
            "input[type='search']",
            "input[placeholder*='Поиск']",
            "input[aria-label*='Поиск']",
            "[data-testid='search-input']",
        ],
        "search_result": [
            "[class*='audio_row']",
            "[class*='AudioRow']",
            "[data-audio]",
            "div[role='listitem']",
        ],
        "play_button": [
            "[class*='AudioRow__play']",
            "[aria-label*='Воспроизвести']",
            "[aria-label*='Play']",
            "button[title*='Play']",
        ],
        "pause_button": [
            "[class*='pause']",
            "[aria-label*='Пауза']",
            "[aria-label*='Pause']",
        ],
        "next_button": [
            "[class*='next']",
            "[aria-label*='Следующий']",
            "[aria-label*='Next']",
        ],
        "player": [
            "[class*='player']",
            "[id*='player']",
            "[class*='AudioPlayer']",
        ],
    },
    # Заготовка: Яндекс Музыка подключается позже по той же схеме.
    "yandex_music": {
        "enabled": False,
        "url": "https://music.yandex.ru/",
        "search_input": [],
        "search_result": [],
    },
}
DEEPSEEK = WEB_SELECTORS["deepseek"]
VK_MUSIC = WEB_SELECTORS["vk_music"]


# ==================== ЛОГ И ДИАГНОСТИКА OLLAMA ====================
def alice_log_dir():
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        return Path(base) / "Alice"
    return Path.home() / ".alice"


def alice_log_path():
    return alice_log_dir() / "alice.log"


LOG_PATH = alice_log_path()


def log_path_text():
    try:
        return str(alice_log_path())
    except Exception:
        return str(LOG_PATH)


def rotate_log_if_needed():
    """Держит alice.log около LOG_MAX_BYTES даже при шквале повторов."""
    try:
        path = alice_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > LOG_MAX_BYTES:
            size = path.stat().st_size
            keep = LOG_MAX_BYTES // 2
            with path.open("rb") as f:
                f.seek(max(0, size - keep))
                tail = f.read()
            path.write_bytes(b"... log rotated ...\n" + tail)
    except Exception:
        pass


def init_log():
    """Создаёт каталог лога и обрезает старый файл до ~1 МБ при запуске."""
    try:
        rotate_log_if_needed()
        path = alice_log_path()
        with path.open("a", encoding="utf-8") as f:
            f.write(f"=== Алиса {ALICE_VERSION} запущена: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    except Exception as e:
        print(f"[Не удалось открыть лог Алисы: {e}]")


def log_event(message, **fields):
    rotate_log_if_needed()
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}"
    for key, value in fields.items():
        if value not in (None, "", [], {}):
            line += f" | {key}={value}"
    try:
        path = alice_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def console_log(message, level="INFO", key=None, min_interval=0.0):
    """Единый вывод диагностики в консоль PowerShell + дубль в alice.log."""
    try:
        now = time.monotonic()
        if key is not None and min_interval:
            if now - CONSOLE_LAST.get(key, -1e9) < min_interval:
                return
            CONSOLE_LAST[key] = now
        line = f"[{time.strftime('%H:%M:%S')}] {level}: {message}"
        print(line, flush=True)
        log_event(f"CONSOLE {level}: {message}")
    except Exception:
        pass


def console_key(text):
    """Убирает '(N c)', чтобы секундные тики не заваливали консоль."""
    return re.sub(r"\s*\(\d+\s*c\)", "", text).strip()


def error_key(text, place=None):
    """Ключ дедупликации: нормализованный текст + место возникновения."""
    normalized = console_key(str(text))
    return f"{place or 'error'}:{normalized}" if place else normalized


def _error_state(key):
    return ERROR_STATE.setdefault(key, {
        "count": 0,
        "console_shown": 0,
        "last_console": -1e9,
        "last_reported": 0,
        "feed_shown": False,
    })


def error_feed_allowed(key):
    """Одинаковая ошибка рисуется в ленте ровно один раз за сессию."""
    with ERROR_STATE_LOCK:
        state = _error_state(key)
        if state["feed_shown"]:
            return False
        state["feed_shown"] = True
        return True


def error_display_active():
    with ERROR_DISPLAY_LOCK:
        return ERROR_DISPLAY_DEPTH > 0


@contextlib.contextmanager
def error_display_guard():
    """Показ ошибки: любые вложенные исключения НЕ должны снова пытаться писать в ленту."""
    global ERROR_DISPLAY_DEPTH
    with ERROR_DISPLAY_LOCK:
        ERROR_DISPLAY_DEPTH += 1
    try:
        yield
    finally:
        with ERROR_DISPLAY_LOCK:
            ERROR_DISPLAY_DEPTH = max(0, ERROR_DISPLAY_DEPTH - 1)


def console_log_error(message, key=None, trace=None, origin=None):
    """Консоль/лог для ошибок: дедуп, «×N повторов», потолок 20 строк одного типа."""
    try:
        key = key or error_key(message, origin)
        now = time.monotonic()
        with ERROR_STATE_LOCK:
            state = _error_state(key)
            state["count"] += 1
            count = state["count"]
            if state["console_shown"] >= ERROR_CONSOLE_LIMIT:
                log_event("ERROR SUPPRESSED", key=key, count=count, detail=message)
                return False
            if count == 1:
                if trace:
                    print(trace, file=sys.stderr, flush=True)
                console_log(str(message), level="ERROR")
                log_event(
                    "EXCEPTION" if trace else "ERROR",
                    origin=origin, detail=message, traceback=trace,
                )
                state["console_shown"] = 1
                state["last_console"] = now
                state["last_reported"] = 1
                return True
            if now - state["last_console"] < ERROR_CONSOLE_INTERVAL:
                log_event("ERROR REPEAT SUPPRESSED", key=key, count=count)
                return False
            repeats = count - state["last_reported"]
            line = f"{message} ×{repeats} повторов"
            console_log(line, level="ERROR")
            log_event("ERROR REPEAT", key=key, repeats=repeats, detail=message)
            state["console_shown"] += 1
            state["last_console"] = now
            state["last_reported"] = count
            return True
    except Exception:
        return False


def report_feed_failure(exc, where="_feed_append_html"):
    """Падение самого вывода в ленту: только консоль + лог, без новой попытки GUI."""
    try:
        trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        text = f"Не удалось вывести строку в ленту ({where}): {exc}. Подробности: {log_path_text()}"
        console_log_error(text, key=error_key(text, where), trace=trace, origin="feed")
    except Exception:
        pass


def install_warning_hooks():
    """Python-warnings идут в консоль и лог, а не теряются."""
    def showwarning(message, category, filename, lineno, file=None, line=None):
        console_log(
            f"{category.__name__}: {message} ({filename}:{lineno})",
            level="WARNING",
            key=f"warn:{category.__name__}:{message}",
            min_interval=1.0,
        )
    warnings.showwarning = showwarning


def report_exception(exc_type, exc, tb, origin="приложение"):
    """Общий обработчик: traceback в консоль/лог + одна красная строка в GUI."""
    try:
        trace = "".join(traceback.format_exception(exc_type, exc, tb))
        short = f"Перехвачена ошибка ({origin}): {exc}. Подробности: {log_path_text()}"
        console_log_error(short, key=error_key(short), trace=trace, origin=origin)
        # Внутри показа ошибки не испускаем error_signal: иначе append_error -> report -> append_error.
        if (
            GUI_WINDOW is not None
            and not error_display_active()
            and "on_global_error" not in origin
            and "append_error" not in origin
            and "feed" not in origin
        ):
            try:
                GUI_WINDOW.error_signal.emit(short)
            except Exception:
                pass
    except Exception:
        pass


def install_exception_hooks():
    """sys.excepthook + threading.excepthook, без завершения процесса."""
    def main_hook(exc_type, exc, tb):
        report_exception(exc_type, exc, tb, "главный поток")

    def thread_hook(args):
        name = getattr(getattr(args, "thread", None), "name", "?")
        report_exception(
            args.exc_type, args.exc_value, args.exc_traceback,
            f"поток {name}",
        )

    sys.excepthook = main_hook
    threading.excepthook = thread_hook


def log_tail(limit=8):
    try:
        lines = alice_log_path().read_text(encoding="utf-8", errors="replace").splitlines()
        return " | ".join(line.strip() for line in lines[-limit:])
    except Exception:
        return ""


def with_log_path(text):
    if "alice.log" in text:
        return text
    return f"{text} Подробности: {log_path_text()}"


def ensure_status(status, text, started=None):
    """Статус в ленту с таймером + лог + консоль (секундные тики с throttle)."""
    emit_status(status, text, started)
    log_event(f"СТАТУС: {text}")
    console_log(text, level="INFO", key=console_key(text), min_interval=1.0)


def ollama_api_get(path, timeout=OLLAMA_API_TIMEOUT):
    """Прямой HTTP GET к локальному API; не зависит от клиентской библиотеки."""
    url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}{path}"
    if requests is not None:
        try:
            response = requests.get(url, timeout=timeout)
            code = getattr(response, "status_code", 200)
            if code != 200:
                return False, None, f"HTTP {code}"
            try:
                return True, response.json(), None
            except Exception:
                return True, getattr(response, "text", ""), None
        except Exception as e:
            return False, None, str(e)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            try:
                return True, json.loads(raw), None
            except Exception:
                return True, raw, None
    except Exception as e:
        return False, None, str(e)


def ollama_api_alive(log=False):
    ok, payload, error = ollama_api_get("/api/version", OLLAMA_API_TIMEOUT)
    if log:
        log_event("HTTP /api/version", ok=ok, error=error, version=payload if ok else None)
    return ok


class OllamaProc:
    def __init__(self, pid, name, exe="", cmdline=None, source="psutil"):
        self.pid = int(pid)
        self.name = name or ""
        self.exe = exe or ""
        self.cmdline = list(cmdline or [])
        self.source = source
        low_name = self.name.lower()
        low_exe = self.exe.lower()
        low_cmd = " ".join(self.cmdline).lower()
        self.is_tray = "ollama app" in low_name or "ollama app" in low_exe
        self.is_server = (
            "serve" in low_cmd
            or "ollamaserver" in low_name
            or "ollamaserver" in low_exe
            or (low_name.startswith("ollama.exe") and "serve" in low_cmd)
        )

    def describe(self):
        role = "трей" if self.is_tray else ("сервер" if self.is_server else "процесс")
        return f"PID {self.pid} {self.name} ({role})"

    def __repr__(self):
        return self.describe()


def ollama_processes():
    """Ищет все ollama* процессы: psutil, а при его отсутствии — tasklist."""
    result = []
    if psutil is not None:
        try:
            for proc in psutil.process_iter(["pid", "name", "exe", "cmdline"]):
                info = getattr(proc, "info", {}) or {}
                name = str(info.get("name") or "")
                exe = str(info.get("exe") or "")
                cmdline = [str(x) for x in (info.get("cmdline") or [])]
                joined = " ".join([name, exe] + cmdline).lower()
                if "ollama" in joined:
                    result.append(OllamaProc(info.get("pid") or 0, name, exe, cmdline, "psutil"))
            return result
        except Exception as e:
            log_event("psutil process_iter failed", error=e)
    if os.name == "nt":
        try:
            raw = subprocess.check_output(
                ["tasklist", "/FO", "CSV", "/NH"], text=True, timeout=5
            )
            for line in raw.splitlines():
                parts = [x.strip('"') for x in line.split('","')]
                if len(parts) >= 2 and "ollama" in parts[0].lower():
                    try:
                        pid = int(parts[1])
                    except Exception:
                        continue
                    result.append(OllamaProc(pid, parts[0], "", [], "tasklist"))
        except Exception as e:
            log_event("tasklist failed", error=e)
    return result


def format_processes(procs):
    if not procs:
        return "нет"
    return "; ".join(p.describe() for p in procs[:8])


def ollama_install_dirs():
    roots = []
    candidates = []
    if os.name == "nt":
        for var in ("LOCALAPPDATA", "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
            value = os.environ.get(var)
            if value:
                base = Path(value)
                candidates.extend([base / "Programs" / "Ollama", base / "Ollama"])
    else:
        value = os.environ.get("LOCALAPPDATA")
        if value:
            candidates.append(Path(value) / "Programs" / "Ollama")
        candidates.append(Path.home() / ".local" / "share" / "ollama")
    found = shutil.which("ollama")
    if found:
        candidates.append(Path(found).resolve().parent)
    for candidate in candidates:
        try:
            if candidate.is_dir() and str(candidate) not in roots:
                roots.append(str(candidate))
        except Exception:
            continue
    return roots


def find_ollama_exes():
    """Динамически перечисляет каталог установки, а не хардкодит одно имя."""
    dirs = ollama_install_dirs()
    all_exes = []
    for dirname in dirs:
        try:
            for exe in sorted(Path(dirname).glob("*.exe")):
                if "ollama" in exe.name.lower() and str(exe) not in all_exes:
                    all_exes.append(str(exe))
        except Exception as e:
            log_event("Не удалось перечислить каталог Ollama", path=dirname, error=e)
    app = None
    cli = None
    server = None
    for item in all_exes:
        low = Path(item).name.lower()
        if app is None and ("ollama app" in low or ("app" in low and "ollama" in low)):
            app = item
        elif server is None and "server" in low and "ollama" in low:
            server = item
        elif cli is None and "ollama" in low and "app" not in low and "server" not in low:
            cli = item
    return {"dirs": dirs, "all": all_exes, "app": app, "cli": cli, "server": server}


def port_in_use(host=OLLAMA_HOST, port=OLLAMA_PORT):
    """Порт занят, даже если HTTP на него не отвечает."""
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False
    except Exception as e:
        log_event("Проверка порта не удалась", error=e)
        return False


def windows_spawn_flags():
    startupinfo = None
    flags = 0
    if os.name == "nt":
        try:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
            startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
        except Exception:
            startupinfo = None
        flags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
        if not flags:
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return startupinfo, flags


def open_child_log_handle():
    path = alice_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", encoding="utf-8")
    OLLAMA_LOG_HANDLES.append(handle)
    return handle


def spawn_ollama_process(args, label, status):
    ensure_status(status, f"Запускаю {label}…")
    handle = open_child_log_handle()
    startupinfo, flags = windows_spawn_flags()
    kwargs = dict(
        cwd=str(Path(args[0]).parent),
        env=ollama_env(),
        stdin=subprocess.DEVNULL,
        stdout=handle,
        stderr=subprocess.STDOUT,
        close_fds=True,
    )
    if startupinfo is not None:
        kwargs["startupinfo"] = startupinfo
    if flags:
        kwargs["creationflags"] = flags
    proc = subprocess.Popen(args, **kwargs)
    pid = getattr(proc, "pid", "?")
    OLLAMA_TRIED_PATHS.append(str(args[0]))
    log_event("SPAWN", label=label, pid=pid, cmd=" ".join(args))
    ensure_status(status, f"Ollama запущена (PID {pid})")
    return proc


def wait_launch_for_api(proc, timeout, status, label):
    started = time.monotonic()
    last_second = 0
    last_process_second = -1
    ensure_status(status, "Жду Ollama…", started)
    while True:
        if ollama_api_alive(log=True):
            return True, "ok"
        code = None
        try:
            code = proc.poll()
        except Exception:
            pass
        if code is not None:
            OLLAMA_EXIT_CODES.append(f"{label}: exit={code}")
            return False, f"exit={code}"
        elapsed = time.monotonic() - started
        if elapsed >= timeout:
            OLLAMA_EXIT_CODES.append(f"{label}: timeout={int(timeout)}c")
            return False, "timeout"
        second = int(elapsed)
        if second != last_second:
            last_second = second
            ensure_status(status, "Жду Ollama…", started)
            if second % 5 == 0 and second != last_process_second:
                last_process_second = second
                procs = ollama_processes()
                ensure_status(status, f"Процессы Ollama: {format_processes(procs)}")
                log_event("Процессы Ollama", processes=format_processes(procs))
        time.sleep(min(OLLAMA_POLL_INTERVAL, max(0.05, timeout - elapsed)))


def kill_ollama_processes(status):
    """Сначала убиваем трей, иначе app.exe сразу перезапустит serve (#14761)."""
    procs = ollama_processes()
    ordered = sorted(procs, key=lambda p: 0 if p.is_tray else 1)
    for proc in ordered:
        log_event("KILL", pid=proc.pid, name=proc.name, cmd=" ".join(proc.cmdline), source=proc.source)
        try:
            if psutil is not None and proc.source == "psutil":
                real = psutil.Process(proc.pid)
                real.terminate()
            else:
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                    check=False,
                )
        except Exception as e:
            log_event("KILL terminate failed", pid=proc.pid, error=e)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        left = ollama_processes()
        if not left:
            break
        time.sleep(0.2)
    for proc in ollama_processes():
        log_event("KILL -9", pid=proc.pid, name=proc.name)
        try:
            if psutil is not None and proc.source == "psutil":
                psutil.Process(proc.pid).kill()
        except Exception as e:
            log_event("KILL -9 failed", pid=proc.pid, error=e)
    ensure_status(status, f"Перезапускаю пару процессов Ollama: {format_processes(ordered)}")
    log_event("KILLED", pids=",".join(str(p.pid) for p in ordered))


def wait_for_ollama(status=None, timeout=None, message="Жду Ollama…"):
    """Опрос /api/version каждые 0.5 с; статус раз в секунду, процессы раз в 5 с."""
    if timeout is None:
        timeout = OLLAMA_START_TIMEOUT
    started = time.monotonic()
    last_second = 0
    last_process_second = -1
    ensure_status(status, message, started)
    while True:
        if ollama_api_alive(log=True):
            return True
        elapsed = time.monotonic() - started
        if elapsed >= timeout:
            return False
        second = int(elapsed)
        if second != last_second:
            last_second = second
            ensure_status(status, message, started)
            if second % 5 == 0 and second != last_process_second:
                last_process_second = second
                procs = ollama_processes()
                ensure_status(status, f"Процессы Ollama: {format_processes(procs)}")
                log_event("Процессы Ollama", processes=format_processes(procs))
        time.sleep(min(OLLAMA_POLL_INTERVAL, max(0.05, timeout - elapsed)))


def launch_ollama_sequence(status, deadline):
    exes = find_ollama_exes()
    log_event("Каталоги Ollama", dirs=";".join(exes["dirs"]) or "не найдены")
    log_event("Найденные exe", exes=";".join(exes["all"]) or "не найдены")
    ensure_status(status, f"Каталоги Ollama: {', '.join(exes['dirs']) or 'не найдены'}")
    ensure_status(status, f"Найденные exe: {', '.join(exes['all']) or 'нет'}")
    if not (exes["app"] or exes["cli"] or exes["server"]):
        return False, f"Ollama не установлена: каталоги {', '.join(exes['dirs']) or 'не найдены'}."

    steps = []
    if exes["app"]:
        steps.append(([exes["app"]], "ollama app.exe", OLLAMA_LAUNCH_STEP_TIMEOUT))
    if exes["cli"]:
        steps.append(([exes["cli"], "serve"], "ollama.exe serve", OLLAMA_LAUNCH_STEP_TIMEOUT))
    if exes["server"]:
        steps.append(([exes["server"]], "OllamaServer.exe", OLLAMA_LAUNCH_STEP_TIMEOUT))
    if exes["cli"]:
        steps.append(([exes["cli"], "list"], "ollama.exe list", OLLAMA_CLIENT_TIMEOUT))

    for args, label, step_timeout in steps:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        step_timeout = min(step_timeout, remaining)
        try:
            proc = spawn_ollama_process(args, label, status)
        except Exception as e:
            OLLAMA_EXIT_CODES.append(f"{label}: spawn_error={e}")
            log_event("SPAWN failed", label=label, error=e)
            continue
        ok, detail = wait_launch_for_api(proc, step_timeout, status, label)
        log_event("STEP", label=label, ok=ok, detail=detail)
        if ok:
            return True, None

    remaining = deadline - time.monotonic()
    if remaining > 0 and wait_for_ollama(status, remaining, "Жду Ollama…"):
        return True, None
    return False, "все способы запуска исчерпаны"


def ollama_model_available():
    ok, payload, error = ollama_api_get("/api/tags", OLLAMA_API_TIMEOUT)
    if not ok:
        return False, False, error
    names = []
    if isinstance(payload, dict):
        for item in payload.get("models", []) or []:
            if isinstance(item, dict):
                name = item.get("model") or item.get("name")
                if name:
                    names.append(str(name))
    present = any(name.startswith(MODEL) or MODEL.startswith(name) for name in names)
    log_event("HTTP /api/tags", present=present, models=",".join(names) or "нет")
    return True, present, None


def finish_ollama_success(status):
    global MODEL_MISSING
    ensure_status(status, "Ollama готова.")
    api_ok, present, error = ollama_model_available()
    MODEL_MISSING = api_ok and not present
    if MODEL_MISSING:
        ensure_status(status, f"Модель не скачана: выполните ollama pull {MODEL}")
    elif api_ok:
        ensure_status(status, f"Модель {MODEL} найдена.")
    else:
        log_event("/api/tags недоступен", error=error)
    return True


def fail_ollama(status, reason):
    global OLLAMA_LAST_ERROR
    OLLAMA_LAST_ERROR = with_log_path(reason)
    OLLAMA_LAST_DETAILS.append(reason)
    ensure_status(status, OLLAMA_LAST_ERROR)
    log_event("ERROR", reason=reason)
    return False


def build_ollama_failure_reason():
    paths = ", ".join(OLLAMA_TRIED_PATHS) or "не найдены"
    codes = "; ".join(OLLAMA_EXIT_CODES) or "нет"
    tail = log_tail(6) or "лог пуст"
    return (
        f"Не удалось запустить Ollama. Пути: {paths}. "
        f"Коды выхода: {codes}. Stderr/лог: {tail}"
    )


def acquire_single_instance():
    """Мьютекс Windows / flock на Linux: второй экземпляр не ломает первый."""
    global SINGLE_INSTANCE_HANDLE, SINGLE_INSTANCE_FD
    if os.environ.get("ALICE_ALLOW_MULTIPLE") == "1":
        return True, None
    if os.name == "nt":
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
            kernel32.CreateMutexW.restype = ctypes.c_void_p
            handle = kernel32.CreateMutexW(None, 0, SINGLE_INSTANCE_NAME)
            if not handle:
                log_event("CreateMutexW failed", error=ctypes.get_last_error())
                return True, None
            if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
                return False, "Алиса уже запущена"
            SINGLE_INSTANCE_HANDLE = handle
            return True, None
        except Exception as e:
            log_event("Мьютекс недоступен", error=e)
            return True, None
    try:
        import fcntl
        path = Path("/tmp") / "alice-single-instance.lock"
        fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        SINGLE_INSTANCE_FD = fd
        return True, None
    except BlockingIOError:
        return False, "Алиса уже запущена"
    except Exception as e:
        log_event("flock недоступен", error=e)
        return True, None


# ==================== ОБЩИЕ ПОМОЩНИКИ ====================
def emit_status(status, text, started=None):
    """Отправляет статус; если передано начало отсчёта — добавляет таймер."""
    if status is None:
        return
    if started is None:
        status(text)
    else:
        status(f"{text} ({int(time.monotonic() - started)} c)")


def call_with_timeout(fn, timeout, default=None):
    """Вызывает fn в отдельном потоке и не ждёт дольше timeout секунд."""
    out = queue.Queue(maxsize=1)

    def runner():
        try:
            out.put((True, fn()))
        except Exception as e:
            out.put((False, e))

    threading.Thread(target=runner, daemon=True).start()
    try:
        return out.get(timeout=timeout)
    except queue.Empty:
        return False, TimeoutError(f"Превышен таймаут {timeout} сек.")


def poll_until(predicate, timeout, status, message, interval=1.0):
    """Ждёт predicate, каждую секунду показывая живой статус с таймером."""
    started = time.monotonic()
    last_second = -1
    while True:
        try:
            if predicate():
                return True
        except Exception:
            pass
        elapsed = time.monotonic() - started
        if elapsed >= timeout:
            return False
        second = int(elapsed)
        if second != last_second:
            last_second = second
            emit_status(status, message, started)
        time.sleep(min(interval, max(0.05, timeout - elapsed)))


def _field(obj, name, default=None):
    """Достаёт поле из dict или объекта (разные версии библиотеки ollama)."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def ollama_alive():
    """Живость определяем прямым HTTP GET /api/version, а не ollama.list()."""
    return ollama_api_alive(log=False)


def list_models():
    ok, payload, error = ollama_api_get("/api/tags", OLLAMA_API_TIMEOUT)
    if not ok:
        log_event("/api/tags недоступен", error=error)
        return []
    names = []
    if isinstance(payload, dict):
        for item in payload.get("models", []) or []:
            if isinstance(item, dict):
                name = item.get("model") or item.get("name")
                if name:
                    names.append(str(name))
    return names


def generate_answer(history, timeout=None, on_chunk=None, on_status=None):
    """
    Стриминг ответа Ollama с общим таймаутом. Возвращает (ответ, текст_ошибки).
    Работает и в GUI-потоке, и в консоли: UI не блокируется.
    """
    if timeout is None:
        timeout = LLM_TIMEOUT
    out = queue.Queue()

    def producer():
        try:
            stream = ollama.chat(
                model=MODEL,
                messages=list(history),
                stream=True,
                options={"num_ctx": NUM_CTX, "temperature": TEMPERATURE},
            )
            for part in stream:
                msg = _field(part, "message", {})
                chunk = _field(msg, "content", "") or ""
                if chunk:
                    out.put(("chunk", chunk))
            out.put(("done", ""))
        except Exception as e:
            out.put(("error", str(e)))

    threading.Thread(target=producer, daemon=True).start()
    started = time.monotonic()
    last_second = -1
    answer = ""
    while True:
        elapsed = time.monotonic() - started
        if elapsed >= timeout:
            return "", f"Таймаут ответа модели ({timeout} сек). Проверь Ollama и попробуй ещё раз."
        second = int(elapsed)
        if second != last_second:
            last_second = second
            if on_status:
                on_status(f"Думаю… ({second} c)")
        try:
            kind, payload = out.get(timeout=0.5)
        except queue.Empty:
            continue
        if kind == "chunk":
            answer += payload
            if on_chunk:
                on_chunk(payload)
        elif kind == "done":
            return answer, None
        elif kind == "error":
            return "", f"Ошибка Ollama: {payload}"


def ask(history, text):
    """Консольный/голосовой запрос к локальному мозгу с таймаутом."""
    history.append({"role": "user", "content": text})
    answer, error = generate_answer(
        history,
        LLM_TIMEOUT,
        on_chunk=lambda c: print(c, end="", flush=True),
        on_status=None,
    )
    print()
    if error:
        drop_dangling(history)
        raise RuntimeError(error)
    if answer:
        history.append({"role": "assistant", "content": answer})
    return history


def drop_dangling(history):
    """Убирает «висящий» вопрос из истории после ошибки."""
    if history and isinstance(history[-1], dict) and history[-1].get("role") == "user":
        history.pop()
    return history


def parse_music_query(query):
    for sep in (" — ", " - ", " – ", " —", " -"):
        if sep in query:
            artist, track = query.split(sep, 1)
            return artist.strip(), track.strip()
    return query.strip(), None


def classify_command(text):
    """Единый слой разбора команд для GUI, --text и --voice."""
    raw = text.strip()
    low = raw.lower()

    if low in EXIT_WORDS:
        return {"kind": "exit", "text": raw, "display": raw}
    if low == "/модель":
        return {"kind": "model_list", "text": raw, "display": raw}
    if low.startswith("/модель "):
        model = raw.split(maxsplit=1)[1].strip()
        return {"kind": "model_set", "model": model, "text": raw, "display": raw}
    if low == "/статус":
        return {"kind": "status", "text": raw, "display": raw}
    if low in {"/старт", "/повторить", "/рестарт"}:
        return {"kind": "restart_ollama", "text": raw, "display": raw}
    if low == "/автозагрузка":
        if not FEATURES["autostart"]:
            return {"kind": "unavailable", "reason": "Настройка автозагрузки появится в следующем блоке.", "text": raw, "display": raw}
        return {"kind": "autostart", "text": raw, "display": raw}
    if low == "/браузер":
        if not (FEATURES["deepseek"] or FEATURES["vk_music"]):
            return {"kind": "unavailable", "reason": "Настройка браузера появится в следующем блоке.", "text": raw, "display": raw}
        return {"kind": "browser_setup", "text": raw, "display": raw}

    if low.startswith("д:") or low.startswith("d:"):
        query = raw.split(":", 1)[1].strip()
        if not FEATURES["deepseek"]:
            return {"kind": "unavailable", "reason": "DeepSeek появится в следующем блоке.", "text": raw, "display": raw}
        if not query:
            return {"kind": "unavailable", "reason": "После «д:» нужен вопрос.", "text": raw, "display": raw}
        return {"kind": "deepseek", "query": query, "text": raw, "display": raw}

    if low in {"пауза", "пауза музыки"}:
        music = {"kind": "music", "action": "pause", "query": "", "text": raw, "display": raw}
    elif low in {"продолжи", "продолжить", "продолжи музыку"}:
        music = {"kind": "music", "action": "resume", "query": "", "text": raw, "display": raw}
    elif low in {"дальше", "следующий трек", "дальше музыку"}:
        music = {"kind": "music", "action": "next", "query": "", "text": raw, "display": raw}
    elif low in {"выключи музыку", "выключить музыку", "останови музыку"}:
        music = {"kind": "music", "action": "stop", "query": "", "text": raw, "display": raw}
    elif low.startswith("включи "):
        query = raw.split(maxsplit=1)[1].strip()
        music = {"kind": "music", "action": "play", "query": query, "text": raw, "display": raw}
    else:
        music = None
    if music:
        if not FEATURES["vk_music"]:
            return {"kind": "unavailable", "reason": "Управление музыкой появится в следующем блоке.", "text": raw, "display": raw}
        return music

    # Итерация 2: AUTO_ROUTER сам решит, уходить ли в DeepSeek. Сейчас выключен.
    if AUTO_ROUTER is not None and AUTO_ROUTER(raw):
        return {"kind": "deepseek", "query": raw, "text": raw, "display": raw}

    return {"kind": "local", "text": raw, "display": raw}


def state_word(state):
    return {"ok": "ок", "warn": "проверяю", "err": "ошибка"}.get(state, "неизвестно")


def status_text(history_len=0, ollama="warn", model="warn", browser="warn"):
    text = (
        f"Алиса {ALICE_VERSION}. Модель: {MODEL} ({state_word(model)}). "
        f"Ollama: {state_word(ollama)}. Браузер: {state_word(browser)}. "
        f"История: {history_len} сообщений."
    )
    if model == "warn" and MODEL_MISSING:
        text += f" Модель не скачана: выполните ollama pull {MODEL}."
    if ollama == "err" and OLLAMA_LAST_ERROR:
        text += f" {OLLAMA_LAST_ERROR}"
    return text


def playwright_available():
    try:
        return importlib.util.find_spec("playwright.sync_api") is not None
    except Exception:
        return False


def health_snapshot(history_len=0, browser_override=None):
    """Один снимок здоровья для индикаторов и /статус."""
    global MODEL_MISSING
    ollama_ok = ollama_api_alive(log=False)
    models = list_models() if ollama_ok else []
    model_ok = any(m.startswith(MODEL) or MODEL.startswith(m) for m in models)
    MODEL_MISSING = ollama_ok and not model_ok
    if browser_override == "err":
        browser_state = "err"
    elif playwright_available() and BROWSER_PROFILE.exists():
        browser_state = "ok"
    elif playwright_available():
        browser_state = "warn"
    else:
        browser_state = "err"
    states = {
        "ollama": "ok" if ollama_ok else "err",
        "model": "ok" if model_ok else ("warn" if ollama_ok else "err"),
        "browser": browser_state,
    }
    return states, status_text(history_len, **states)


# ==================== НАДЁЖНЫЙ ЗАПУСК OLLAMA ====================
PORT_BUSY_ERROR = (
    "Порт 11434 занят, но Ollama не отвечает — перезапусти Ollama "
    "(или нажми Повторить)"
)


def ollama_env():
    """Env для запускаемого сервера: настройки передаём явно, не из профиля."""
    env = os.environ.copy()
    env["OLLAMA_CONTEXT_LENGTH"] = "8192"
    env["OLLAMA_FLASH_ATTENTION"] = "1"
    return env


def ensure_ollama(status=print, timeout=None):
    """
    Единый алгоритм для GUI, --text и --voice.

    Учитывает баги Windows-сборки Ollama:
    - #12691: ollama app.exe в трее есть, а ollama.exe serve не поднялся;
    - #14761: app.exe сам перезапускает serve, поэтому убивать надо пару.
    """
    global OLLAMA_LAST_ERROR
    if timeout is None:
        timeout = OLLAMA_START_TIMEOUT
    OLLAMA_LAST_ERROR = ""
    OLLAMA_LAST_DETAILS.clear()
    OLLAMA_TRIED_PATHS.clear()
    OLLAMA_EXIT_CODES.clear()
    log_event("ENSURE START", timeout=timeout, model=MODEL)

    ensure_status(status, "Проверяю Ollama: HTTP /api/version…")
    ok, payload, error = ollama_api_get("/api/version", OLLAMA_API_TIMEOUT)
    if ok:
        log_event("HTTP OK", version=payload)
        alive_procs = ollama_processes()
        if alive_procs:
            pids = ", ".join(str(p.pid) for p in alive_procs)
            ensure_status(status, f"Ollama запущена (PID {pids})")
        ensure_status(status, "Ollama: ок (HTTP API отвечает).")
        ensure_status(status, "Лишние процессы не запускаю.")
        return finish_ollama_success(status)

    log_event("HTTP FAIL", error=error)
    ensure_status(status, "Ollama API не отвечает. Смотрю процессы…")
    procs = ollama_processes()
    if procs:
        pids = ", ".join(str(p.pid) for p in procs)
        ensure_status(status, f"Ollama запущена (PID {pids})")
    ensure_status(status, f"Процессы Ollama: {format_processes(procs)}")
    log_event("Процессы Ollama", processes=format_processes(procs))

    if procs:
        ensure_status(status, "Ollama запущена, но API не отвечает. Жду 10 сек…")
        if wait_for_ollama(status, OLLAMA_BUG_WAIT, "Жду Ollama…"):
            return finish_ollama_success(status)
        ensure_status(
            status,
            "Ollama запущена, но сервер не отвечает (известный баг #12691) — перезапускаю",
        )
        kill_ollama_processes(status)

    if port_in_use() and not ollama_api_alive(log=True):
        return fail_ollama(status, PORT_BUSY_ERROR)

    launch_deadline = time.monotonic() + timeout
    launched, detail = launch_ollama_sequence(status, launch_deadline)
    if launched or ollama_api_alive(log=True):
        return finish_ollama_success(status)

    if port_in_use():
        return fail_ollama(status, PORT_BUSY_ERROR)
    log_event("ENSURE FAIL", detail=detail)
    return fail_ollama(status, build_ollama_failure_reason())


def windows_startup_folder():
    """Папка shell:startup. COM, если есть pywin32; иначе стандартный путь."""
    try:
        import win32com.client
        shell = win32com.client.Dispatch("WScript.Shell")
        folder = shell.SpecialFolders("Startup")
        if folder:
            return Path(str(folder))
    except Exception as e:
        log_event("COM WScript.Shell недоступен", error=e)
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    return Path.home() / "AppData" / "Roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


def read_shortcut_target(lnk):
    try:
        import win32com.client
        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortcut(str(lnk))
        return str(getattr(shortcut, "TargetPath", "") or "")
    except Exception as e:
        log_event("Не удалось прочитать ярлык", path=lnk, error=e)
        return ""


def create_shortcut_com(app, lnk):
    import win32com.client
    shell = win32com.client.Dispatch("WScript.Shell")
    shortcut = shell.CreateShortcut(str(lnk))
    shortcut.TargetPath = str(app)
    shortcut.WorkingDirectory = str(Path(app).parent)
    shortcut.Description = "Ollama — локальный сервер для Алисы"
    shortcut.Save()
    return shortcut


def create_shortcut_powershell(app, lnk):
    """Официальный рецепт WScript.Shell через powershell.exe без pywin32."""
    script = (
        f'$s=(New-Object -ComObject WScript.Shell).CreateShortcut("{lnk}"); '
        f'$s.TargetPath="{app}"; '
        f'$s.WorkingDirectory="{Path(app).parent}"; '
        '$s.Save()'
    )
    kwargs = dict(
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15,
        check=False,
    )
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        **kwargs,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"PowerShell exit={result.returncode}: "
            f"{(result.stderr or result.stdout or '').strip()}"
        )
    return result


def autostart_ollama():
    """
    Создаёт/проверяет ярлык shell:startup строго на ollama app.exe.
    Ярлык на CLI ollama.exe — типичная причина «стоит в автозагрузке, но не работает».
    """
    if os.name != "nt":
        return False, "Автозагрузка настраивается только в Windows."

    exes = find_ollama_exes()
    app = exes["app"]
    if not app:
        dirs = ", ".join(exes["dirs"]) or "не найдены"
        return False, f"Ollama не установлена: каталоги {dirs}."
    if "app" not in Path(app).name.lower():
        return False, f"Ollama не установлена: не найден ollama app.exe среди {', '.join(exes['all'])}."

    startup = windows_startup_folder()
    startup.mkdir(parents=True, exist_ok=True)
    lnk = startup / "Ollama.lnk"
    log_event("AUTOSTART", app=app, shortcut=lnk, startup=startup)

    if lnk.exists():
        target = read_shortcut_target(lnk)
        if target and Path(target).name.lower() == Path(app).name.lower():
            return True, f"Ярлык уже есть: {lnk}"
        log_event("Ярлык указывает не туда", current=target, expected=app)

    try:
        create_shortcut_com(app, lnk)
        method = "pywin32 COM"
    except Exception as e:
        log_event("COM-создание ярлыка не удалось", error=e)
        try:
            create_shortcut_powershell(app, lnk)
            method = "PowerShell WScript.Shell"
        except Exception as e2:
            log_event("PowerShell-создание ярлыка не удалось", error=e2)
            return False, f"Не удалось создать ярлык автозагрузки: {e2}"
    log_event("Ярлык создан", method=method, target=app, path=lnk)
    return True, f"Ярлык создан: {lnk}"


def restart_ollama_in_background(status=print):
    """Команда /старт: повторяет ensure, не блокируя консоль/голос."""
    status("Повторяю запуск Ollama в фоне…")
    log_event("RESTART /старт")

    def runner():
        try:
            ensure_ollama(status)
        except Exception as e:
            fail_ollama(status, f"Ошибка при повторном запуске Ollama: {e}")

    threading.Thread(target=runner, daemon=True).start()


# ==================== БРАУЗЕР АССИСТЕНТА (Playwright) ====================
class BrowserSession:
    """Один отдельный профиль браузера на все веб-сценарии."""

    def __init__(self, status=print):
        self.status = status
        self._pw = None
        self.context = None
        self.page = None
        self.vk_page = None

    def open(self):
        if not playwright_available():
            raise RuntimeError("Playwright не установлен. Выполни: pip install playwright")
        from playwright.sync_api import sync_playwright
        BROWSER_PROFILE.mkdir(parents=True, exist_ok=True)
        self._pw = sync_playwright().start()
        self.context = self._pw.chromium.launch_persistent_context(
            user_data_dir=str(BROWSER_PROFILE),
            headless=False,
            args=["--no-first-run", "--no-default-browser-check"],
        )
        self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
        self.page.set_default_timeout(10000)
        self.vk_page = None
        return self.page

    def close(self):
        for closer in (
            lambda: self.context.close() if self.context else None,
            lambda: self._pw.stop() if self._pw else None,
        ):
            try:
                closer()
            except Exception:
                pass
        self._pw = None
        self.context = None
        self.page = None
        self.vk_page = None


def find_locator(page, selectors):
    for sel in selectors:
        try:
            loc = page.locator(sel)
            if loc.count() > 0:
                return loc.first
        except Exception:
            continue
    return None


def has_any(page, selectors):
    return find_locator(page, selectors) is not None


def wait_for_locator(page, selectors, timeout, status, message):
    found = {"loc": None}

    def predicate():
        found["loc"] = find_locator(page, selectors)
        return found["loc"] is not None

    if not poll_until(predicate, timeout, status, message, interval=1.0):
        return None
    return found["loc"]


def fill_locator(page, loc, text):
    try:
        loc.fill(text)
    except Exception:
        loc.click()
        page.keyboard.insert_text(text)


def click_first(page, selectors):
    loc = find_locator(page, selectors)
    if loc:
        loc.click()
        return True
    return False


def answer_snapshot(page):
    for sel in DEEPSEEK["answer"]:
        try:
            loc = page.locator(sel)
            if loc.count() > 0:
                return loc.last.inner_text(timeout=2000).strip(), True
        except Exception:
            continue
    return "", False


def wait_captcha(page, status, timeout=DEEPSEEK_CAPTCHA_TIMEOUT):
    if not has_any(page, DEEPSEEK["captcha"]):
        return True
    emit_status(status, "Ответь на капчу в браузере")
    ok = poll_until(
        lambda: not has_any(page, DEEPSEEK["captcha"]),
        timeout,
        status,
        "Ответь на капчу в браузере",
        interval=1.0,
    )
    return ok


def wait_deepseek_login(page, status):
    # Уже залогинены — не беспокоим пользователя лишним сообщением.
    if has_any(page, DEEPSEEK["logged_in"]):
        return True
    emit_status(status, "Залогинься в окне браузера (один раз)")
    return poll_until(
        lambda: has_any(page, DEEPSEEK["logged_in"]),
        DEEPSEEK_LOGIN_TIMEOUT,
        status,
        "Залогинься в окне браузера (один раз)",
        interval=1.0,
    )


def wait_deepseek_answer(page, status, baseline):
    started = time.monotonic()
    last_second = -1
    last_text = None
    stable_since = None
    saw_selector = False
    while True:
        elapsed = time.monotonic() - started
        if elapsed >= DEEPSEEK_ANSWER_TIMEOUT:
            if not saw_selector:
                return False, "Сайт изменился: не нашла поле ответа DeepSeek."
            return False, "DeepSeek не ответил за 60 секунд. Проверь сеть или попробуй ещё раз."

        if has_any(page, DEEPSEEK["captcha"]):
            if not wait_captcha(page, status):
                return False, "Капча не подтверждена за 3 минуты."

        text, found = answer_snapshot(page)
        saw_selector = saw_selector or found
        if text and text != baseline:
            if text == last_text:
                if stable_since is not None and time.monotonic() - stable_since >= 2:
                    return True, text
            else:
                last_text = text
                stable_since = time.monotonic()

        second = int(elapsed)
        if second != last_second:
            last_second = second
            emit_status(status, "Жду ответ DeepSeek…", started)
        time.sleep(1)


def deepseek_answer(session, query, status=print):
    emit_status(status, "Открываю DeepSeek…")
    try:
        if session.page is None:
            session.open()
        page = session.page
        try:
            page.goto(DEEPSEEK["url"], wait_until="domcontentloaded", timeout=30000)
        except Exception as e:
            return False, f"Не удалось открыть DeepSeek. Проверь сеть: {e}"

        if not wait_deepseek_login(page, status):
            return False, "Не дождалась логина в DeepSeek за 5 минут. Открой /браузер и залогинься."
        if not wait_captcha(page, status):
            return False, "Капча не подтверждена за 3 минуты."

        field = wait_for_locator(page, DEEPSEEK["input"], 15, status, "Ищу поле ввода DeepSeek…")
        if field is None:
            return False, "Сайт изменился: не нашла поле ввода DeepSeek."
        fill_locator(page, field, query)
        baseline, _ = answer_snapshot(page)
        if not click_first(page, DEEPSEEK["send"]):
            try:
                field.press("Enter")
            except Exception as e:
                return False, f"Не удалось отправить запрос в DeepSeek: {e}"
        return wait_deepseek_answer(page, status, baseline)
    except Exception as e:
        try:
            session.close()
        except Exception:
            pass
        return False, f"Не удалось получить ответ DeepSeek. Сайт мог измениться: {e}"

def browser_setup(session, status=print):
    try:
        session.open()
        session.page.goto(DEEPSEEK["url"], wait_until="domcontentloaded", timeout=30000)
        if FEATURES["vk_music"]:
            session.vk_page = session.context.new_page()
            session.vk_page.goto(VK_MUSIC["url"], wait_until="domcontentloaded", timeout=30000)
    except Exception as e:
        session.close()
        return False, f"Не удалось открыть окна для логина: {e}"

    if FEATURES["vk_music"]:
        emit_status(status, "Окно браузера ассистента открыто. Залогинься в DeepSeek и VK Музыке, затем закрой его.")
    else:
        emit_status(status, "Окно браузера ассистента открыто. Залогинься в DeepSeek, затем закрой его.")
    started = time.monotonic()
    last_second = -1
    while time.monotonic() - started < BROWSER_SETUP_TIMEOUT:
        try:
            if not session.context.pages:
                break
        except Exception:
            break
        second = int(time.monotonic() - started)
        if second != last_second and second % 5 == 0:
            last_second = second
            emit_status(status, "Жду настройку браузера…", started)
        time.sleep(1)
    session.close()
    return True, "Профиль браузера ассистента готов. Логин сохранён."


def wait_vk_login(page, status):
    if has_any(page, VK_MUSIC["logged_in"]):
        return True
    emit_status(status, "Открой VK в окне браузера ассистента, залогинься")
    return poll_until(
        lambda: has_any(page, VK_MUSIC["logged_in"]),
        VK_LOGIN_TIMEOUT,
        status,
        "Открой VK в окне браузера ассистента, залогинься",
        interval=1.0,
    )


def format_track_label(text):
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if len(lines) >= 2:
        return f"{lines[0]} — {lines[1]}"
    return lines[0] if lines else text.strip()


def vk_results(page):
    for sel in VK_MUSIC["search_result"]:
        try:
            loc = page.locator(sel)
            count = min(loc.count(), 10)
            if count:
                items = []
                for i in range(count):
                    row = loc.nth(i)
                    try:
                        text = row.inner_text(timeout=2000).strip()
                    except Exception:
                        continue
                    if text:
                        items.append((row, text))
                if items:
                    return items
        except Exception:
            continue
    return []


def vk_exact_match(text, query):
    haystack = re.sub(r"\s+", " ", text.lower())
    tokens = [t for t in re.split(r"\s+", query.lower()) if t]
    return bool(tokens) and all(t in haystack for t in tokens)


def vk_music_action(session, intent, status=print):
    action = intent.get("action", "play")
    query = intent.get("query", "")
    if session.page is None:
        session.open()
    page = session.vk_page or session.page
    session.vk_page = page
    try:
        if "vk.com" not in (page.url or ""):
            page.goto(VK_MUSIC["url"], wait_until="domcontentloaded", timeout=30000)
    except Exception as e:
        return False, f"Не удалось открыть VK Музыку. Проверь сеть: {e}"

    if not wait_vk_login(page, status):
        return False, "Не дождалась логина в VK за 5 минут. Открой /браузер и залогинься."

    try:
        if action == "play":
            emit_status(status, f"Ищу в VK Музыке: {query}…")
            field = wait_for_locator(page, VK_MUSIC["search_input"], 15, status, "Ищу поле поиска VK Музыки…")
            if field is None:
                return False, "Сайт изменился: не нашла поиск в VK Музыке."
            fill_locator(page, field, query)
            try:
                field.press("Enter")
            except Exception:
                pass

            found = {"items": []}
            def results_ready():
                found["items"] = vk_results(page)
                return bool(found["items"])

            if not poll_until(results_ready, 20, status, "Жду результаты поиска…", interval=1.0):
                return False, "VK Музыка не показала результаты поиска."

            items = found["items"]
            chosen = None
            for row, text in items:
                if vk_exact_match(text, query):
                    chosen = (row, text)
                    break
            if chosen is None:
                hints = ", ".join(format_track_label(t) for _r, t in items[:3])
                return True, f"Не нашёл. Ближайшие варианты: {hints or 'ничего не найдено'}."

            row, text = chosen
            try:
                row.click()
            except Exception:
                pass
            if not click_first(page, VK_MUSIC["play_button"]):
                try:
                    page.keyboard.press("Enter")
                except Exception:
                    pass
            return True, f"Включаю: {format_track_label(text)}"

        if action == "pause":
            if not click_first(page, VK_MUSIC["pause_button"]):
                page.keyboard.press("Space")
            return True, "Пауза."
        if action == "resume":
            if not click_first(page, VK_MUSIC["play_button"]):
                page.keyboard.press("Space")
            return True, "Продолжаю."
        if action == "next":
            if not click_first(page, VK_MUSIC["next_button"]):
                page.keyboard.press("ArrowRight")
            return True, "Следующий трек."
        if action == "stop":
            if not click_first(page, VK_MUSIC["pause_button"]):
                page.keyboard.press("Space")
            return True, "Музыка выключена."
        return False, f"Неизвестная музыкальная команда: {action}"
    except Exception as e:
        return False, f"Не удалось управлять VK Музыкой. Разметка могла измениться: {e}"


def execute_browser_intent(session, intent, status=print):
    """Общий обработчик браузерных сценариев для GUI, --text и --voice."""
    kind = intent.get("kind")
    if kind == "browser_setup":
        return browser_setup(session, status)
    if kind == "deepseek":
        return deepseek_answer(session, intent.get("query", ""), status)
    if kind == "music":
        return vk_music_action(session, intent, status)
    return False, "Неизвестная команда браузера."


class ConsoleBrowser:
    """Синхронный постоянный браузер для консольного и голосового режимов."""

    def __init__(self, status=print):
        self.status = status
        self.session = None
        self.lock = threading.Lock()

    def execute(self, intent):
        with self.lock:
            try:
                if self.session is None:
                    self.session = BrowserSession(self.status)
                ok, text = execute_browser_intent(self.session, intent, self.status)
                if not ok:
                    self.session.close()
                    self.session = None
                return ok, text
            except Exception as e:
                if self.session:
                    self.session.close()
                    self.session = None
                return False, f"Не удалось выполнить действие в браузере. Состояние сброшено: {e}"

    def close(self):
        with self.lock:
            if self.session:
                self.session.close()
                self.session = None


# ==================== ТЕКСТОВЫЙ РЕЖИМ ====================
def command_hint():
    """Подсказка только по включённым на этом этапе возможностям."""
    parts = ["/модель", "/модель <имя>", "/статус", "/старт"]
    if FEATURES["autostart"]:
        parts.append("/автозагрузка")
    if FEATURES["deepseek"]:
        parts.append("д: <вопрос>")
    if FEATURES["vk_music"]:
        parts.append("музыка")
    if FEATURES["deepseek"] or FEATURES["vk_music"]:
        parts.append("/браузер")
    parts.append("выход")
    return ", ".join(parts)


def main_text():
    global MODEL
    print(f"Алиса {ALICE_VERSION} (текст) | модель: {MODEL} | контекст: {NUM_CTX}")
    print(f"Команды: {command_hint()}\n")
    if not ensure_ollama(print):
        print(f"Система: {OLLAMA_LAST_ERROR or with_log_path('Ollama недоступна.')}")
    browser = ConsoleBrowser(print)
    history = [{"role": "system", "content": SYSTEM}]

    while True:
        try:
            if sys.stdin.isatty():
                text = input("Ты: ").strip()
            else:
                text = input().strip()
                print(f"Ты: {text}")
        except (EOFError, KeyboardInterrupt):
            print("\nПока!")
            break
        if not text:
            continue
        intent = classify_command(text)
        kind = intent["kind"]

        if kind == "exit":
            print("Пока!")
            break
        if kind == "model_list":
            names = list_models()
            print("Скачанные модели:" if names else "Скачанных моделей нет.")
            for name in names:
                mark = "  <- текущая" if name.startswith(MODEL) else ""
                print(f"  - {name}{mark}")
            continue
        if kind == "model_set":
            MODEL = intent["model"]
            history = [{"role": "system", "content": SYSTEM}]
            print(f"Переключилась на {MODEL}, история очищена.")
            continue
        if kind == "status":
            _states, text_out = health_snapshot(len(history) - 1)
            print(text_out)
            continue
        if kind == "restart_ollama":
            restart_ollama_in_background(print)
            continue
        if kind == "autostart":
            ok, response = autostart_ollama()
            print(("Система: " if not ok else "") + response)
            continue
        if kind == "unavailable":
            print(f"Система: {intent['reason']}")
            continue
        if kind in {"deepseek", "music", "browser_setup"}:
            ok, response = browser.execute(intent)
            if ok and kind == "deepseek":
                print(f"DeepSeek: {response}")
                history.append({"role": "assistant", "content": f"DeepSeek: {response}"})
            elif ok:
                print(f"Система: {response}")
            else:
                print(f"Система: {response}")
            continue
        if kind == "local":
            print("Алиса: ", end="", flush=True)
            try:
                history = ask(history, intent["text"])
            except Exception as e:
                print(f"[Ошибка: {e}]")
            print()
            continue

    browser.close()


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
            for item in root.rglob(fname):
                return item
        except OSError:
            continue
    p = Path.home() / fname
    return p if p.exists() else None


def calibrate_threshold(status=None):
    """Меряет шум комнаты и выставляет порог срабатывания на речь."""
    if not load_voice_stack(status=status):
        emit_status(status, "Микрофон недоступен, использую порог по умолчанию.")
        return 0.01
    np = VOICE_STACK["np"]
    sd = VOICE_STACK["sd"]
    emit_status(status, "Калибровка микрофона — помолчи полсекунды...")
    try:
        data = sd.rec(int(0.6 * SR), samplerate=SR, channels=1, dtype="float32")
        sd.wait()
        rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
    except Exception:
        rms = 0.0
    th = max(0.010, min(rms * 3.0, 0.05))
    emit_status(status, f"Порог: {th:.4f}")
    return th


def listen(threshold, status=None):
    """Слушает микрофон. Возвращает речь (float32, 16 кГц) или None при тишине."""
    if not load_voice_stack(status=status):
        emit_status(status, "Микрофон недоступен.")
        return None
    np = VOICE_STACK["np"]
    sd = VOICE_STACK["sd"]
    block = int(SR * 0.03)
    pre = []
    buf = []
    started = False
    quiet = 0
    t0 = time.time()
    tick_started = time.monotonic()
    last_second = -1
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=block) as st:
        while True:
            data, _ = st.read(block)
            data = data.reshape(-1)
            rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
            now = time.time() - t0
            second = int(time.monotonic() - tick_started)
            if second != last_second:
                last_second = second
                emit_status(status, "Слушаю…", tick_started)

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
        return None
    return audio


def transcribe(model, audio):
    segments, _info = model.transcribe(
        audio,
        language="ru",
        vad_filter=True,
        beam_size=5,
        no_speech_threshold=0.5,
    )
    return "".join(seg.text for seg in segments).strip()


def say(voice, text):
    """Озвучивает текст голосом Piper и ждёт конца воспроизведения."""
    if not load_voice_stack(need_tts=True):
        print(f"[Не удалось озвучить: {VOICE_ERR}]")
        return
    winsound = VOICE_STACK["winsound"]
    try:
        with wave.open(str(REPLY_WAV), "wb") as f:
            voice.synthesize_wav(text, f)
        winsound.PlaySound(str(REPLY_WAV), winsound.SND_FILENAME | winsound.SND_NODEFAULT)
    except Exception as e:
        print(f"[Не удалось озвучить: {e}]")


def main_voice():
    global MODEL
    print("=== Голосовая Алиса ===")
    if not load_voice_stack(need_stt=True, need_tts=True, status=print):
        print(f"Голосовой стек не установлен: {VOICE_ERR}")
        print("Выполни:  pip install faster-whisper piper-tts sounddevice numpy")
        return 1

    WhisperModel = VOICE_STACK["WhisperModel"]
    PiperVoice = VOICE_STACK["PiperVoice"]

    if not ensure_ollama(print):
        print(f"Система: {OLLAMA_LAST_ERROR or with_log_path('Ollama недоступна.')}")
    browser = ConsoleBrowser(print)

    onnx = find_voice(VOICE_NAME)
    if onnx is None:
        print(f"Голос {VOICE_NAME} не найден. Выполни:")
        print(f"  python -m piper.download_voices {VOICE_NAME}")
        browser.close()
        return 1
    print(f"Голос: {onnx}")

    print(f"Распознавание: whisper '{WHISPER_MODEL}' (первый раз — скачается ~460 МБ)...")
    stt = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    print("Синтез: Piper...")
    tts_voice = PiperVoice.load(str(onnx))
    threshold = calibrate_threshold(print)
    history = [{"role": "system", "content": SYSTEM}]

    say(tts_voice, "Алиса запущена. Скажи что-нибудь.")
    print(f"Говори спокойно. Команды: {command_hint()}.\n")

    while True:
        print("Слушаю...")
        audio = listen(threshold, print)
        if audio is None:
            continue
        heard = transcribe(stt, audio)
        if not heard:
            continue
        print(f"Ты: {heard}")
        intent = classify_command(heard)
        kind = intent["kind"]

        if kind == "exit":
            say(tts_voice, "До встречи.")
            print("Пока!")
            break
        if kind == "model_list":
            names = list_models()
            say(tts_voice, "Скачанные модели: " + (", ".join(names) if names else "нет"))
            continue
        if kind == "model_set":
            MODEL = intent["model"]
            history = [{"role": "system", "content": SYSTEM}]
            say(tts_voice, f"Переключилась на {MODEL}.")
            continue
        if kind == "status":
            _states, text_out = health_snapshot(len(history) - 1)
            print(text_out)
            say(tts_voice, "Статус в консоли.")
            continue
        if kind == "restart_ollama":
            print("Повторяю запуск Ollama в фоне…")
            say(tts_voice, "Повторяю запуск Олламы.")
            restart_ollama_in_background(print)
            continue
        if kind == "autostart":
            ok, response = autostart_ollama()
            print(f"Система: {response}")
            say(tts_voice, response)
            continue
        if kind == "unavailable":
            print(f"Система: {intent['reason']}")
            say(tts_voice, intent["reason"])
            continue
        if kind in {"deepseek", "music", "browser_setup"}:
            ok, response = browser.execute(intent)
            if ok and kind == "deepseek":
                print(f"DeepSeek: {response}")
                history.append({"role": "assistant", "content": f"DeepSeek: {response}"})
            else:
                print(f"Система: {response}")
            say(tts_voice, response)
            continue
        if kind == "local":
            print("Алиса: ", end="", flush=True)
            try:
                history = ask(history, heard)
            except Exception as e:
                print(f"[Ошибка мозга: {e}]")
                say(tts_voice, "Не получилось ответить.")
                continue
            answer = history[-1].get("content", "") if isinstance(history[-1], dict) else ""
            if answer:
                say(tts_voice, answer)
            continue

    browser.close()


# ==================== GUI-РЕЖИМ (PySide6) ====================
def main_gui():
    try:
        from PySide6.QtCore import (
            QObject, QThread, QTimer, Qt, QtMsgType, Signal, Slot,
            qInstallMessageHandler,
        )
        from PySide6.QtGui import QFont, QTextCursor
        from PySide6.QtWidgets import (
            QApplication, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
            QPushButton, QTextEdit, QVBoxLayout, QWidget,
        )
    except Exception as e:
        print(f"Не удалось загрузить PySide6: {e}")
        print("Выполни:  pip install PySide6")
        return 1

    MAIN_QT_THREAD = QThread.currentThread()
    try:
        def qt_message_handler(mode, context, message):
            mode_name = getattr(mode, "name", str(mode))
            console_log(
                f"Qt {mode_name}: {message}",
                level="WARNING",
                key=f"qt:{mode_name}:{message}",
                min_interval=1.0,
            )
        qInstallMessageHandler(qt_message_handler)
    except Exception as e:
        console_log(f"Не удалось установить обработчик Qt-сообщений: {e}", level="WARNING")

    def gui_only(fn):
        """Слот/метод GUI: из чужого потока уходим через queued ui_call, ошибки не роняем окно."""
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            try:
                current = QThread.currentThread()
                owner = self.thread() if hasattr(self, "thread") else MAIN_QT_THREAD
                if current is not owner:
                    self.ui_call.emit((fn.__name__, args, kwargs))
                    return None
            except Exception:
                pass
            guard = (
                error_display_guard()
                if fn.__name__ in ERROR_DISPLAY_METHODS
                else contextlib.nullcontext()
            )
            with guard:
                try:
                    return fn(self, *args, **kwargs)
                except Exception:
                    report_exception(*sys.exc_info(), origin=f"слот {fn.__name__}")
                    return None
        return wrapper

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
        status = Signal(str)
        completed = Signal(str, bool)

        def __init__(self, history):
            super().__init__()
            self.history = list(history)

        def run(self):
            answer, error = generate_answer(
                self.history,
                LLM_TIMEOUT,
                on_chunk=self.chunk.emit,
                on_status=self.status.emit,
            )
            self.completed.emit(answer if error is None else error, error is None)

    class FunctionWorker(QObject):
        completed = Signal(str, bool)

        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def run(self):
            try:
                self.completed.emit(self.fn(), True)
            except Exception as e:
                self.completed.emit(f"Ошибка: {e}", False)

    class HealthWorker(QObject):
        result = Signal(str, str, str, str)

        def __init__(self, history_len, browser_override=None):
            super().__init__()
            self.history_len = history_len
            self.browser_override = browser_override

        def run(self):
            states, text = health_snapshot(self.history_len, self.browser_override)
            self.result.emit(states["ollama"], states["model"], states["browser"], text)

    class BrowserWorker(QObject):
        status = Signal(str)
        result = Signal(str, bool, str)

        def __init__(self):
            super().__init__()
            self.queue = queue.Queue()
            self.session = None
            self._running = True

        def submit(self, intent):
            self.queue.put(intent)

        def stop(self):
            self._running = False
            self.queue.put(None)

        def run(self):
            while self._running:
                try:
                    intent = self.queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if intent is None:
                    break
                try:
                    if self.session is None:
                        self.session = BrowserSession(self.status.emit)
                    ok, text = execute_browser_intent(self.session, intent, self.status.emit)
                    if not ok and self.session:
                        self.session.close()
                        self.session = None
                    self.result.emit(text, ok, intent.get("kind", ""))
                except Exception as e:
                    if self.session:
                        self.session.close()
                        self.session = None
                    self.result.emit(f"Не удалось выполнить действие в браузере. Состояние сброшено: {e}", False, intent.get("kind", ""))

    class VoiceWorker(QObject):
        status = Signal(str)
        completed = Signal(str, bool)

        def run(self):
            if not load_voice_stack(need_stt=True, status=self.status.emit):
                self.completed.emit(
                    f"Голосовой стек не установлен: {VOICE_ERR}. Проверь микрофон и зависимости.",
                    False,
                )
                return
            try:
                WhisperModel = VOICE_STACK["WhisperModel"]
                stt = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
                threshold = calibrate_threshold(self.status.emit)
                audio = listen(threshold, self.status.emit)
                if audio is None:
                    self.completed.emit("Не расслышала. Проверь микрофон или попробуй ещё раз.", False)
                    return
                text = transcribe(stt, audio)
                if not text:
                    self.completed.emit("Не расслышала. Проверь микрофон или попробуй ещё раз.", False)
                else:
                    self.completed.emit(text, True)
            except Exception as e:
                self.completed.emit(f"Микрофон недоступен или слишком шумно: {e}", False)

    class ChatWindow(QMainWindow):
        error_signal = Signal(str)
        ui_call = Signal(object)

        def __init__(self):
            super().__init__()
            global GUI_WINDOW
            GUI_WINDOW = self
            self.setWindowTitle(f"Алиса {ALICE_VERSION} — локальный ассистент")
            self.resize(860, 620)
            self.setMinimumSize(560, 430)

            self.history = [{"role": "system", "content": SYSTEM}]
            self.ollama_ready = False
            self.busy = False
            self.important_statuses = set()
            self.assistant_open = False
            self.live_marker = None
            self._error_display_depth = 0
            self.first_local_request = True
            self.browser_health = None
            self.health = {"ollama": "warn", "model": "warn", "browser": "warn"}
            self.health_text = status_text(0)

            self.startup_thread = None
            self.startup_worker = None
            self.chat_thread = None
            self.chat_worker = None
            self.health_thread = None
            self.health_worker = None
            self.health_busy = False
            self.browser_thread = None
            self.browser_worker = None
            self.voice_thread = None
            self.voice_worker = None
            self.autostart_thread = None
            self.autostart_worker = None

            central = QWidget(self)
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(10)

            header = QHBoxLayout()
            self.ollama_indicator = QLabel("● Ollama", self)
            self.model_indicator = QLabel("● Модель", self)
            self.browser_indicator = QLabel("● Браузер", self)
            for label in (self.ollama_indicator, self.model_indicator, self.browser_indicator):
                header.addWidget(label)
            header.addStretch(1)
            self.header_widget = QWidget(self)
            self.header_widget.setLayout(header)
            layout.addWidget(self.header_widget)
            self.header_widget.setVisible(FEATURES["status_bar"])

            self.chat = QTextEdit(self)
            self.chat.setReadOnly(True)
            self.chat.setFont(QFont("Segoe UI", 11))
            layout.addWidget(self.chat, 1)

            row = QHBoxLayout()
            self.input = QLineEdit(self)
            self.input.setPlaceholderText("Напиши сообщение и нажми Enter...")
            self.input.setFont(QFont("Segoe UI", 11))
            self.input.returnPressed.connect(self.send_message, Qt.QueuedConnection)
            row.addWidget(self.input, 1)

            self.send_button = QPushButton("Отправить", self)
            self.send_button.clicked.connect(self.send_message, Qt.QueuedConnection)
            row.addWidget(self.send_button)

            self.action_buttons = [self.send_button]

            if FEATURES["deepseek"]:
                self.deepseek_button = QPushButton("DeepSeek", self)
                self.deepseek_button.setToolTip("Отправить вопрос в DeepSeek")
                self.deepseek_button.clicked.connect(lambda: self.send_message(True), Qt.QueuedConnection)
                row.addWidget(self.deepseek_button)
                self.action_buttons.append(self.deepseek_button)

            if FEATURES["deepseek"] or FEATURES["vk_music"]:
                self.browser_button = QPushButton("Браузер", self)
                self.browser_button.setToolTip("Один раз настроить профиль браузера ассистента")
                self.browser_button.clicked.connect(self.open_browser_setup, Qt.QueuedConnection)
                row.addWidget(self.browser_button)
                self.action_buttons.append(self.browser_button)

            self.retry_button = QPushButton("Повторить", self)
            self.retry_button.setToolTip("Повторить запуск Ollama (/старт)")
            self.retry_button.clicked.connect(self.start_ollama_check, Qt.QueuedConnection)
            row.addWidget(self.retry_button)
            self.action_buttons.append(self.retry_button)

            if FEATURES["voice_gui"]:
                self.mic_button = QPushButton("🎙", self)
                self.mic_button.setToolTip("Записать голосом — текст попадёт в поле ввода")
                self.mic_button.clicked.connect(self.start_voice_capture, Qt.QueuedConnection)
                row.addWidget(self.mic_button)
                self.action_buttons.append(self.mic_button)

            layout.addLayout(row)

            self.status_label = QLabel("Проверяю Ollama...", self)
            layout.addWidget(self.status_label)

            self.error_signal.connect(self.on_global_error, Qt.QueuedConnection)
            self.ui_call.connect(self.on_ui_call, Qt.QueuedConnection)
            self.append_status(f"Алиса {ALICE_VERSION} запущена. Проверяю Ollama...")
            self.set_busy(True, "Проверяю Ollama...")
            self.input.setFocus()
            QTimer.singleShot(0, self.start_ollama_check)

            self.health_timer = QTimer(self)
            self.health_timer.timeout.connect(self.start_health_check, Qt.QueuedConnection)
            self.health_timer.start(HEALTH_INTERVAL_MS)
            QTimer.singleShot(0, self.start_health_check)

        # ---------- защита GUI-потока ----------
        @Slot(object)
        @gui_only
        def on_ui_call(self, payload):
            name, args, kwargs = payload
            getattr(self, name)(*args, **kwargs)

        @Slot(str)
        @gui_only
        def on_global_error(self, message):
            try:
                self.append_error(message, source="report_exception")
            finally:
                self.set_busy(False, "Ошибка перехвачена")
                self.input.setEnabled(True)
                self.input.setFocus()

        # ---------- лента ----------
        def _feed_append_html(self, html_text, newline=True, live=False):
            """Единственная точка вставки в QTextEdit: moveCursor + insertHtml + append('')."""
            try:
                if live:
                    cursor = self.chat.textCursor()
                    cursor.movePosition(QTextCursor.End)
                    if self.live_marker is not None:
                        cursor.setPosition(self.live_marker, QTextCursor.MoveAnchor)
                        cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
                        cursor.removeSelectedText()
                    else:
                        self.live_marker = cursor.position() + 1
                self.chat.moveCursor(QTextCursor.End)
                self.chat.insertHtml(html_text)
                if newline:
                    self.chat.append("")
                self.scroll_down()
            except Exception as e:
                # QTextEdit.insertHtml может отсутствовать/упасть; append(str) — корректный fallback.
                try:
                    plain = re.sub(r"<[^>]+>", "", str(html_text))
                    self.chat.append(plain)
                    self.live_marker = None
                    self.scroll_down()
                except Exception as e2:
                    report_feed_failure(e2, "_feed_append_html")

        @gui_only
        def clear_live_status(self):
            self.live_marker = None

        @gui_only
        def set_live_status(self, text):
            console_log(
                f"Система: {text}",
                level="INFO",
                key=console_key(text),
                min_interval=5.0,
            )
            self._feed_append_html(
                f"Система: {html.escape(str(text))}", newline=True, live=True
            )

        @Slot(str)
        @gui_only
        def append_status(self, text):
            console_log(text, level="INFO", key=console_key(text), min_interval=1.0)
            self.clear_live_status()
            self._feed_append_html(f"Система: {html.escape(str(text))}")

        @gui_only
        def append_error(self, text, source=None):
            message = str(text)
            key = error_key(message)
            console_log_error(message, key=key, origin=source)
            # Повторный заход в показ ошибки не трогает ленту: только консоль + лог.
            if self._error_display_depth:
                return
            if not error_feed_allowed(key):
                return
            self._error_display_depth += 1
            try:
                self.clear_live_status()
                self._feed_append_html(
                    f'<span style="color:#ff6b6b;">Система: {html.escape(message)}</span>'
                )
            except Exception:
                report_exception(*sys.exc_info(), origin="показ ошибки в ленте")
            finally:
                self._error_display_depth -= 1

        @gui_only
        def append_user(self, text):
            console_log(f"Ты: {text}", level="USER")
            self.clear_live_status()
            self._feed_append_html(f"Ты: {html.escape(str(text))}")

        @gui_only
        def append_deepseek(self, text):
            console_log(f"DeepSeek: {text}", level="WEB")
            self.clear_live_status()
            self._feed_append_html(f"DeepSeek: {html.escape(str(text))}")

        @gui_only
        def begin_assistant(self):
            self.clear_live_status()
            self._feed_append_html("Алиса: ", newline=False)
            self.assistant_open = True

        @gui_only
        def finish_assistant(self):
            if self.assistant_open:
                self._feed_append_html("", newline=True)
                self.assistant_open = False

        @Slot(str)
        @gui_only
        def append_chunk(self, chunk):
            self.clear_live_status()
            self._feed_append_html(html.escape(str(chunk)), newline=False)

        @gui_only
        def scroll_down(self):
            self.chat.moveCursor(QTextCursor.End)
            self.chat.ensureCursorVisible()
            bar = self.chat.verticalScrollBar()
            bar.setValue(bar.maximum())

        @gui_only
        def set_busy(self, busy, label=None):
            self.busy = busy
            self.input.setEnabled(not busy)
            for button in self.action_buttons:
                button.setEnabled(not busy)
            if busy:
                self.important_statuses.clear()
            if label:
                self.status_label.setText(label)

        @Slot(str)
        @gui_only
        def on_progress(self, text):
            self.status_label.setText(text)
            # Во время стриминга ответа не вставляем таймер в ленту,
            # иначе он «разрывает» печатаемую строку Алисы.
            if self.assistant_open:
                return
            # Ключевые этапы остаются в истории ленты, а односекундные
            # таймеры обновляют только одну живую строку.
            key = next((
                prefix for prefix in (
                    "Открываю DeepSeek",
                    "Залогинься в окне браузера",
                    "Ответь на капчу",
                    "Ищу в VK Музыке",
                    "Открой VK",
                    "Окно браузера ассистента открыто",
                    "Проверяю Ollama",
                    "Ollama API не отвечает",
                    "Процессы Ollama",
                    "Запускаю ollama app.exe",
                    "Запускаю ollama.exe serve",
                    "Запускаю OllamaServer.exe",
                    "Запускаю ollama.exe list",
                    "Жду Ollama",
                    "Ollama запущена, но сервер",
                    "Каталоги Ollama",
                    "Найденные exe",
                    "Модель не скачана",
                    "Порт 11434 занят",
                    "Не удалось запустить Ollama",
                ) if text.startswith(prefix)
            ), None)
            if key and key not in self.important_statuses:
                self.append_status(text)
                self.important_statuses.add(key)
            else:
                self.set_live_status(text)

        # ---------- Ollama ----------
        @gui_only
        def start_ollama_check(self):
            if self.busy and self.startup_worker is not None:
                return
            self.clear_live_status()
            self.set_busy(True, "Проверяю Ollama...")
            thread = QThread(self)
            worker = StartupWorker()
            worker.moveToThread(thread)
            worker.status.connect(self.on_progress, Qt.QueuedConnection)
            worker.completed.connect(self.on_startup_done, Qt.QueuedConnection)
            worker.completed.connect(thread.quit)
            worker.completed.connect(worker.deleteLater)
            thread.started.connect(worker.run)
            thread.finished.connect(thread.deleteLater)
            self.startup_thread = thread
            self.startup_worker = worker
            thread.start()

        @Slot(bool)
        @gui_only
        def on_startup_done(self, ok):
            self.clear_live_status()
            self.ollama_ready = ok
            self.startup_worker = None
            if ok:
                self.append_status("Ollama готова. Можно писать.")
                if MODEL_MISSING:
                    self.append_status(f"Модель не скачана: выполните ollama pull {MODEL}")
                    self.update_indicator(self.model_indicator, "Модель", "warn")
                self.set_busy(False, "Ollama: ок")
            else:
                self.append_error(OLLAMA_LAST_ERROR or with_log_path("Ollama недоступна."))
                self.set_busy(False, "Ollama: ошибка")
            self.start_health_check()

        # ---------- здоровье ----------
        @gui_only
        def update_indicator(self, label, name, state):
            colors = {"ok": "#4ec9b0", "warn": "#dcdcaa", "err": "#f14c4c"}
            label.setText(f"● {name}: {state_word(state)}")
            label.setStyleSheet(f"color: {colors.get(state, '#dcdcaa')};")

        @gui_only
        def start_health_check(self):
            if self.health_busy:
                return
            self.health_busy = True
            thread = QThread(self)
            worker = HealthWorker(max(0, len(self.history) - 1), self.browser_health)
            worker.moveToThread(thread)
            worker.result.connect(self.on_health_result, Qt.QueuedConnection)
            worker.result.connect(thread.quit)
            worker.result.connect(worker.deleteLater)
            thread.started.connect(worker.run)
            thread.finished.connect(thread.deleteLater)
            self.health_thread = thread
            self.health_worker = worker
            thread.start()

        @Slot(str, str, str, str)
        @gui_only
        def on_health_result(self, ollama_state, model_state, browser_state, text):
            self.health_busy = False
            self.health = {"ollama": ollama_state, "model": model_state, "browser": browser_state}
            self.health_text = text
            if FEATURES["status_bar"]:
                self.update_indicator(self.ollama_indicator, "Ollama", ollama_state)
                self.update_indicator(self.model_indicator, "Модель", model_state)
                self.update_indicator(self.browser_indicator, "Браузер", browser_state)

        # ---------- браузер ----------
        @gui_only
        def ensure_browser_worker(self):
            if not playwright_available():
                self.append_error("Playwright не установлен. Выполни: pip install playwright")
                return False
            if self.browser_worker is None:
                self.browser_worker = BrowserWorker()
                self.browser_thread = QThread(self)
                self.browser_worker.moveToThread(self.browser_thread)
                self.browser_worker.status.connect(self.on_progress, Qt.QueuedConnection)
                self.browser_worker.result.connect(self.on_browser_result, Qt.QueuedConnection)
                self.browser_thread.started.connect(self.browser_worker.run)
                self.browser_thread.finished.connect(self.browser_thread.deleteLater)
                self.browser_thread.start()
            return True

        @Slot()
        @gui_only
        def open_browser_setup(self):
            if self.busy:
                return
            if not self.ensure_browser_worker():
                return
            intent = classify_command("/браузер")
            if intent.get("kind") == "unavailable":
                self.append_status(intent["reason"])
                return
            self.clear_live_status()
            self.set_busy(True, "Открываю браузер…")
            self.browser_worker.submit(intent)

        @gui_only
        def start_browser_intent(self, intent):
            if not self.ensure_browser_worker():
                return
            self.clear_live_status()
            self.set_busy(True, "Открываю браузер…")
            self.browser_worker.submit(intent)

        @Slot(str, bool, str)
        @gui_only
        def on_browser_result(self, payload, ok, kind):
            self.clear_live_status()
            if not ok:
                self.browser_health = "err"
                self.append_error(payload)
                self.set_busy(False, "Ошибка")
                self.start_health_check()
                return
            self.browser_health = "ok"
            if kind == "deepseek":
                self.append_deepseek(payload)
                self.history.append({"role": "assistant", "content": f"DeepSeek: {payload}"})
            else:
                self.append_status(payload)
            self.set_busy(False, "Готово")
            self.start_health_check()

        # ---------- локальный мозг ----------
        @gui_only
        def start_chat(self, intent):
            if not self.ollama_ready:
                self.append_error("Ollama недоступна. Запусти её и перезапусти Алису.")
                return
            if self.first_local_request:
                self.append_status("Грузим модель, первый ответ может занять до минуты…")
                self.first_local_request = False
            self.append_status("Думаю…")
            self.begin_assistant()
            self.history.append({"role": "user", "content": intent["text"]})
            self.set_busy(True, "Алиса печатает…")
            thread = QThread(self)
            worker = ChatWorker(self.history)
            worker.moveToThread(thread)
            worker.chunk.connect(self.append_chunk, Qt.QueuedConnection)
            worker.status.connect(self.on_progress, Qt.QueuedConnection)
            worker.completed.connect(self.on_chat_completed, Qt.QueuedConnection)
            worker.completed.connect(thread.quit)
            worker.completed.connect(worker.deleteLater)
            thread.started.connect(worker.run)
            thread.finished.connect(thread.deleteLater)
            self.chat_thread = thread
            self.chat_worker = worker
            thread.start()

        @Slot(str, bool)
        @gui_only
        def on_chat_completed(self, payload, ok):
            self.clear_live_status()
            self.finish_assistant()
            if ok:
                console_log(f"Алиса: {payload}", level="ASSISTANT")
                if payload:
                    self.history.append({"role": "assistant", "content": payload})
                self.set_busy(False, "Готово")
            else:
                drop_dangling(self.history)
                self.append_error(payload)
                self.set_busy(False, "Ошибка LLM")
            self.input.setFocus()

        # ---------- команды ----------
        @gui_only
        def start_model_list(self):
            self.set_busy(True, "Проверяю модели…")
            thread = QThread(self)
            worker = FunctionWorker(lambda: "\n".join(f"  - {n}" for n in list_models()) or "Скачанных моделей нет.")
            worker.moveToThread(thread)
            worker.completed.connect(self.on_model_list, Qt.QueuedConnection)
            worker.completed.connect(thread.quit)
            worker.completed.connect(worker.deleteLater)
            thread.started.connect(worker.run)
            thread.finished.connect(thread.deleteLater)
            self.chat_thread = thread
            self.chat_worker = worker
            thread.start()

        @Slot(str, bool)
        @gui_only
        def on_model_list(self, payload, ok):
            if ok:
                self.append_status("Скачанные модели:\n" + payload)
            else:
                self.append_error(payload)
            self.set_busy(False, "Готово")

        @Slot()
        @gui_only
        def send_message(self, force_deepseek=False):
            if self.busy:
                return
            text = self.input.text().strip()
            if not text:
                return
            if force_deepseek and not text.lower().startswith(("д:", "d:")):
                text = "д: " + text
            self.input.clear()
            intent = classify_command(text)
            self.handle_intent(intent)

        @gui_only
        def handle_intent(self, intent):
            global MODEL
            kind = intent.get("kind")
            display = intent.get("display", intent.get("text", ""))
            if kind == "exit":
                self.append_user(display)
                self.append_status("До встречи.")
                self.close()
                return
            if kind == "unavailable":
                self.append_status(intent.get("reason", "Команда пока недоступна."))
                return
            if kind == "model_list":
                self.start_model_list()
                return
            if kind == "model_set":
                MODEL = intent["model"]
                self.history = [{"role": "system", "content": SYSTEM}]
                self.append_status(f"Переключилась на {MODEL}, история очищена.")
                self.start_health_check()
                return
            if kind == "status":
                self.append_status(self.health_text or status_text(len(self.history) - 1, **self.health))
                self.start_health_check()
                return
            if kind == "restart_ollama":
                self.append_user(display)
                self.append_status("Повторяю запуск Ollama в фоне…")
                self.start_ollama_check()
                return
            if kind == "autostart":
                self.append_user(display)
                self.start_autostart()
                return
            if kind == "browser_setup":
                self.append_user(display)
                self.start_browser_intent(intent)
                return
            if kind == "deepseek":
                self.append_user(display)
                self.start_browser_intent(intent)
                return
            if kind == "music":
                self.append_user(display)
                self.start_browser_intent(intent)
                return
            if kind == "local":
                self.append_user(display)
                self.start_chat(intent)
                return
            self.append_error("Не поняла команду.")

        # ---------- голосовая кнопка ----------
        @gui_only
        def start_autostart(self):
            if self.busy:
                return
            self.clear_live_status()
            self.set_busy(True, "Настраиваю автозагрузку…")
            thread = QThread(self)
            worker = FunctionWorker(lambda: autostart_ollama()[1])
            worker.moveToThread(thread)
            worker.completed.connect(self.on_autostart_result, Qt.QueuedConnection)
            worker.completed.connect(thread.quit)
            worker.completed.connect(worker.deleteLater)
            thread.started.connect(worker.run)
            thread.finished.connect(thread.deleteLater)
            self.autostart_thread = thread
            self.autostart_worker = worker
            thread.start()

        @Slot(str, bool)
        @gui_only
        def on_autostart_result(self, text, ok):
            self.clear_live_status()
            self.autostart_worker = None
            if ok:
                self.append_status(text)
            else:
                self.append_error(with_log_path(text))
            self.set_busy(False, "Готово" if ok else "Автозагрузка")

        @gui_only
        def start_voice_capture(self):
            if self.busy:
                return
            if not FEATURES["voice_gui"]:
                return
            if not load_voice_stack(need_stt=True):
                self.append_error(
                    f"Голосовой стек не установлен: {VOICE_ERR}. Проверь микрофон и зависимости."
                )
                return
            self.clear_live_status()
            self.set_busy(True, "Слушаю…")
            thread = QThread(self)
            worker = VoiceWorker()
            worker.moveToThread(thread)
            worker.status.connect(self.on_progress, Qt.QueuedConnection)
            worker.completed.connect(self.on_voice_result, Qt.QueuedConnection)
            worker.completed.connect(thread.quit)
            worker.completed.connect(worker.deleteLater)
            thread.started.connect(worker.run)
            thread.finished.connect(thread.deleteLater)
            self.voice_thread = thread
            self.voice_worker = worker
            thread.start()

        @Slot(str, bool)
        @gui_only
        def on_voice_result(self, text, ok):
            self.clear_live_status()
            if ok:
                self.input.setText(text)
                self.append_status("Распознала, текст в поле ввода.")
                self.set_busy(False, "Готово")
            else:
                self.append_error(text)
                self.set_busy(False, "Микрофон")
            self.input.setFocus()

        @gui_only
        def closeEvent(self, event):
            global GUI_WINDOW
            if GUI_WINDOW is self:
                GUI_WINDOW = None
            try:
                if self.browser_worker:
                    self.browser_worker.stop()
                if self.browser_thread:
                    self.browser_thread.quit()
            except Exception:
                pass
            try:
                super().closeEvent(event)
            except Exception:
                pass

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
            padding: 9px 12px;
        }
        QPushButton:hover { background-color: #1177bb; }
        QPushButton:disabled { background-color: #3a3a3a; color: #888888; }
        QLabel { color: #9cdcfe; }
    """)
    window = ChatWindow()
    window.show()
    while True:
        try:
            return app.exec()
        except Exception:
            report_exception(*sys.exc_info(), origin="цикл событий Qt")
            try:
                if not window.isVisible():
                    return 1
            except Exception:
                return 1


# ==================== ТОЧКА ВХОДА ====================
def main():
    global MODEL
    ap = argparse.ArgumentParser(description="Алиса v2 — локальный ассистент")
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--text", action="store_true", help="текстовый чат в консоли")
    modes.add_argument("--voice", action="store_true", help="голосовой режим")
    ap.add_argument("--model", default=None, help=f"модель Ollama (по умолчанию {MODEL})")
    args = ap.parse_args()
    if args.model:
        MODEL = args.model
    single_ok, single_message = acquire_single_instance()
    if not single_ok:
        print(single_message)
        log_event("Второй экземпляр", detail=single_message)
        if not (args.text or args.voice):
            try:
                from PySide6.QtWidgets import QApplication, QMessageBox
                app = QApplication.instance() or QApplication(sys.argv)
                QMessageBox.warning(None, "Алиса", single_message)
                del app
            except Exception:
                pass
        return 1
    if args.text:
        return main_text()
    elif args.voice:
        return main_voice()
    return main_gui()


init_log()
install_warning_hooks()
install_exception_hooks()

if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        print("\nПока!")
