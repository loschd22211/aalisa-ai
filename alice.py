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
import html
import importlib.util
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
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

# Голосовой стек может отсутствовать — окно и консоль всё равно работают.
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

# Включаются последовательно по блокам: 1 → B → C → E → D.
FEATURES = {
    "deepseek": True,
    "vk_music": True,
    "status_bar": True,
    "voice_gui": False,
}

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
    """Проверяет, отвечает ли локальный сервер Ollama, с таймаутом."""
    ok, _ = call_with_timeout(ollama.list, OLLAMA_CALL_TIMEOUT)
    return ok


def list_models():
    ok, resp = call_with_timeout(ollama.list, OLLAMA_CALL_TIMEOUT)
    if not ok:
        return []
    names = []
    for m in _field(resp, "models", []) or []:
        n = _field(m, "model") or _field(m, "name")
        if n:
            names.append(n)
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
    return (
        f"Алиса {ALICE_VERSION}. Модель: {MODEL} ({state_word(model)}). "
        f"Ollama: {state_word(ollama)}. Браузер: {state_word(browser)}. "
        f"История: {history_len} сообщений."
    )


def playwright_available():
    try:
        return importlib.util.find_spec("playwright.sync_api") is not None
    except Exception:
        return False


def health_snapshot(history_len=0, browser_override=None):
    """Один снимок здоровья для индикаторов и /статус."""
    ollama_ok = ollama_alive()
    models = list_models() if ollama_ok else []
    model_ok = any(m.startswith(MODEL) or MODEL.startswith(m) for m in models)
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


# ==================== АВТОЗАПУСК OLLAMA ====================
def ollama_env():
    """Env для запускаемого сервера: настройки передаём явно, не из профиля."""
    env = os.environ.copy()
    env["OLLAMA_CONTEXT_LENGTH"] = "8192"
    env["OLLAMA_FLASH_ATTENTION"] = "1"
    return env


def start_ollama_windows(env, status):
    """Windows: сначала скрытый ollama serve, запасной вариант — ollama app.exe."""
    local = os.environ.get("LOCALAPPDATA")
    serve = None
    if local:
        candidate = Path(local) / "Programs" / "Ollama" / "ollama.exe"
        if candidate.exists():
            serve = str(candidate)
    if serve is None:
        serve = shutil.which("ollama")

    if serve:
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
            emit_status(status, "Запускаю Ollama скрытым процессом ollama serve…")
            return True
        except Exception as e:
            emit_status(status, f"Не удалось запустить ollama serve: {e}. Пробую запасной вариант.")

    app = Path(local) / "Programs" / "Ollama" / "ollama app.exe" if local else None
    if app is not None and app.exists():
        try:
            subprocess.Popen([str(app)], env=env, close_fds=True)
            emit_status(status, "Запустила запасной ollama app.exe.")
            return True
        except Exception as e:
            emit_status(status, f"Не удалось запустить ollama app.exe: {e}.")
    else:
        emit_status(status, "Не нашла ни ollama.exe, ни ollama app.exe.")
    return False


def wait_for_ollama(status=None, timeout=None):
    if timeout is None:
        timeout = OLLAMA_START_TIMEOUT
    return poll_until(ollama_alive, timeout, status, "Запускаю Ollama…", interval=1.0)


def ensure_ollama(status=print, timeout=None):
    """Проверяет Ollama и при необходимости поднимает её скрытым процессом."""
    if timeout is None:
        timeout = OLLAMA_START_TIMEOUT
    if ollama_alive():
        emit_status(status, "Ollama: ок.")
        emit_status(status, "Ollama готова.")
        return True

    emit_status(status, "Ollama не отвечает. Запускаю её сама...")
    if os.name != "nt":
        emit_status(status, "Автозапуск настроен для Windows. Запусти Ollama вручную.")
        return False

    if not start_ollama_windows(ollama_env(), status):
        emit_status(status, "Не удалось запустить Ollama. Запусти её вручную и перезапусти Алису.")
        return False

    if wait_for_ollama(status, timeout):
        emit_status(status, "Ollama готова.")
        return True

    emit_status(status, "Ollama не ответила вовремя. Проверь, что она запущена, и перезапусти Алису.")
    return False


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
    parts = ["/модель", "/модель <имя>", "/статус"]
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
    ensure_ollama(print)
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
            for p in root.rglob(fname):
                return p
        except OSError:
            continue
    p = Path.home() / fname
    return p if p.exists() else None


def calibrate_threshold(status=None):
    """Меряет шум комнаты и выставляет порог срабатывания на речь."""
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
    global MODEL
    print("=== Голосовая Алиса ===")
    if not VOICE_OK:
        print(f"Голосовой стек не установлен: {VOICE_ERR}")
        print("Выполни:  pip install faster-whisper piper-tts sounddevice numpy")
        return 1

    ensure_ollama(print)
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
            if not VOICE_OK:
                self.completed.emit("Голосовой стек не установлен. Проверь микрофон и зависимости.", False)
                return
            try:
                self.status.emit("Грузим распознавание речи…")
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
        def __init__(self):
            super().__init__()
            self.setWindowTitle(f"Алиса {ALICE_VERSION} — локальный ассистент")
            self.resize(860, 620)
            self.setMinimumSize(560, 430)

            self.history = [{"role": "system", "content": SYSTEM}]
            self.ollama_ready = False
            self.busy = False
            self.important_statuses = set()
            self.assistant_open = False
            self.live_marker = None
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
            self.input.returnPressed.connect(self.send_message)
            row.addWidget(self.input, 1)

            self.send_button = QPushButton("Отправить", self)
            self.send_button.clicked.connect(self.send_message)
            row.addWidget(self.send_button)

            self.action_buttons = [self.send_button]

            if FEATURES["deepseek"]:
                self.deepseek_button = QPushButton("DeepSeek", self)
                self.deepseek_button.setToolTip("Отправить вопрос в DeepSeek")
                self.deepseek_button.clicked.connect(lambda: self.send_message(True))
                row.addWidget(self.deepseek_button)
                self.action_buttons.append(self.deepseek_button)

            if FEATURES["deepseek"] or FEATURES["vk_music"]:
                self.browser_button = QPushButton("Браузер", self)
                self.browser_button.setToolTip("Один раз настроить профиль браузера ассистента")
                self.browser_button.clicked.connect(self.open_browser_setup)
                row.addWidget(self.browser_button)
                self.action_buttons.append(self.browser_button)

            if FEATURES["voice_gui"]:
                self.mic_button = QPushButton("🎙", self)
                self.mic_button.setToolTip("Записать голосом — текст попадёт в поле ввода")
                self.mic_button.clicked.connect(self.start_voice_capture)
                row.addWidget(self.mic_button)
                self.action_buttons.append(self.mic_button)

            layout.addLayout(row)

            self.status_label = QLabel("Проверяю Ollama...", self)
            layout.addWidget(self.status_label)

            self.append_status(f"Алиса {ALICE_VERSION} запущена. Проверяю Ollama...")
            self.set_busy(True, "Проверяю Ollama...")
            self.input.setFocus()
            QTimer.singleShot(0, self.start_ollama_check)

            self.health_timer = QTimer(self)
            self.health_timer.timeout.connect(self.start_health_check)
            self.health_timer.start(HEALTH_INTERVAL_MS)
            QTimer.singleShot(0, self.start_health_check)

        # ---------- лента ----------
        def clear_live_status(self):
            self.live_marker = None

        def set_live_status(self, text):
            try:
                cursor = self.chat.textCursor()
                cursor.movePosition(QTextCursor.End)
                if self.live_marker is None:
                    self.live_marker = cursor.position() + 1
                    self.chat.append(f"Система: {text}")
                else:
                    cursor.setPosition(self.live_marker, QTextCursor.MoveAnchor)
                    cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
                    cursor.removeSelectedText()
                    cursor.insertText(f"Система: {text}")
            except Exception:
                self.chat.append(f"Система: {text}")
                self.live_marker = None
            self.scroll_down()

        @Slot(str)
        def append_status(self, text):
            self.clear_live_status()
            self.chat.append(f"Система: {text}")
            self.scroll_down()

        def append_error(self, text):
            self.clear_live_status()
            self.chat.appendHtml(
                f'<span style="color:#ff6b6b;">Система: {html.escape(text)}</span>'
            )
            self.scroll_down()

        def append_user(self, text):
            self.clear_live_status()
            self.chat.append(f"Ты: {text}")
            self.scroll_down()

        def append_deepseek(self, text):
            self.clear_live_status()
            self.chat.append(f"DeepSeek: {text}")
            self.scroll_down()

        def begin_assistant(self):
            self.clear_live_status()
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
            self.clear_live_status()
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
            for button in self.action_buttons:
                button.setEnabled(not busy)
            if busy:
                self.important_statuses.clear()
            if label:
                self.status_label.setText(label)

        @Slot(str)
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
                ) if text.startswith(prefix)
            ), None)
            if key and key not in self.important_statuses:
                self.append_status(text)
                self.important_statuses.add(key)
            else:
                self.set_live_status(text)

        # ---------- Ollama ----------
        def start_ollama_check(self):
            self.set_busy(True, "Проверяю Ollama...")
            thread = QThread(self)
            worker = StartupWorker()
            worker.moveToThread(thread)
            worker.status.connect(self.on_progress)
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
            self.clear_live_status()
            self.ollama_ready = ok
            if ok:
                self.append_status("Ollama готова. Можно писать.")
                self.set_busy(False, "Ollama: ок")
            else:
                self.append_error("Ollama недоступна. Запусти её вручную и перезапусти Алису.")
                self.set_busy(False, "Ollama: ошибка")

        # ---------- здоровье ----------
        def update_indicator(self, label, name, state):
            colors = {"ok": "#4ec9b0", "warn": "#dcdcaa", "err": "#f14c4c"}
            label.setText(f"● {name}: {state_word(state)}")
            label.setStyleSheet(f"color: {colors.get(state, '#dcdcaa')};")

        def start_health_check(self):
            if self.health_busy:
                return
            self.health_busy = True
            thread = QThread(self)
            worker = HealthWorker(max(0, len(self.history) - 1), self.browser_health)
            worker.moveToThread(thread)
            worker.result.connect(self.on_health_result)
            worker.result.connect(thread.quit)
            worker.result.connect(worker.deleteLater)
            thread.started.connect(worker.run)
            thread.finished.connect(thread.deleteLater)
            self.health_thread = thread
            self.health_worker = worker
            thread.start()

        @Slot(str, str, str, str)
        def on_health_result(self, ollama_state, model_state, browser_state, text):
            self.health_busy = False
            self.health = {"ollama": ollama_state, "model": model_state, "browser": browser_state}
            self.health_text = text
            if FEATURES["status_bar"]:
                self.update_indicator(self.ollama_indicator, "Ollama", ollama_state)
                self.update_indicator(self.model_indicator, "Модель", model_state)
                self.update_indicator(self.browser_indicator, "Браузер", browser_state)

        # ---------- браузер ----------
        def ensure_browser_worker(self):
            if not playwright_available():
                self.append_error("Playwright не установлен. Выполни: pip install playwright")
                return False
            if self.browser_worker is None:
                self.browser_worker = BrowserWorker()
                self.browser_thread = QThread(self)
                self.browser_worker.moveToThread(self.browser_thread)
                self.browser_worker.status.connect(self.on_progress)
                self.browser_worker.result.connect(self.on_browser_result)
                self.browser_thread.started.connect(self.browser_worker.run)
                self.browser_thread.finished.connect(self.browser_thread.deleteLater)
                self.browser_thread.start()
            return True

        @Slot()
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

        def start_browser_intent(self, intent):
            if not self.ensure_browser_worker():
                return
            self.clear_live_status()
            self.set_busy(True, "Открываю браузер…")
            self.browser_worker.submit(intent)

        @Slot(str, bool, str)
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
            worker.chunk.connect(self.append_chunk)
            worker.status.connect(self.on_progress)
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
            self.clear_live_status()
            self.finish_assistant()
            if ok:
                if payload:
                    self.history.append({"role": "assistant", "content": payload})
                self.set_busy(False, "Готово")
            else:
                drop_dangling(self.history)
                self.append_error(payload)
                self.set_busy(False, "Ошибка LLM")
            self.input.setFocus()

        # ---------- команды ----------
        def start_model_list(self):
            self.set_busy(True, "Проверяю модели…")
            thread = QThread(self)
            worker = FunctionWorker(lambda: "\n".join(f"  - {n}" for n in list_models()) or "Скачанных моделей нет.")
            worker.moveToThread(thread)
            worker.completed.connect(self.on_model_list)
            worker.completed.connect(thread.quit)
            worker.completed.connect(worker.deleteLater)
            thread.started.connect(worker.run)
            thread.finished.connect(thread.deleteLater)
            self.chat_thread = thread
            self.chat_worker = worker
            thread.start()

        @Slot(str, bool)
        def on_model_list(self, payload, ok):
            if ok:
                self.append_status("Скачанные модели:\n" + payload)
            else:
                self.append_error(payload)
            self.set_busy(False, "Готово")

        @Slot()
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
        def start_voice_capture(self):
            if self.busy:
                return
            if not FEATURES["voice_gui"]:
                return
            if not VOICE_OK:
                self.append_error("Голосовой стек не установлен. Проверь микрофон и зависимости.")
                return
            self.clear_live_status()
            self.set_busy(True, "Слушаю…")
            thread = QThread(self)
            worker = VoiceWorker()
            worker.moveToThread(thread)
            worker.status.connect(self.on_progress)
            worker.completed.connect(self.on_voice_result)
            worker.completed.connect(thread.quit)
            worker.completed.connect(worker.deleteLater)
            thread.started.connect(worker.run)
            thread.finished.connect(thread.deleteLater)
            self.voice_thread = thread
            self.voice_worker = worker
            thread.start()

        @Slot(str, bool)
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

        def closeEvent(self, event):
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
    return app.exec()


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
