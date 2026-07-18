"""Telegram notifier: pushes alert snapshots to your chat and records your
verdict (real threat / false alarm) back into the event log.

Setup (once):
  1. In Telegram, talk to @BotFather -> /newbot -> copy the bot token.
  2. Paste the token in the dashboard's Telegram card and save.
  3. Open your new bot's chat and press Start — the chat is auto-detected.

Config persists in telegram.json (gitignored). Every alert at or above the
minimum level is sent as a photo with two buttons; pressing one stores the
verdict on the event row, which doubles as labeled data for retraining.
"""

import json
import os
import threading
import time
from pathlib import Path

import requests

from .config import BASE_DIR, SNAPSHOT_DIR

CONFIG_PATH = BASE_DIR / "telegram.json"
LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
LEVEL_EMOJI = {"LOW": "🟡", "MEDIUM": "🟠", "HIGH": "🔴", "CRITICAL": "🟣"}


class TelegramNotifier:
    def __init__(self, on_verdict=None):
        self.token: str | None = None
        self.chat_id: str | None = None
        self.min_level = "MEDIUM"
        self.on_verdict = on_verdict          # callback(event_id, verdict)
        self.last_error: str | None = None
        self._offset = 0
        self._poller: threading.Thread | None = None
        self.load()

    # ----------------------------------------------------------- config --
    def load(self):
        if CONFIG_PATH.exists():
            try:
                cfg = json.loads(CONFIG_PATH.read_text())
            except Exception:
                cfg = {}
            self.token = cfg.get("token") or None
            self.chat_id = cfg.get("chat_id") or None
            self.min_level = cfg.get("min_level") or "MEDIUM"
        # Env-var fallback for headless deployments (no dashboard access)
        self.token = self.token or os.environ.get("TELEGRAM_BOT_TOKEN") or None
        self.chat_id = (self.chat_id
                        or os.environ.get("TELEGRAM_CHAT_ID") or None)
        self._ensure_poller()

    def save(self):
        CONFIG_PATH.write_text(json.dumps({
            "token": self.token, "chat_id": self.chat_id,
            "min_level": self.min_level,
        }, indent=2))

    def configure(self, token=None, chat_id=None, min_level=None):
        if token is not None:
            self.token = token.strip() or None
            self._offset = 0
            self.last_error = None
        if chat_id is not None:
            self.chat_id = str(chat_id).strip() or None
        if min_level in LEVELS:
            self.min_level = min_level
        self.save()
        self._ensure_poller()

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    def status(self) -> dict:
        return {
            "configured": self.enabled,
            "chat_id": self.chat_id,
            "min_level": self.min_level,
            "error": self.last_error,
        }

    # -------------------------------------------------------------- api --
    def _api(self, method, files=None, http_timeout=15, **params):
        url = f"https://api.telegram.org/bot{self.token}/{method}"
        # Nested structures (reply_markup) must be JSON strings in form data
        data = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                for k, v in params.items()}
        r = requests.post(url, data=data, files=files, timeout=http_timeout)
        payload = r.json()
        if not payload.get("ok"):
            raise RuntimeError(payload.get("description", f"HTTP {r.status_code}"))
        return payload["result"]

    def send_test(self) -> dict:
        if not self.enabled:
            return {"ok": False, "error": "No bot token configured"}
        if not self.chat_id:
            return {"ok": False, "error":
                    "No chat detected yet — open your bot in Telegram and press Start"}
        try:
            self._api("sendMessage", chat_id=self.chat_id,
                      text="🔔 Test alert from AI CCTV Threat Monitor — "
                           "notifications are working.")
            self.last_error = None
            return {"ok": True}
        except Exception as e:
            self.last_error = str(e)
            return {"ok": False, "error": str(e)}

    # ------------------------------------------------------------ alerts --
    def send(self, alert: dict):
        """Notifier interface: fire-and-forget so the video pipeline never
        blocks on the network."""
        if not self.enabled or not self.chat_id:
            return
        if LEVELS.index(alert["level"]) < LEVELS.index(self.min_level):
            return
        threading.Thread(target=self._send_alert, args=(alert,),
                         daemon=True).start()

    def _send_alert(self, alert: dict):
        caption = (f"{LEVEL_EMOJI.get(alert['level'], '⚠️')} "
                   f"{alert['level']}: {alert['message']}\n"
                   f"🕒 {alert['ts'].replace('T', ' ')}\n\n"
                   f"Is this a real threat?")
        keyboard = {"inline_keyboard": [[
            {"text": "✅ Real threat",
             "callback_data": f"fb:{alert.get('id', 0)}:real"},
            {"text": "🔕 False alarm",
             "callback_data": f"fb:{alert.get('id', 0)}:false"},
        ]]}
        try:
            snapshot = alert.get("snapshot")
            path = SNAPSHOT_DIR / snapshot if snapshot else None
            if path is not None and path.exists():
                with open(path, "rb") as f:
                    self._api("sendPhoto", files={"photo": f},
                              chat_id=self.chat_id, caption=caption,
                              reply_markup=keyboard, http_timeout=30)
            else:
                self._api("sendMessage", chat_id=self.chat_id, text=caption,
                          reply_markup=keyboard)
            self.last_error = None
        except Exception as e:
            self.last_error = str(e)
            print(f"Telegram send failed: {e}")

    # ---------------------------------------------------------- feedback --
    def _ensure_poller(self):
        if self.enabled and (self._poller is None
                             or not self._poller.is_alive()):
            self._poller = threading.Thread(target=self._poll_loop,
                                            daemon=True)
            self._poller.start()

    def _poll_loop(self):
        """Long-poll getUpdates: auto-detect the owner's chat on /start and
        record verdict button presses."""
        while True:
            if not self.enabled:
                time.sleep(3)
                continue
            try:
                updates = self._api(
                    "getUpdates", offset=self._offset + 1, timeout=25,
                    allowed_updates=["message", "callback_query"],
                    http_timeout=35)
                for update in updates:
                    self._offset = max(self._offset, update["update_id"])
                    self._handle_update(update)
                self.last_error = None
            except Exception as e:
                self.last_error = str(e)
                time.sleep(20)   # bad token / network — don't hammer the API

    def _handle_update(self, update: dict):
        message = update.get("message")
        if message and not self.chat_id:
            # First person to message the bot becomes the alert recipient
            self.chat_id = str(message["chat"]["id"])
            self.save()
            try:
                self._api("sendMessage", chat_id=self.chat_id,
                          text="✅ This chat will now receive CCTV threat "
                               "alerts. You can send a test from the "
                               "dashboard.")
            except Exception:
                pass
            return

        query = update.get("callback_query")
        if not query:
            return
        try:
            _, event_id, verdict = query["data"].split(":")
            verdict = "real" if verdict == "real" else "false_alarm"
            if self.on_verdict:
                self.on_verdict(int(event_id), verdict)
            self._api("answerCallbackQuery", callback_query_id=query["id"],
                      text="Recorded — thanks!")
            # Stamp the verdict onto the message and drop the buttons
            msg = query.get("message", {})
            stamp = ("☑️ You marked this as a REAL THREAT"
                     if verdict == "real"
                     else "🔕 You marked this as a FALSE ALARM")
            old_caption = msg.get("caption") or msg.get("text") or ""
            new_caption = old_caption.replace("\n\nIs this a real threat?", "")
            new_caption += f"\n\n{stamp}"
            edit = ("editMessageCaption" if msg.get("caption")
                    else "editMessageText")
            kwargs = ({"caption": new_caption} if msg.get("caption")
                      else {"text": new_caption})
            self._api(edit, chat_id=msg["chat"]["id"],
                      message_id=msg["message_id"], **kwargs)
        except Exception as e:
            print(f"Telegram feedback handling failed: {e}")
