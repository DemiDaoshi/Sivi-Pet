import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILES_PATH = os.path.join(PROJECT_ROOT, "profiles.json")

FALLBACK_SYSTEM_PROMPT = "Ты — Sivi, универсальный ассистент. Отвечай на русском, кратко и по делу."


def history_path_for_profile(profile: str) -> str:
    """default → history.json, остальные профили → history_<профиль>.json."""
    if profile == "default":
        return os.path.join(PROJECT_ROOT, "history.json")
    return os.path.join(PROJECT_ROOT, f"history_{profile}.json")


def load_profiles() -> dict:
    """profiles.json; при ошибке — профиль по умолчанию."""
    try:
        with open(PROFILES_PATH, "r", encoding="utf-8") as f:
            profiles = json.load(f)
        if isinstance(profiles, dict) and profiles:
            return profiles
    except (OSError, json.JSONDecodeError) as e:
        print(f"[Профили] Не удалось прочитать {PROFILES_PATH}: {e}")
    return {"default": {"name": "Sivi", "system_prompt": FALLBACK_SYSTEM_PROMPT}}


class HistoryManager:
    def __init__(self, profile="default", filepath=None, autosave_interval=3):
        self.profile = profile
        self.filepath = filepath or history_path_for_profile(profile)
        self.autosave_interval = autosave_interval
        self._message_counter = 0
        self.messages = []

        self.system_prompt = self._system_prompt_for(profile)
        self.load()
        self._ensure_system_prompt()

    def _system_prompt_for(self, profile: str) -> str:
        profiles = load_profiles()
        entry = profiles.get(profile) or profiles.get("default") or {}
        return entry.get("system_prompt") or FALLBACK_SYSTEM_PROMPT

    def _ensure_system_prompt(self):
        """Промпт профиля всегда стоит первым сообщением."""
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = self.system_prompt
        else:
            self.messages.insert(0, {"role": "system", "content": self.system_prompt})

    def load(self):
        if not os.path.exists(self.filepath):
            self.messages = []
            return
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            self.messages = []
            return

        if isinstance(data, list):
            self.messages = [m for m in data
                             if isinstance(m, dict) and "role" in m and "content" in m]
        else:
            self.messages = []

    def save(self):
        # Атомарная запись, чтобы не оставить битый файл.
        tmp_path = self.filepath + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.filepath)

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._message_counter += 1

        if self._message_counter >= self.autosave_interval:
            self.save()
            self._message_counter = 0

    def set_profile(self, profile: str):
        """Переключение профиля: другой промпт и своя история."""
        self.profile = profile
        self.filepath = history_path_for_profile(profile)
        self.system_prompt = self._system_prompt_for(profile)
        self._message_counter = 0
        self.load()
        self._ensure_system_prompt()

    def clear(self):
        self.messages = []
        self._message_counter = 0
        self._ensure_system_prompt()
        self.save()

    def force_save(self):
        self.save()
