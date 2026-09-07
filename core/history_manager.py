# core/history_manager.py (полностью переписанный)
import json
import os

class HistoryManager:
    def __init__(self, profile="default", filepath="history.json"):
        self.filepath = filepath
        self.messages = []
        
        self.autosave_interval = 3
        self._message_counter = 0
        
        
        with open('profiles.json', 'r', encoding='utf-8') as f:
            profiles = json.load(f)
        
        self.profile = profile
        
        if profile in profiles and "system_prompt" in profiles[profile]:
            self.system_prompt = profiles[profile]["system_prompt"]
        else:
            self.system_prompt = profiles["default"]["system_prompt"]
        
            self.load()
        
        if len(self.messages) == 0:
            self._add_system_prompt_internal()

    def _add_system_prompt_internal(self):
        """Добавляет системный промпт только для отправки модели (не в messages для UI)"""
        if not self.messages or self.messages[0]["role"] != "system":
            self.messages.insert(0, {"role": "system", "content": self.system_prompt})
    
    def load(self):
        """Загружает историю из файла"""
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r', encoding='utf-8') as f:
                try:
                    self.messages = json.load(f)


                except json.JSONDecodeError:
                    self.messages = []

    def save(self):
        """Сохраняет историю в файл"""
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)
    
    def add_message(self, role: str, content: str):
        """Добавляет сообщение и автосохраняет каждые N сообщений"""
        self.messages.append({"role": role, "content": content})
        self._message_counter += 1
        
        if self._message_counter >= self.autosave_interval:
            self.save()
            self._message_counter = 0
    
    def clear(self):
        """Очищает историю и сохраняет"""
        self.messages = []
        self._message_counter = 0
        self.save()
    
    def force_save(self):
        """Форсирует сохранение (например, при закрытии окна)"""
        self.save()