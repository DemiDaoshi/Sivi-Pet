# gui/chat_window.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from PyQt5.QtWidgets import (QApplication, QWidget, QLineEdit, QPushButton,
                             QVBoxLayout, QTextEdit, QFrame, QLabel, QComboBox)
from PyQt5.QtCore import QThread, pyqtSignal
from core.lm_client import LmClient
from core.history_manager import HistoryManager


class RequestThread(QThread):
    """Поток для HTTP-запроса, чтобы не морозить GUI."""
    finished = pyqtSignal(str)  
    error = pyqtSignal(str)     

    def __init__(self, client: LmClient, messages: list):
        super().__init__()
        self.client = client
        self.messages = messages
        self.history = HistoryManager(profile="default")

    def run(self):
        try:
            answer = self.client.send_message(self.messages)
            self.finished.emit(answer)
        except ConnectionError as e:
            self.error.emit(str(e))

class ChatWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.client = LmClient()
        
        with open('profiles.json', 'r') as f:
            profiles = json.load(f)
        profile_names = list(profiles.keys())
        self.profile_combo = QComboBox()
        for name in profile_names:
            self.profile_combo.addItem(name)
        self.profile_combo.setCurrentIndex(0)
        self.profile_combo.setCurrentIndex(0)
        profile_name = self.profile_combo.currentText()  
        self.current_profile = profile_name
        
        
        self.history = HistoryManager(profile=self.current_profile)
        
        self.initUI()  

    def initUI(self):
        self.setWindowTitle('Sivi — Твой ассистент')
        self.setGeometry(300, 300, 500, 450)

        layout = QVBoxLayout()
        profile_combo = QComboBox()
        for name in ['default', 'work']:
            profile_combo.addItem(name)
        profile_combo.setCurrentIndex(0)
        self.profile_combo = profile_combo

        self.status_indicator = QFrame(self)
        self.status_indicator.setFixedSize(30, 30)
        self.status_indicator.setStyleSheet("background-color: #4CAF50; border-radius: 15px;")

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.input_field = QLineEdit()
        self.profile_combo = QComboBox()  
        self.send_button = QPushButton("Отправить")
        self.save_button = QPushButton("Сохранить")
        self.clear_button = QPushButton("Забыть всё")

        layout = QVBoxLayout()
        layout.addWidget(QLabel("История диалога:"))
        layout.addWidget(self.status_indicator)
        layout.addWidget(self.chat_display)
        layout.addWidget(self.input_field)
        layout.addWidget(profile_combo)
        layout.addWidget(self.send_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.clear_button)
        self.setLayout(layout)

    
        self.update_display()

        
        self.send_button.clicked.connect(self.on_send)
        self.save_button.clicked.connect(self.on_save)
        self.input_field.returnPressed.connect(self.on_send)
        self.clear_button.clicked.connect(self.on_clear)

    def update_display(self):
        """Перерисовать историю в текстовом поле."""
        self.chat_display.clear()
        for msg in self.history.messages:
            role = "Ты" if msg["role"] == "user" else "Ассистент"
            self.chat_display.append(f"{role}: {msg['content']}")

    def on_send(self):
        text = self.input_field.text().strip()
        if not text:
            return
        self.input_field.clear()

     
        self.history.add_message("user", text)
        self.update_display()

        
        self.request_thread = RequestThread(self.client, self.history.messages)
        self.request_thread.finished.connect(self.on_response)
        self.request_thread.error.connect(self.on_error)
        self.send_button.setEnabled(False)   
        self.request_thread.start()
        self.status_indicator.setStyleSheet("background-color: #FF9800; border-radius: 15px;")

    def on_save(self):
        self.history.force_save()

    def on_response(self, answer: str):
        self.history.add_message("assistant", answer)
        self.update_display()
        self.send_button.setEnabled(True)
        self.history.force_save()   
        self.status_indicator.setStyleSheet("background-color: #4CAF50; border-radius: 15px;")

    def on_error(self, error_msg: str):
        self.chat_display.append(f"Ошибка: {error_msg}")
        self.send_button.setEnabled(True)
        self.status_indicator.setStyleSheet("background-color: #F44336; border-radius: 15px;")

    def on_clear(self):
        self.history.clear()
        self.update_display()

    def closeEvent(self, event):
        self.history.force_save()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec_())