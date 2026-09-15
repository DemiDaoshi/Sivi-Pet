import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QComboBox, QFrame, QHBoxLayout,
                             QLabel, QLineEdit, QMessageBox, QPushButton,
                             QTextEdit, QVBoxLayout, QWidget)

from core.chat_engine import ChatEngine
from core.history_manager import HistoryManager, load_profiles
from core.lm_client import LmClient, LmClientError
from core.rag_manager import RagManager
from core.tool_manager import ToolManager

COLOR_IDLE = "#4CAF50"   # зелёный
COLOR_BUSY = "#FF9800"   # оранжевый
COLOR_ERROR = "#F44336"  # красный


class RequestThread(QThread):
    """Выполняет один ход диалога, чтобы не морозить интерфейс."""

    # Не перекрываем встроенный QThread.finished.
    responded = pyqtSignal(str, bool)  # ответ, был ли использован RAG
    failed = pyqtSignal(str)

    def __init__(self, engine: ChatEngine, text: str):
        super().__init__()
        self.engine = engine
        self.text = text

    def run(self):
        try:
            result = self.engine.send(self.text)
            self.responded.emit(result.answer, result.used_rag)
        except LmClientError as e:
            self.failed.emit(str(e))
        except Exception as e:  # поток не должен падать молча
            self.failed.emit(f"{type(e).__name__}: {e}")


class ChatWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.client = LmClient()
        self.rag = RagManager()
        self.history = None
        self.engine = None
        self.thread = None

        self.profiles = load_profiles()
        self.initUI()

        first_profile = next(iter(self.profiles.keys()), "default")
        self.load_profile(first_profile)

    def initUI(self):
        self.setWindowTitle("Sivi — Твой ассистент")
        self.setGeometry(300, 300, 560, 520)

        self.status_indicator = QFrame(self)
        self.status_indicator.setFixedSize(16, 16)
        self.status_label = QLabel("Готов")
        self.status_label.setWordWrap(True)

        self.profile_combo = QComboBox()
        for name in self.profiles.keys():
            self.profile_combo.addItem(name)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Введите сообщение и нажмите Enter...")

        self.send_button = QPushButton("Отправить")
        self.save_button = QPushButton("Сохранить")
        self.clear_button = QPushButton("Забыть всё")

        status_row = QHBoxLayout()
        status_row.addWidget(self.status_indicator)
        status_row.addWidget(self.status_label, 1)

        profile_row = QHBoxLayout()
        profile_row.addWidget(QLabel("Профиль:"))
        profile_row.addWidget(self.profile_combo, 1)

        buttons_row = QHBoxLayout()
        buttons_row.addWidget(self.send_button)
        buttons_row.addWidget(self.save_button)
        buttons_row.addWidget(self.clear_button)

        layout = QVBoxLayout()
        layout.addLayout(status_row)
        layout.addLayout(profile_row)
        layout.addWidget(QLabel("История диалога:"))
        layout.addWidget(self.chat_display, 1)
        layout.addWidget(self.input_field)
        layout.addLayout(buttons_row)
        self.setLayout(layout)

        self.send_button.clicked.connect(self.on_send)
        self.save_button.clicked.connect(self.on_save)
        self.clear_button.clicked.connect(self.on_clear)
        self.input_field.returnPressed.connect(self.on_send)
        self.profile_combo.currentTextChanged.connect(self.on_profile_changed)

        self.set_status(COLOR_IDLE, self._startup_status())

    def _startup_status(self) -> str:
        if self.rag.available:
            return "Готов. База знаний доступна (RAG индексируется при первом запросе)."
        return ("Готов. RAG недоступен: нет пакетов "
                f"{', '.join(self.rag.missing_dependencies)}.")

    # --- Профили и история -------------------------------------------------

    def load_profile(self, profile_name: str):
        self.history = HistoryManager(profile=profile_name)
        self.engine = ChatEngine(self.client, self.history, ToolManager(self.rag))
        self.update_display()

    def on_profile_changed(self, profile_name: str):
        if not profile_name or self.thread is not None:
            return
        if self.history is not None:
            self.history.force_save()
        self.load_profile(profile_name)

    def update_display(self):
        """Перерисовывает историю в текстовом поле."""
        self.chat_display.clear()
        for msg in self.history.messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                self.chat_display.append(f"Ты: {content}")
            elif role == "assistant":
                self.chat_display.append(f"Ассистент: {content}")
            # системные сообщения (промпт) не показываем

    # --- Действия ----------------------------------------------------------

    def on_send(self):
        text = self.input_field.text().strip()
        if not text or self.thread is not None:
            return

        self.input_field.clear()
        self.set_busy(True)

        self.thread = RequestThread(self.engine, text)
        self.thread.responded.connect(self.on_response)
        self.thread.failed.connect(self.on_error)
        self.thread.finished.connect(self.on_thread_finished)
        self.thread.start()

    def on_save(self):
        if self.history is not None:
            self.history.force_save()
            self.set_status(COLOR_IDLE, f"История сохранена в {os.path.basename(self.history.filepath)}")

    def on_clear(self):
        if self.history is None:
            return
        answer = QMessageBox.question(
            self, "Забыть всё", "Удалить всю историю этого профиля?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            self.history.clear()
            self.update_display()
            self.set_status(COLOR_IDLE, "История очищена.")

    # --- Обратные вызовы потока -------------------------------------------

    def on_response(self, answer: str, used_rag: bool):
        self.update_display()  # история уже обновлена движком внутри потока
        self.history.force_save()
        marker = " (использован RAG)" if used_rag else ""
        self.set_status(COLOR_IDLE, f"Готов{marker}")

    def on_error(self, error_msg: str):
        self.update_display()
        self.set_status(COLOR_ERROR, f"Ошибка: {error_msg}")

    def on_thread_finished(self):
        self.thread = None
        self.set_busy(False)

    # --- Вспомогательное ---------------------------------------------------

    def set_busy(self, busy: bool):
        self.send_button.setEnabled(not busy)
        self.input_field.setEnabled(not busy)
        self.profile_combo.setEnabled(not busy)
        if busy:
            self.set_status(COLOR_BUSY, "Думаю...")

    def set_status(self, color: str, text: str):
        self.status_indicator.setStyleSheet(
            f"background-color: {color}; border-radius: 8px;"
        )
        self.status_label.setText(text)

    def closeEvent(self, event):
        if self.thread is not None and self.thread.isRunning():
            # Ждём поток, иначе Qt убьёт его при выходе.
            if not self.thread.wait(15000):
                self.thread.terminate()
                self.thread.wait(2000)
        if self.history is not None:
            self.history.force_save()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec_())
