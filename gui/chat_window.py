import html
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag_manager import preload_dependencies

# torch обязан загрузиться до PyQt5, иначе на Windows падает c10.dll (WinError 1114).
preload_dependencies()

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame,
                             QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                             QPushButton, QTextEdit, QVBoxLayout, QWidget)

from core.chat_engine import ChatEngine
from core.history_manager import HistoryManager, load_profiles
from core.lm_client import LmClient, LmClientError
from core.rag_manager import RagManager
from core.tool_manager import ToolManager
from eval.judge import Judge, verdict_line

COLOR_IDLE = "#4CAF50"   # зелёный
COLOR_BUSY = "#FF9800"   # оранжевый
COLOR_ERROR = "#F44336"  # красный

VERDICT_BG = {"good": "#E8F5E9", "mid": "#FFF8E1", "bad": "#FFEBEE"}
VERDICT_FG = {"good": "#1B5E20", "mid": "#E65100", "bad": "#B71C1C"}


def verdict_tone(verdict):
    """good / mid / bad по оценке судьи — для цвета блока."""
    if verdict is None or verdict.score is None:
        return "mid"
    if verdict.score >= 4:
        return "good"
    if verdict.score <= 2:
        return "bad"
    return "mid"


class RequestThread(QThread):
    """Выполняет один ход диалога, чтобы не морозить интерфейс."""

    # Не перекрываем встроенный QThread.finished.
    responded = pyqtSignal(str, bool, int, object)  # ответ, был ли RAG, фрагменты, источники
    failed = pyqtSignal(str)

    def __init__(self, engine: ChatEngine, text: str, force_rag: bool = False):
        super().__init__()
        self.engine = engine
        self.text = text
        self.force_rag = force_rag

    def run(self):
        try:
            result = self.engine.send(self.text, force_rag=self.force_rag)
            self.responded.emit(result.answer, result.used_rag, result.fragments, result.sources)
        except LmClientError as e:
            self.failed.emit(str(e))
        except Exception as e:  # поток не должен падать молча
            self.failed.emit(f"{type(e).__name__}: {e}")


class JudgeThread(QThread):
    """Отдельный запрос судьи, чтобы окно не подвисало на время оценки."""

    judged = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, judge: Judge, question: str, answer: str, contexts=(), reference=None):
        super().__init__()
        self.judge = judge
        self.question = question
        self.answer = answer
        self.contexts = contexts
        self.reference = reference

    def run(self):
        try:
            verdict = self.judge.evaluate(
                self.question, self.answer, self.contexts, reference=self.reference
            )
            self.judged.emit(verdict)
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
        self.judge = Judge(self.client)
        self.judge_thread = None
        self.last_question = None
        self.last_answer = None
        self.last_sources = ()

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

        self.rag_check = QCheckBox("Искать в базе знаний")
        self.rag_check.setToolTip(
            "Включено: перед ответом ищем по базе знаний по тексту вашего сообщения.\n"
            "Выключено: модель сама решает, вызвать ли поиск через [RAG: ...]."
        )

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Введите сообщение и нажмите Enter...")

        self.send_button = QPushButton("Отправить")
        self.judge_button = QPushButton("Оценить")
        self.judge_button.setToolTip(
            "Судья (та же модель, но в отдельном контексте) проверит последний ответ:\n"
            "отвечает ли он на вопрос и опирается ли на найденные фрагменты."
        )
        self.judge_button.setEnabled(False)
        self.save_button = QPushButton("Сохранить")
        self.clear_button = QPushButton("Забыть всё")

        status_row = QHBoxLayout()
        status_row.addWidget(self.status_indicator)
        status_row.addWidget(self.status_label, 1)

        profile_row = QHBoxLayout()
        profile_row.addWidget(QLabel("Профиль:"))
        profile_row.addWidget(self.profile_combo, 1)
        profile_row.addWidget(self.rag_check)

        buttons_row = QHBoxLayout()
        buttons_row.addWidget(self.send_button)
        buttons_row.addWidget(self.judge_button)
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
        self.judge_button.clicked.connect(self.on_judge)
        self.save_button.clicked.connect(self.on_save)
        self.clear_button.clicked.connect(self.on_clear)
        self.input_field.returnPressed.connect(self.on_send)
        self.profile_combo.currentTextChanged.connect(self.on_profile_changed)

        self.set_status(COLOR_IDLE, self._startup_status())
        self.rag_check.setEnabled(self.rag.available)

    def _startup_status(self) -> str:
        if self.rag.available:
            return "Готов. База знаний доступна (RAG индексируется при первом запросе)."
        return f"Готов. RAG недоступен: {self.rag.unavailable_reason()}."

    # --- Профили и история -------------------------------------------------

    def load_profile(self, profile_name: str):
        self.history = HistoryManager(profile=profile_name)
        self.engine = ChatEngine(self.client, self.history, ToolManager(self.rag))
        self.reset_last_turn()
        self.update_display()

    def reset_last_turn(self):
        self.last_question = None
        self.last_answer = None
        self.last_sources = ()
        self.judge_button.setEnabled(False)

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
        self.reset_last_turn()
        self.last_question = text
        self.set_busy(True)

        self.thread = RequestThread(self.engine, text, self.rag_check.isChecked())
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
            self.reset_last_turn()
            self.update_display()
            self.set_status(COLOR_IDLE, "История очищена.")

    # --- Обратные вызовы потока -------------------------------------------

    def on_response(self, answer: str, used_rag: bool, fragments: int, sources):
        self.update_display()  # история уже обновлена движком внутри потока
        self.history.force_save()
        self.last_answer = answer
        self.last_sources = tuple(sources or ())
        self.judge_button.setEnabled(bool(answer))
        if not self.rag.available:
            status = f"Готов. RAG недоступен: {self.rag.unavailable_reason()}."
        elif used_rag and fragments:
            status = f"Готов. RAG: {fragments} фрагм."
        elif used_rag:
            status = "Готов. Использован RAG."
        elif self.rag_check.isChecked():
            status = "Готов. В базе знаний ничего подходящего не найдено."
        else:
            status = "Готов"
        self.set_status(COLOR_IDLE, status)

    def on_error(self, error_msg: str):
        self.update_display()
        self.set_status(COLOR_ERROR, f"Ошибка: {error_msg}")

    def on_thread_finished(self):
        self.thread = None
        self.set_busy(False)

    # --- Судья -------------------------------------------------------------

    def on_judge(self):
        if self.judge_thread is not None or not self.last_answer:
            return

        self.set_busy(True)
        self.judge_button.setEnabled(False)
        self.status_label.setText("Судья проверяет последний ответ...")

        self.judge_thread = JudgeThread(
            self.judge, self.last_question or "", self.last_answer, self.last_sources
        )
        self.judge_thread.judged.connect(self.on_judged)
        self.judge_thread.failed.connect(self.on_judge_failed)
        self.judge_thread.finished.connect(self.on_judge_thread_finished)
        self.judge_thread.start()

    def on_judged(self, verdict):
        self.append_judge_block(verdict)
        score = verdict.score if verdict.score is not None else "-"
        self.set_status(COLOR_IDLE, f"Готов. Судья: оценка {score}/5.")

    def on_judge_failed(self, error_msg: str):
        self.append_html(
            f'<hr><font color="{COLOR_ERROR}">Судья не смог оценить: '
            f'{html.escape(error_msg)}</font>'
        )
        self.set_status(COLOR_ERROR, f"Судья: ошибка — {error_msg}")

    def on_judge_thread_finished(self):
        self.judge_thread = None
        self.set_busy(False)

    def append_judge_block(self, verdict):
        tone = verdict_tone(verdict)
        background = VERDICT_BG[tone]
        foreground = VERDICT_FG[tone]
        reason = html.escape(verdict.reason or "")
        block = (
            f'<hr><table width="100%" cellspacing="0" cellpadding="8" bgcolor="{background}">'
            f'<tr><td><b><font color="{foreground}">Судья</font></b>: '
            f'<font color="{foreground}">{html.escape(verdict_line(verdict))}</font>'
        )
        if reason:
            block += f'<br><font color="{foreground}">{reason}</font>'
        block += "</td></tr></table>"
        self.append_html(block)

    def append_html(self, markup: str):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertHtml(markup)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    # --- Вспомогательное ---------------------------------------------------

    def set_busy(self, busy: bool):
        self.send_button.setEnabled(not busy)
        self.input_field.setEnabled(not busy)
        self.profile_combo.setEnabled(not busy)
        self.rag_check.setEnabled(not busy and self.rag.available)
        self.judge_button.setEnabled(
            not busy and bool(self.last_answer) and self.judge_thread is None
        )
        if busy:
            self.set_status(COLOR_BUSY, "Думаю...")

    def set_status(self, color: str, text: str):
        self.status_indicator.setStyleSheet(
            f"background-color: {color}; border-radius: 8px;"
        )
        self.status_label.setText(text)

    def closeEvent(self, event):
        for thread in (self.thread, self.judge_thread):
            if thread is not None and thread.isRunning():
                # Ждём поток, иначе Qt убьёт его при выходе.
                if not thread.wait(15000):
                    thread.terminate()
                    thread.wait(2000)
        if self.history is not None:
            self.history.force_save()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec_())
