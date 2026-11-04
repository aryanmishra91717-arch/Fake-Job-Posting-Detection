import os
import sys

import joblib
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fake_job_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except Exception as error:
    raise SystemExit(
        "Could not load the model files. "
        "Check that both .pkl files are in your project folder. "
        f"Details: {error}"
    ) from error


class JobDetectionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fake Job Detection System")
        self.setMinimumSize(600, 500)
        self.resize(700, 600)

        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(25, 20, 25, 20)
        layout.setSpacing(10)

        heading = QLabel("FAKE JOB DETECTION SYSTEM")
        heading.setObjectName("heading")
        layout.addWidget(heading)

        title_label = QLabel("Enter Job Title:")
        title_label.setObjectName("fieldLabel")
        layout.addWidget(title_label)

        self.title_entry = QLineEdit()
        self.title_entry.setPlaceholderText("e.g. Software Engineer")
        layout.addWidget(self.title_entry)

        description_label = QLabel("Enter Job Description:")
        description_label.setObjectName("fieldLabel")
        layout.addWidget(description_label)

        self.description_box = QTextEdit()
        self.description_box.setPlaceholderText("Paste the job description here")
        layout.addWidget(self.description_box, stretch=1)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        check_button = QPushButton("CHECK JOB")
        check_button.setObjectName("checkButton")
        check_button.clicked.connect(self.check_job)
        button_layout.addWidget(check_button)

        clear_button = QPushButton("CLEAR")
        clear_button.clicked.connect(self.clear_fields)
        button_layout.addWidget(clear_button)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        self.result_label = QLabel("Prediction will appear here")
        self.result_label.setObjectName("resultLabel")
        self.result_label.setWordWrap(True)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.result_label)

        self.setStyleSheet(
            """
            QWidget#centralWidget {
                background-color: #f0f3f7;
            }
            QLabel#heading {
                color: #17365d;
                font-size: 20px;
                font-weight: bold;
                padding-bottom: 8px;
            }
            QLabel#fieldLabel {
                color: #222222;
                font-size: 13px;
                font-weight: bold;
                padding-top: 4px;
            }
            QLineEdit, QTextEdit {
                background-color: white;
                color: #111111;
                border: 1px solid #b8c2ce;
                border-radius: 4px;
                padding: 8px;
                font-size: 13px;
            }
            QPushButton {
                padding: 9px 22px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton#checkButton {
                color: white;
                background-color: #17365d;
                border: none;
                border-radius: 4px;
            }
            QPushButton#checkButton:hover {
                background-color: #28578b;
            }
            QLabel#resultLabel {
                color: #17365d;
                font-size: 13px;
                font-weight: bold;
                padding-top: 6px;
            }
            """
        )
        self.title_entry.setFocus()

    def check_job(self):
        title = self.title_entry.text().strip()
        description = self.description_box.toPlainText().strip()

        if not title or not description:
            QMessageBox.warning(
                self,
                "Missing Information",
                "Please enter both the job title and description.",
            )
            return

        try:
            job_text = title + " " + description
            features = vectorizer.transform([job_text])
            prediction = model.predict(features)[0]
        except Exception as error:
            QMessageBox.critical(self, "Prediction Error", str(error))
            return

        if str(prediction) == "1":
            self.result_label.setText("POTENTIALLY FRAUDULENT JOB")
            self.result_label.setStyleSheet("color: #c62828;")
        else:
            self.result_label.setText("JOB APPEARS REAL")
            self.result_label.setStyleSheet("color: #237a36;")

    def clear_fields(self):
        self.title_entry.clear()
        self.description_box.clear()
        self.result_label.setText("Prediction will appear here")
        self.result_label.setStyleSheet("color: #17365d;")
        self.title_entry.setFocus()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = JobDetectionWindow()
    window.show()
    sys.exit(app.exec())
