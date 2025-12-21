from typing import Optional

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QLabel, QSlider, QHBoxLayout, QVBoxLayout, QErrorMessage


class QLabelSlider(QWidget):
    valueChanged = Signal(int)

    def __init__(self, text: str, orientation: Qt.Orientation, parent: Optional[QWidget] = None):
        super().__init__(parent=parent)
        self._text = text

        if orientation == Qt.Orientation.Horizontal:
            layout = QHBoxLayout()
            self.setLayout(layout)
        elif orientation == Qt.Orientation.Vertical:
            layout = QVBoxLayout()
            self.setLayout(layout)
        else:
            raise QErrorMessage("Orientation must be 'Horizontal' or 'Vertical'")

        self._slider = QSlider(orientation=orientation)
        self._label = QLabel(f"{self._text}: {self._slider.value()}")
        layout.addWidget(self._label)
        layout.addWidget(self._slider)

        self._slider.valueChanged.connect(self._on_valueChanged)

    def _on_valueChanged(self, value: int):
        self.valueChanged.emit(value)
        #self._label.setText(f"{self._text}: {value}")
        self._label.setText(f"{self._text}")

    def setMinimum(self, value: int):
        self._slider.setMinimum(value)

    def setMaximum(self, value: int):
        self._slider.setMaximum(value)

    def value(self) -> int:
        return self._slider.value()

    def setValue(self, value):
        self._slider.setValue(value)


class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Elements
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.test = QLabelSlider('Hello', Qt.Orientation.Horizontal)
        layout.addWidget(self.test)


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())
