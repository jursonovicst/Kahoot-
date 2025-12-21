import json
from typing import Tuple, Optional

import pyautogui
from PIL import Image, ImageGrab
from PySide6 import QtWidgets
from PySide6.QtCore import Signal, Slot, QRect, Qt
from PySide6.QtGui import QKeyEvent, QPainter, QColor, QPen, QFont, QCloseEvent, QBrush
from PySide6.QtWidgets import QPushButton, QApplication, QWidget, QVBoxLayout, QLabel, QHBoxLayout

from PIL.ImageQt import toqpixmap


class QSnapper(QWidget):
    # in - snapped image
    image = Signal(Image.Image)

    # in - area to mark (use 0 area rect to delete mark)
    mark = Signal(str, QRect)

    # in - result
    result = Signal(str)
    result_clear = Signal(str)

    # out - set snap rectangle
    rect = Signal(int, int, int, int)

    def __init__(self, tl: Tuple[int, int], br: Tuple[int, int]) -> None:
        super().__init__()
        self._tl = tl
        self._br = br

        self._image = None
        self._marks = {}
        self._result = None

        l1 = QVBoxLayout()
        self.setLayout(l1)

        l2 = QHBoxLayout()
        l1.addLayout(l2)

        self._snap_b = QPushButton("Set snap rectangle")
        self._snap_b.setFixedWidth(150)
        l2.addWidget(self._snap_b)
        self._snap_l = QLabel(f"{self._tl[0]},{self._tl[1]} - {self._br[0]},{self._br[1]}")
        self._snap_l.setFixedWidth(150)
        l2.addWidget(self._snap_l)

        self._image_l = QLabel()
        l1.addWidget(self._image_l)

        self._snap_b.clicked.connect(self._on_clicked)

        self.image.connect(self._on_image)
        self.mark.connect(self._on_mark)
        self.result.connect(self._on_result)
        self.result_clear.connect(self._on_result_clear)
        # self._slider.valueChanged.connect(self.on_valueChanged)
        # self.apply()

    def _on_clicked(self, e):
        self._snap_b.setEnabled(False)
        self._snap_l.setText(f"_,_ - _,_")
        self._snap_b.setText("Press SPACE for TL")
        self._tl = None
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, obj, event):
        if isinstance(event, QKeyEvent):
            if event.key() == 32 and event.type() == 51:
                if self._tl is None:
                    self._tl = pyautogui.position()
                    self._snap_l.setText(f"{self._tl[0]},{self._tl[1]} - _,_")
                    self._snap_b.setText("Press SPACE for BR")
                else:
                    QApplication.instance().removeEventFilter(self)
                    self._br = pyautogui.position()
                    self.rect.emit(*self._tl, *self._br)
                    self._snap_b.setText("Set snap rectangle")
                    self._snap_l.setText(f"{self._tl[0]},{self._tl[1]} - {self._br[0]},{self._br[1]}")
                    self._snap_b.setEnabled(True)
                return True
        return False

    @Slot(Image.Image)
    def _on_image(self, img: Image.Image):
        self._image = img
        self._redraw()

    @Slot(str, QRect)
    def _on_mark(self, key: str, mark: QRect):
        if mark.height() * mark.width() == 0 and key in self._marks:
            self._marks.pop(key)
        else:
            self._marks[key] = mark
        self._redraw()

    @Slot(str)
    def _on_result(self, result: str):
        self._result = result
        self._redraw()

    @Slot(str)
    def _on_result_clear(self, _: str):
        self._result = None
        self._redraw()

    def _redraw(self):
        pmap = toqpixmap(self._image)

        painter = QPainter(pmap)
        painter.setPen(QPen(QColor('purple'), 5))
        painter.setFont(QFont("Arial", 20))
        for key, mark in self._marks.items():
            if mark.height() * mark.width() > 0:
                if self._result is not None and self._result != key and key != 'Q':
                    painter.fillRect(mark, QBrush(QColor(10, 10, 10, 200)))
                    painter.drawText(mark, Qt.AlignmentFlag.AlignLeft, key)
                else:
                    painter.drawRect(mark)
                    painter.drawText(mark, Qt.AlignmentFlag.AlignLeft, key)
        self._image_l.setPixmap(pmap)

    @property
    def getrect(self) -> Tuple[int, int, int, int]:
        return *self._tl, *self._br

    def serialize(self) -> dict:
        return {"tl": self._tl,
                "br": self._br}

    @classmethod
    def unserialize(cls, data: dict) -> "QSnapper":
        return QSnapper(data["tl"], data["br"])


class Window(QtWidgets.QWidget):
    image = Signal(Image.Image)
    mark = Signal(str, QRect)

    def __init__(self):
        super().__init__()

        # Elements
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        with open("/tmp/qsnapper.json", "rt") as f:
            self.test = QSnapper.unserialize(data=json.load(f))
            layout.addWidget(self.test)

        self.image.connect(self.test.image)
        self.mark.connect(self.test.mark)
        self.test.rect.connect(self.on_rect)

        self._img = None

    def on_rect(self, x1: int, y1: int, x2: int, y2: int):
        with ImageGrab.grab((x1, y1, x2, y2)) as img_rgba:
            self.image.emit(img_rgba)
        self.mark.emit("X", QRect(10, 10, 50, 50))

    def closeEvent(self, event: QCloseEvent):
        with open("/tmp/qsnapper.json", "wt") as f:
            json.dump(self.test.serialize(), f)


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())
