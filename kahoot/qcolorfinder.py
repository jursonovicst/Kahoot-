import json
from typing import Tuple

import cv2
import numpy as np
import pyautogui
from PySide6 import QtWidgets
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QKeyEvent, QCloseEvent
from PySide6.QtWidgets import QPushButton, QWidget, QApplication, QSlider, QHBoxLayout, QVBoxLayout

from kahoot import QLabelSlider


class QColorFinder(QWidget):
    rgb = Signal(int, int, int)
    acc = Signal(int, int, int)
    kernel = Signal(int)
    crop = Signal(int, int)

    @staticmethod
    def rgb2hsv(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
        hsv = cv2.cvtColor(np.uint8([[[*rgb]]]), cv2.COLOR_RGB2HSV)[0, 0]
        return hsv[0], hsv[1], hsv[2]

    def __init__(self, key: str, rgb: Tuple[int, int, int], acc: Tuple[int, int, int], kernel: int,
                 crop: Tuple[int, int]):
        super().__init__()

        self._key = key
        self._rgb = rgb

        self.setFixedWidth(400)
        l1 = QVBoxLayout()
        self.setLayout(l1)
        self._colorpicker_b = QPushButton()
        l1.addWidget(self._colorpicker_b)

        l2 = QHBoxLayout()
        l1.addLayout(l2)

        self._acc_s_h = QSlider(Qt.Orientation.Horizontal)
        self._acc_s_h.setMinimum(0)
        self._acc_s_h.setMaximum(20)
        self._acc_s_h.setValue(acc[0])
        self._acc_s_s = QSlider(Qt.Orientation.Horizontal)
        self._acc_s_s.setMinimum(0)
        self._acc_s_s.setMaximum(100)
        self._acc_s_s.setValue(acc[1])
        self._acc_s_v = QSlider(Qt.Orientation.Horizontal)
        self._acc_s_v.setMinimum(0)
        self._acc_s_v.setMaximum(100)
        self._acc_s_v.setValue(acc[2])
        l2.addWidget(self._acc_s_h)
        l2.addWidget(self._acc_s_s)
        l2.addWidget(self._acc_s_v)

        self._kernel_ls = QLabelSlider("Kernel", Qt.Orientation.Horizontal)
        self._kernel_ls.setMinimum(2)
        self._kernel_ls.setMaximum(20)
        self._kernel_ls.setValue(kernel)
        l1.addWidget(self._kernel_ls)

        l3 = QHBoxLayout()
        l1.addLayout(l3)
        self._cropx_ls = QLabelSlider("Crop X", Qt.Orientation.Horizontal)
        self._cropx_ls.setMinimum(0)
        self._cropx_ls.setMaximum(100)
        self._cropx_ls.setValue(crop[0])
        self._cropy_ls = QLabelSlider("Crop Y", Qt.Orientation.Horizontal)
        self._cropy_ls.setMinimum(0)
        self._cropy_ls.setMaximum(100)
        self._cropy_ls.setValue(crop[1])
        l3.addWidget(self._cropx_ls)
        l3.addWidget(self._cropy_ls)

        self._colorpicker_b.setStyleSheet(f"background-color:rgb({self._rgb[0]},{self._rgb[1]},{self._rgb[2]})")
        self._colorpicker_b.setText(self._label_colorpicker())

        self._colorpicker_b.clicked.connect(self._on_colorpicker)
        self._acc_s_h.valueChanged.connect(self._on_acc)
        self._acc_s_s.valueChanged.connect(self._on_acc)
        self._acc_s_v.valueChanged.connect(self._on_acc)
        self._kernel_ls.valueChanged.connect(self._on_kernel)
        self._cropx_ls.valueChanged.connect(self._on_crop)
        self._cropy_ls.valueChanged.connect(self._on_crop)

    @property
    def hsv(self) -> Tuple[int, int, int]:
        return QColorFinder.rgb2hsv(self._rgb)

    def _on_colorpicker(self):
        self._colorpicker_b.setEnabled(False)
        self._colorpicker_b.setText("Press SPACE to capture")
        QApplication.instance().installEventFilter(self)

    def _on_acc(self, _: int):
        self.acc.emit(self._acc_s_h.value(), self._acc_s_s.value(), self._acc_s_v.value())
        self._colorpicker_b.setText(self._label_colorpicker())

    def _on_kernel(self, _: int):
        self.kernel.emit(self._kernel_ls.value())

    def _on_crop(self, _: int):
        self.crop.emit(self._cropx_ls.value(), self._cropy_ls.value())

    def _label_colorpicker(self):
        h, s, v = self.hsv
        return f"{self._key} - HSV: {h}±{self._acc_s_h.value()}°, {s}±{self._acc_s_s.value()}, {v}±{self._acc_s_v.value()}"

    def eventFilter(self, obj, event):
        if isinstance(event, QKeyEvent):
            if event.key() == 32 and event.type() == 51:
                QApplication.instance().removeEventFilter(self)
                x, y = pyautogui.position()
                self._rgb = pyautogui.pixel(x, y)
                self.rgb.emit(*self._rgb)
                self._colorpicker_b.setText(self._label_colorpicker())
                self._colorpicker_b.setStyleSheet(f"background-color:rgb({self._rgb[0]},{self._rgb[1]},{self._rgb[2]})")

                self._colorpicker_b.setEnabled(True)
                return True
        return False

    def getrgb(self) -> Tuple[int, int, int]:
        return self._rgb

    def getacc(self) -> Tuple[int, int, int]:
        return self._acc_s_h.value(), self._acc_s_s.value(), self._acc_s_v.value()

    def getkernel(self) -> int:
        return self._kernel_ls.value()

    def getcrop(self) -> Tuple[int, int]:
        return self._cropx_ls.value(), self._cropy_ls.value()

    def serialize(self) -> dict:
        return {"rgb": self._rgb,
                "acc": (self._acc_s_h.value(), self._acc_s_s.value(), self._acc_s_v.value()),
                "kernel": self._kernel_ls.value(),
                "crop": (self._cropx_ls.value(), self._cropy_ls.value())}

    @classmethod
    def unserialize(cls, key: str, data: dict) -> "QColorFinder":
        return QColorFinder(key, data['rgb'], data['acc'], data['kernel'], data['crop'])


class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Elements
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        with open("/tmp/qcolorfinder.json", "rt") as f:
            self.test = QColorFinder.unserialize('H', data=json.load(f))
            layout.addWidget(self.test)

    def closeEvent(self, event: QCloseEvent):
        with open("/tmp/qcolorfinder.json", "wt") as f:
            json.dump(self.test.serialize(), f)


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())
