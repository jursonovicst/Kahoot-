from typing import Tuple, Callable

import cv2
import numpy as np
import pyautogui
import pytesseract
from PIL import ImageGrab, Image, ImageQt
from PIL.ImageQt import ImageQt
from PyQt6 import QtWidgets
from PyQt6.QtCore import pyqtSignal as Signal, pyqtSlot as Slot, QRect, QByteArray, Qt, QObject, \
    QThread
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QBrush, QKeyEvent, QFont
from PyQt6.QtWidgets import QPushButton, QLabel, QGridLayout, QApplication, QSlider
from openai import OpenAI


class SnapPicker(QPushButton):
    rect = Signal(int, int, int, int)

    def __init__(self, text: str, t: int, l: int, b: int, r: int) -> None:
        super().__init__()
        self._text = text
        self._tl = t, l
        self._br = b, r
        self.apply()

    def mousePressEvent(self, e):
        self.setEnabled(False)
        self.setText("Press SPACE for TL")
        self._tl = None
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, obj, event):
        if isinstance(event, QKeyEvent):
            if event.key() == 32 and event.type() == 51:
                if self._tl is None:
                    self._tl = pyautogui.position()
                    self.setText("Press SPACE for BR")
                    return True
                else:
                    QApplication.instance().removeEventFilter(self)
                    self._br = pyautogui.position()
                    self.rect.emit(*self._tl, *self._br)
                    self.apply()
                    self.setEnabled(True)
                    return True
        return False

    def apply(self):
        self.setText(f"{self._text} ({self._tl[0]},{self._tl[1]} - {self._br[0]}, {self._br[1]})")

    @property
    def getrect(self) -> Tuple[int, int, int, int]:
        return *self._tl, *self._br


class QColorPicker(QPushButton):
    rgb = Signal(int, int, int)

    def __init__(self, text: str, r: int, g: int, b: int):
        self._text = text
        super().__init__(text)
        self._rgb = r, g, b
        self.apply()

    def mousePressEvent(self, e):
        self.setEnabled(False)
        self.setText("Press SPACE to capture")
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, obj, event):
        if isinstance(event, QKeyEvent):
            if event.key() == 32 and event.type() == 51:
                QApplication.instance().removeEventFilter(self)
                x, y = pyautogui.position()
                self._rgb = pyautogui.pixel(x, y)
                self.rgb.emit(*self._rgb)
                self.apply()
                self.setEnabled(True)
                return True
        return False

    def apply(self):
        self.setText(f"{self._text} ({self._rgb[0]},{self._rgb[1]},{self._rgb[2]})")
        self.setStyleSheet(f"background-color:rgb({self._rgb[0]},{self._rgb[1]},{self._rgb[2]})")

    def getrgb(self) -> Tuple[int, int, int]:
        return self._rgb


class QSnapLabel(QLabel):
    image = Signal(str, int, int, QByteArray)
    rect = Signal(str, QRect)
    result = Signal(str)

    def __init__(self):
        super().__init__()
        self.image.connect(self.on_image)
        self.rect.connect(self.on_rect)
        self.result.connect(self.on_result)
        self._image = None
        self._rects = {}
        self._result = ''

    @Slot(str, int, int, QByteArray)
    def on_image(self, mode: str, w: int, h: int, data: QByteArray):
        if w * h > 0:
            del self._image
            img = Image.frombytes(mode, (w, h), data.data())
            self._image = ImageQt(img)
            self.draw()

    @Slot(str, QRect)
    def on_rect(self, key: str, rect: QRect):
        self._rects[key.upper()] = rect
        self.draw()

    @Slot(str)
    def on_result(self, key: str):
        self._result = key.upper()
        self.draw()

    def draw(self) -> None:
        if self._image is not None:
            self.setFixedSize(self._image.width(), self._image.height())
            pmap = QPixmap.fromImage(self._image)
            painter = QPainter(pmap)
            painter.setPen(QPen(QColor('purple'), 5))
            painter.setFont(QFont("Arial", 20))
            for key, rect in self._rects.items():
                if rect.height() * rect.width() > 0:
                    if self._result and self._result != key and key != 'Q':
                        painter.fillRect(rect, QBrush(QColor(10, 10, 10, 200)))
                        painter.drawText(rect, Qt.AlignmentFlag.AlignLeft, key)
                    else:
                        painter.drawRect(rect)
                        painter.drawText(rect, Qt.AlignmentFlag.AlignLeft, key)
            self.setPixmap(pmap)


class Snapper(QObject):
    """Captures screen"""
    snap = Signal()
    rect = Signal(int, int, int, int)

    started = Signal()
    image = Signal(str, int, int, QByteArray)
    same = Signal()

    def __init__(self, x1: int, y1: int, x2: int, y2: int) -> None:
        super().__init__()
        self.snap.connect(self.on_snap)
        self.rect.connect(self.on_rect)
        self._ref = np.zeros((32, 32), np.uint8)
        self._rect = (x1, y1, x2, y2)

    @Slot()
    def on_snap(self):
        print("Snapping...")
        self.started.emit()
        img = ImageGrab.grab(self._rect).convert('RGB')
        ref = np.array(img.convert('L').resize((32, 32), resample=Image.BICUBIC)).astype(np.uint8)
        if np.abs(ref - self._ref).sum() > 0:
            print("Snap OK")
            self.image.emit(img.mode, *img.size, QByteArray(img.tobytes()))
            self._ref = ref
        else:
            print("Snap same")
            self.same.emit()

    @Slot(int, int, int, int)
    def on_rect(self, x1: int, y1: int, x2: int, y2: int):
        self._rect = x1, y1, x2, y2
        self.on_snap()


class ColorFinder(QObject):
    """Locates region of interest and runs OCR on it"""
    image = Signal(str, int, int, QByteArray)
    rect = Signal(str, QRect)
    rgb = Signal(int, int, int)
    cropx = Signal(int)

    txt = Signal(str)
    toosmall = Signal()
    colornotfound = Signal()

    def __init__(self, name: str, rgb: Tuple[int, int, int], cselect: Callable, cropx: int = 0, accuracy=0.05) -> None:
        super().__init__()
        self._name = name
        self._hsv = cv2.cvtColor(np.uint8([[[*rgb]]]), cv2.COLOR_RGB2HSV)[0, 0]
        self._cselect = cselect
        self._cropx = cropx
        self._accuracy = accuracy

        self.image.connect(self.on_image)
        self.rect.connect(self.on_rect)
        self.rgb.connect(self.on_rgb)
        self.cropx.connect(self.on_cropx)

    @Slot(str, QRect)
    def on_rect(self, name: str, rect: QRect):
        pass

    @Slot(int, int, int)
    def on_rgb(self, r: int, g: int, b: int):
        self._hsv = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)[0, 0]

    @Slot(int)
    def on_cropx(self, cropx: int):
        self._cropx = cropx

    @staticmethod
    def get_mask(hsv: Tuple[int, int, int], accuracy: float, hsvframe: np.ndarray) -> np.ndarray:
        h, s, v = hsv

        frame = hsvframe.copy()
        # rotate the frame hue values to have the target at 90
        frame[:, :, 0] = (frame[:, :, 0] - h + 90) % 180

        mask = cv2.inRange(frame,
                           np.array([90 - 180 * accuracy, max(0, s - 256 * accuracy), max(0, v - 256 * accuracy)]),
                           np.array([90 + 180 * accuracy, min(255, s + 256 * accuracy), min(255, v + 256 * accuracy)]))
        return mask

    @Slot(str, int, int, QByteArray)
    def on_image(self, mode: str, w: int, h: int, data: QByteArray):
        print(f"{self._name} Colorfinding ({self._hsv}, {self._cropx})...")
        img = Image.frombytes(mode, (w, h), data.data())

        bgrframe = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # convert to HSV
        hsvframe = cv2.cvtColor(bgrframe, cv2.COLOR_BGR2HSV).copy()

        # get masks for the specified colors
        mask = ColorFinder.get_mask(self._hsv, self._accuracy, hsvframe)

        # Morphological Transform, Dilation
        # for each color and bitwise_and operator
        # between imageFrame and mask determines
        # to detect only that particular color
        kernel = np.ones((15, 15), "uint8")

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Creating contour to track color
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"{self._name} Color not found")
            self.rect.emit(self._name, QRect(0, 0, 0, 0))
            self.txt.emit('')
            self.colornotfound.emit()
            return

        # Select

        if (contour := self._cselect(contours)) is None or cv2.contourArea(contour) == 0:
            print(f"{self._name} No match")
            self.rect.emit(self._name, QRect(0, 0, 0, 0))
            self.txt.emit('')
            return

        x, y, w, h = cv2.boundingRect(contour)
        print(f"{self._name} Rect found {x} {y} {w} {h}")
        self.rect.emit(self._name, QRect(x + self._cropx, y, w - self._cropx, h))

        cropframe = cv2.cvtColor(bgrframe[y:y + h, min(x + self._cropx, x + w - 2):x + w], cv2.COLOR_BGR2GRAY)
        txt = pytesseract.image_to_string(cropframe).replace('\n', ' ').strip()
        print(f"{self._name} OCR OK: '{txt}'")
        self.txt.emit(txt)


class Chatter(QObject):
    """Asks ChatGPT and retreives the response"""

    question = Signal(str)  # Text read
    answer_a = Signal(str)
    answer_b = Signal(str)
    answer_c = Signal(str)
    answer_d = Signal(str)

    prompt = Signal(str)
    same = Signal()
    invalid = Signal()
    result = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.question.connect(self.on_question)
        self.answer_a.connect(self.on_answer_a)
        self.answer_b.connect(self.on_answer_b)
        self.answer_c.connect(self.on_answer_c)
        self.answer_d.connect(self.on_answer_d)
        self._question = None
        self._answer_a = None
        self._answer_b = None
        self._answer_c = None
        self._answer_d = None

        self._ref = ''

        self._client = OpenAI()

    @Slot(str)
    def on_question(self, txt: str):
        self._question = txt
        self.chat()

    @Slot(str)
    def on_answer_a(self, txt: str):
        self._answer_a = txt
        self.chat()

    @Slot(str)
    def on_answer_b(self, txt: str):
        self._answer_b = txt
        self.chat()

    @Slot(str)
    def on_answer_c(self, txt: str):
        self._answer_c = txt
        self.chat()

    @Slot(str)
    def on_answer_d(self, txt: str):
        self._answer_d = txt
        self.chat()

    def chat(self):
        if self._question is None or self._answer_a is None or self._answer_b is None or self._answer_c is None or self._answer_d is None:
            return

        # full house
        if self._question and self._answer_a and self._answer_b and self._answer_c and self._answer_d:
            prompt = f"{self._question} A: {self._answer_a}, B: {self._answer_b}, C: {self._answer_c}, D: {self._answer_d}. Tell me only the letter of the correct answer!"
        elif self._question and self._answer_a in ['True', 'False'] and self._answer_b in ['True',
                                                                                           'False'] and not self._answer_c and not self._answer_d:
            prompt = f"{self._question} Is it true or false? A: {self._answer_a}, B: {self._answer_b}. Tell me only the letter of the correct answer!"
        else:
            print(
                f"ChatGPT invalid: {self._question}, {self._answer_a}, {self._answer_b}, {self._answer_c}, {self._answer_d}")
            self._question = self._answer_a = self._answer_b = self._answer_c = self._answer_d = None
            self.invalid.emit()
            return

        self._question = self._answer_a = self._answer_b = self._answer_c = self._answer_d = None

        if prompt == self._ref:
            print(f"ChatGPT same ({prompt})")
            self.same.emit()
            return

        self._ref = prompt
        self.prompt.emit(prompt)

        print(f"ChatGPT ask... ({prompt})")
        self.result.emit('')

        # ask chatgpt....
        if True:
            response = self._client.responses.create(model="gpt-5-mini", input=prompt)
            if response.output_text in ['A', 'B', 'C', 'D']:
                result = response.output_text
            else:
                print(f"ChatGPT invalid result: {response.output_text}")
                self.invalid.emit()
                return
        else:
            result = 'A'

        print(f"ChatGPT result ({result})")
        self.result.emit(result)


class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Elements
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        layout_control = QGridLayout()
        layout.addLayout(layout_control)

        self.snappicker = SnapPicker("Snap area", 0, 0, 1600, 1400)
        layout_control.addWidget(self.snappicker, 0, 0)

        self.color_q = QColorPicker("Q", 255, 255, 255)
        self.color_q.setFixedWidth(200)
        self.max_q = QSlider(Qt.Orientation.Horizontal)
        self.max_q.setRange(0, 100)
        self.max_q.setValue(30)
        self.max_q.setFixedWidth(100)
        layout_control.addWidget(self.color_q, 1, 0)
        layout_control.addWidget(QLabel("max distance:"), 1, 1)
        layout_control.addWidget(self.max_q, 1, 2)

        self.color_a = QColorPicker('A', 226, 27, 60)
        self.color_a.setFixedWidth(200)
        self.crop_a = QSlider(Qt.Orientation.Horizontal)
        self.crop_a.setRange(0, 200)
        self.crop_a.setValue(75)
        self.crop_a.setFixedWidth(100)
        self.color_b = QColorPicker('B', 19, 104, 206)
        self.color_b.setFixedWidth(200)
        self.crop_b = QSlider(Qt.Orientation.Horizontal)
        self.crop_b.setRange(0, 200)
        self.crop_b.setValue(75)
        self.crop_b.setFixedWidth(100)
        layout_control.addWidget(self.color_a, 2, 0)
        layout_control.addWidget(QLabel("crop:"), 2, 1)
        layout_control.addWidget(self.crop_a, 2, 2)
        layout_control.addWidget(self.color_b, 2, 3)
        layout_control.addWidget(QLabel("crop:"), 2, 4)
        layout_control.addWidget(self.crop_b, 2, 5)

        self.color_c = QColorPicker('C', 216, 158, 0)
        self.color_c.setFixedWidth(200)
        self.crop_c = QSlider(Qt.Orientation.Horizontal)
        self.crop_c.setRange(0, 200)
        self.crop_c.setValue(75)
        self.crop_c.setFixedWidth(100)
        self.color_d = QColorPicker('D', 38, 137, 12)
        self.color_d.setFixedWidth(200)
        self.crop_d = QSlider(Qt.Orientation.Horizontal)
        self.crop_d.setRange(0, 200)
        self.crop_d.setValue(75)
        self.crop_d.setFixedWidth(100)
        layout_control.addWidget(self.color_c, 3, 0)
        layout_control.addWidget(QLabel("crop:"), 3, 1)
        layout_control.addWidget(self.crop_c, 3, 2)
        layout_control.addWidget(self.color_d, 3, 3)
        layout_control.addWidget(QLabel("crop:"), 3, 4)
        layout_control.addWidget(self.crop_d, 3, 5)

        self.snap = QSnapLabel()
        self.snap.setStyleSheet("border: 1px solid black")
        layout.addWidget(self.snap)

        layout_result = QGridLayout()
        layout.addLayout(layout_result)

        self.prompt = QLabel("Prompt: ---")
        self.result = QLabel("Result: ---")
        self.stop = QPushButton("Stop")
        layout_result.addWidget(self.prompt, 0, 0)
        layout_result.addWidget(self.result, 1, 0)
        layout_result.addWidget(self.stop, 2, 0)

        # Workers
        self.snapper = Snapper(*self.snappicker.getrect)
        self.colorfinder_q = ColorFinder('Q', self.color_q.getrgb(),
                                         lambda contours: max([cc for cc in contours if cv2.boundingRect(cc)[1] < 50],
                                                              key=lambda c: cv2.contourArea(c), default=None), 0)
        self.colorfinder_a = ColorFinder('A', self.color_a.getrgb(),
                                         lambda contours: max(contours, key=lambda c: cv2.contourArea(c)),
                                         self.crop_a.value())
        self.colorfinder_b = ColorFinder('B', self.color_b.getrgb(),
                                         lambda contours: max(contours, key=lambda c: cv2.contourArea(c)),
                                         self.crop_b.value())
        self.colorfinder_c = ColorFinder('C', self.color_c.getrgb(),
                                         lambda contours: max(contours, key=lambda c: cv2.contourArea(c)),
                                         self.crop_c.value())
        self.colorfinder_d = ColorFinder('D', self.color_d.getrgb(),
                                         lambda contours: max(contours, key=lambda c: cv2.contourArea(c)),
                                         self.crop_d.value())
        self.chatter = Chatter()

        # Connect signals
        self.snappicker.rect.connect(self.snapper.rect)
        # self.stop.clicked.connect(self.snapper.snap)
        self.snapper.same.connect(self.snapper.snap)  # loop
        self.chatter.result.connect(self.snapper.snap)  # loop
        self.chatter.same.connect(self.snapper.snap)  # loop
        self.chatter.invalid.connect(self.snapper.snap)  # loop

        self.snapper.image.connect(self.snap.image)
        self.colorfinder_q.rect.connect(self.snap.rect)
        self.colorfinder_a.rect.connect(self.snap.rect)
        self.colorfinder_b.rect.connect(self.snap.rect)
        self.colorfinder_c.rect.connect(self.snap.rect)
        self.colorfinder_d.rect.connect(self.snap.rect)
        self.chatter.result.connect(self.snap.result)

        self.snapper.image.connect(self.colorfinder_q.image)
        self.color_q.rgb.connect(self.colorfinder_q.rgb)
        self.snapper.image.connect(self.colorfinder_a.image)
        self.color_a.rgb.connect(self.colorfinder_a.rgb)
        self.snapper.image.connect(self.colorfinder_b.image)
        self.color_b.rgb.connect(self.colorfinder_b.rgb)
        self.snapper.image.connect(self.colorfinder_c.image)
        self.color_c.rgb.connect(self.colorfinder_c.rgb)
        self.snapper.image.connect(self.colorfinder_d.image)
        self.color_d.rgb.connect(self.colorfinder_d.rgb)

        self.colorfinder_q.txt.connect(self.chatter.question)
        self.colorfinder_a.txt.connect(self.chatter.answer_a)
        self.colorfinder_b.txt.connect(self.chatter.answer_b)
        self.colorfinder_c.txt.connect(self.chatter.answer_c)
        self.colorfinder_d.txt.connect(self.chatter.answer_d)

        self.chatter.prompt.connect(self.prompt.setText)
        self.chatter.result.connect(self.result.setText)

        # Threads
        self.snapper_thread = QThread()
        self.snapper.moveToThread(self.snapper_thread)
        self.snapper_thread.start()

        self.colorfinder_q_thread = QThread()
        self.colorfinder_a_thread = QThread()
        self.colorfinder_b_thread = QThread()
        self.colorfinder_c_thread = QThread()
        self.colorfinder_d_thread = QThread()
        self.colorfinder_q.moveToThread(self.colorfinder_q_thread)
        self.colorfinder_a.moveToThread(self.colorfinder_a_thread)
        self.colorfinder_b.moveToThread(self.colorfinder_b_thread)
        self.colorfinder_c.moveToThread(self.colorfinder_c_thread)
        self.colorfinder_d.moveToThread(self.colorfinder_d_thread)
        self.colorfinder_q_thread.start()
        self.colorfinder_a_thread.start()
        self.colorfinder_b_thread.start()
        self.colorfinder_c_thread.start()
        self.colorfinder_d_thread.start()

        self.chatter_thread = QThread()
        self.chatter.moveToThread(self.chatter_thread)
        self.chatter_thread.start()


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())
