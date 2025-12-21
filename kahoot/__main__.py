import json
import random
from typing import Tuple, Callable

import cv2
import numpy as np
import pytesseract
from PIL import ImageGrab, Image, ImageChops
from PySide6 import QtWidgets
from PySide6.QtCore import Signal, Slot, QRect, QObject, QThread
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QPushButton, QLabel, QGridLayout, QComboBox
from openai import OpenAI

from kahoot import QColorFinder, QSnapper


class Snapper(QObject):
    """Captures screen"""
    # in - set snap rectangle
    rect = Signal(int, int, int, int)

    # in - take a snapshot
    snap = Signal()

    # out - snapped image
    image = Signal(Image.Image)

    def __init__(self, x1: int, y1: int, x2: int, y2: int) -> None:
        super().__init__()

        self._rect = (x1, y1, x2, y2)

        # store reference image for comparison
        self._ref_rgba = None

        self.rect.connect(self.on_rect)
        self.snap.connect(self.on_snap)

    @Slot(int, int, int, int)
    def on_rect(self, x1: int, y1: int, x2: int, y2: int):
        self._rect = x1, y1, x2, y2
        self.on_snap()

    @Slot()
    def on_snap(self):
        if self._ref_rgba is None:
            with ImageGrab.grab(bbox=self._rect) as img_rgba:
                self.image.emit(img_rgba)
        else:
            while True:
                with ImageGrab.grab(bbox=self._rect) as img_rgba:
                    with ImageChops.difference(img_rgba, self._ref_rgba) as diff_rgba:
                        with diff_rgba.convert("RGB") as diff_rgb:
                            if diff_rgb.getbbox():
                                self.image.emit(img_rgba)
                                self._ref_rgba.close()
                                self._ref_rgba = img_rgba.copy()
                                break


class ColorFinder(QObject):
    """Locates region of interest and runs OCR on it"""
    # in - image to analyse
    image = Signal(Image.Image)

    # in - color to search
    rgb = Signal(int, int, int)
    acc = Signal(int, int, int)
    kernel = Signal(int)
    crop = Signal(int, int)

    # in - language
    lang = Signal(str)

    # out - area to mark
    mark = Signal(str, QRect)
    mark_too_small = Signal()
    mark_color_not_found = Signal()

    # out - recognized text
    txt = Signal(str)

    def __init__(self, name: str, rgb: Tuple[int, int, int], acc: Tuple[int, int, int], kernel: int,
                 crop: Tuple[int, int], lang: str, cselect: Callable) -> None:
        super().__init__()
        self._name = name
        self._hsv = cv2.cvtColor(np.uint8([[[*rgb]]]), cv2.COLOR_RGB2HSV)[0, 0].astype(int)
        self._acc = acc
        self._kernel = kernel
        self._crop = crop

        self._lang = lang

        self._cselect = cselect

        self.image.connect(self.on_image)
        self.rgb.connect(self.on_rgb)
        self.acc.connect(self.on_acc)
        self.kernel.connect(self.on_kernel)
        self.lang.connect(self.on_lang)
        self.crop.connect(self.on_crop)

    @staticmethod
    def get_mask(hsv: Tuple[int, int, int], acc: Tuple[int, int, int],
                 hsvframe: np.ndarray) -> np.ndarray:
        h, s, v = hsv
        a_h, a_s, a_v = acc

        # rotate the frame hue values to have the target at 90 degree
        hsvframe[:, :, 0] = (hsvframe[:, :, 0] - h + 90) % 180

        mask = cv2.inRange(hsvframe,
                           np.array([90 - a_h, max(0, s - a_s), max(0, v - a_v)]),
                           np.array([90 + a_h, min(255, s + a_s), min(255, v + a_v)]))

        return mask

    @Slot(Image.Image)
    def on_image(self, img: Image.Image) -> None:

        bgrframe = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)

        # convert to HSV
        hsvframe = cv2.cvtColor(bgrframe, cv2.COLOR_BGR2HSV)

        # get masks for the specified colors
        mask = ColorFinder.get_mask(self._hsv, self._acc, hsvframe)

        # Morphological Transform, Dilation
        # for each color and bitwise_and operator
        # between imageFrame and mask determines
        # to detect only that particular color
        kernel = np.ones((self._kernel, self._kernel), "uint8")
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Creating contour to track color
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"{self._name} Color not found ({self._hsv})")
            self.mark.emit(self._name, QRect(0, 0, 0, 0))  # delete existing mark
            self.txt.emit('')
            self.mark_color_not_found.emit()
            return

        # Select
        if (contour := self._cselect(contours)) is None or cv2.contourArea(contour) == 0:
            print(f"{self._name} No match")
            self.mark.emit(self._name, QRect(0, 0, 0, 0))  # delete existing mark
            self.txt.emit('')
            return

        x, y, w, h = cv2.boundingRect(contour)
        w = max(2, w - self._crop[0] - self._crop[1])
        x = min(x + self._crop[0], x + w)
        self.mark.emit(self._name, QRect(x, y, w, h))

        cropframe = cv2.cvtColor(bgrframe[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        txt = pytesseract.image_to_string(cropframe, lang=self._lang).replace('\n', ' ').strip()
        self.txt.emit(txt)

    @Slot(int, int, int)
    def on_rgb(self, r: int, g: int, b: int):
        self._hsv = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)[0, 0]

    @Slot(int, int, int)
    def on_acc(self, h: int, s: int, v: int):
        self._acc = h, s, v

    @Slot(int)
    def on_kernel(self, kernel: int):
        self._kernel = kernel

    @Slot(int, int)
    def on_crop(self, cropx: int, cropy: int):
        self._crop = cropx, cropy

    @Slot(str)
    def on_lang(self, lang: str):
        self._lang = lang


class Chatter(QObject):
    """Asks ChatGPT and retreives the response"""

    # in - prompt parts
    question = Signal(str)
    answer_a = Signal(str)
    answer_b = Signal(str)
    answer_c = Signal(str)
    answer_d = Signal(str)

    # out - prompt
    prompt = Signal(str)
    prompt_same = Signal()
    prompt_invalid = Signal()

    # out - result
    result = Signal(str)

    dict_inst_q = {
        "eng": "Tell me only the letter of the correct answer!",
        "hun": "Csak a helyes válasz betűjét mondjad meg!"
    }
    dict_inst_b = {
        "eng": "Is it true or false?",
        "hun": "Igaz vagy hamis?"
    }
    dict_bool = {
        "eng": ["True", "False"],
        "hun": ["Igaz", "Hamis"],
    }

    def __init__(self, lang: str) -> None:
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
        self._lang = lang

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
            prompt = f"{self._question} A: {self._answer_a}, B: {self._answer_b}, C: {self._answer_c}, D: {self._answer_d}. {self.dict_inst_q[self._lang]}"
        elif self._question and self._answer_a and self._answer_b and self._answer_c and not self._answer_d:
            prompt = f"{self._question} A: {self._answer_a}, B: {self._answer_b}, C: {self._answer_c}. {self.dict_inst_q[self._lang]}"
        elif self._question and self._answer_a and self._answer_b and not self._answer_c and not self._answer_d:
            prompt = f"{self._question} A: {self._answer_a}, B: {self._answer_b}. {self.dict_inst_q[self._lang]}"
        #        elif self._question and self._answer_a in self.dict_bool[self._lang] and self._answer_b in self.dict_bool[self._lang] and not self._answer_c and not self._answer_d:
        #            prompt = f"{self._question} {self.dict_inst_b[self._lang]} A: {self._answer_a}, B: {self._answer_b}. {self.dict_inst_q[self._lang]}"
        else:
            print(
                f"ChatGPT invalid: {self._question if self._question else '-'}, "
                f"{self._answer_a if self._answer_a else '-'}, "
                f"{self._answer_b if self._answer_b else '-'}, "
                f"{self._answer_c if self._answer_c else '-'}, "
                f"{self._answer_d if self._answer_d else '-'}")
            self._question = self._answer_a = self._answer_b = self._answer_c = self._answer_d = None
            self.prompt_invalid.emit()
            return

        self._question = self._answer_a = self._answer_b = self._answer_c = self._answer_d = None

        if prompt == self._ref:
            self.prompt_same.emit()
            return

        self._ref = prompt
        self.prompt.emit(prompt)

        self.result.emit('')

        # ask chatgpt....
        if True:
            response = self._client.responses.create(model="gpt-5-mini", input=prompt)
            if response.output_text in ['A', 'B', 'C', 'D']:
                result = response.output_text
            else:
                print(f"ChatGPT invalid result: {response.output_text}")
                self.prompt_invalid.emit()
                return
        else:
            result = random.choice(['A', 'B', 'C', 'D'])

        self.result.emit(result)


class QPrompt(QLabel):
    pass


class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        with open("kahoot.json", "rt") as f:
            data = json.load(f)

            # Threads
            self.snapper_thread = QThread()
            self.colorfinder_q_thread = QThread()
            self.colorfinder_a_thread = QThread()
            self.colorfinder_b_thread = QThread()
            self.colorfinder_c_thread = QThread()
            self.colorfinder_d_thread = QThread()
            self.chatter_thread = QThread()

            # Elements
            layout = QtWidgets.QVBoxLayout()
            self.setLayout(layout)

            self.snap_s = QSnapper.unserialize(data['snapper'])
            layout.addWidget(self.snap_s)

            layout_control = QGridLayout()
            layout.addLayout(layout_control)

            self.q_cf = QColorFinder.unserialize('Q', data['q_cf'])
            self.lang_cb = QComboBox()
            self.lang_cb.addItems(['eng', 'hun'])
            self.lang_cb.setCurrentText(data['lang'])
            layout_control.addWidget(self.q_cf, 0, 0)
            layout_control.addWidget(self.lang_cb, 0, 1)

            # self.max_q = QSlider(Qt.Orientation.Horizontal)
            # self.max_q.setRange(0, 100)
            # self.max_q.setValue(30)
            # self.max_q.setFixedWidth(100)
            # layout_control.addWidget(QLabel("max distance:"), 0, 1)
            # layout_control.addWidget(self.max_q, 0, 1)

            self.a_cf = QColorFinder.unserialize('A', data['a_cf'])
            self.b_cf = QColorFinder.unserialize('B', data['b_cf'])
            layout_control.addWidget(self.a_cf, 1, 0)
            layout_control.addWidget(self.b_cf, 1, 1)

            self.c_cf = QColorFinder.unserialize('C', data['c_cf'])
            self.d_cf = QColorFinder.unserialize('D', data['d_cf'])
            layout_control.addWidget(self.c_cf, 2, 0)
            layout_control.addWidget(self.d_cf, 2, 1)

            self.q_l = QLabel()
            layout_control.addWidget(self.q_l, 3, 0)
            self.a_l = QLabel()
            layout_control.addWidget(self.a_l, 4, 0)
            self.b_l = QLabel()
            layout_control.addWidget(self.b_l, 4, 1)
            self.c_l = QLabel()
            layout_control.addWidget(self.c_l, 5, 0)
            self.d_l = QLabel()
            layout_control.addWidget(self.d_l, 5, 1)

            self.prompt_l = QLabel()
            self.result_l = QLabel()
            self.start_b = QPushButton("Start")
            layout.addWidget(self.prompt_l)
            layout.addWidget(self.result_l)
            layout.addWidget(self.start_b)

            # Workers
            self.snapper = Snapper(*self.snap_s.getrect)
            self.colorfinder_q = ColorFinder('Q', self.q_cf.getrgb(), self.q_cf.getacc(), self.q_cf.getkernel(),
                                             self.q_cf.getcrop(),
                                             self.lang_cb.currentText(),
                                             lambda contours: max(
                                                 [cc for cc in contours if cv2.boundingRect(cc)[1] < 50],
                                                 key=lambda c: cv2.contourArea(c), default=None))
            self.colorfinder_a = ColorFinder('A', self.a_cf.getrgb(), self.a_cf.getacc(), self.a_cf.getkernel(),
                                             self.a_cf.getcrop(),
                                             self.lang_cb.currentText(),
                                             lambda contours: max(contours, key=lambda c: cv2.contourArea(c)))
            self.colorfinder_b = ColorFinder('B', self.b_cf.getrgb(), self.b_cf.getacc(), self.b_cf.getkernel(),
                                             self.b_cf.getcrop(),
                                             self.lang_cb.currentText(),
                                             lambda contours: max(contours, key=lambda c: cv2.contourArea(c)))
            self.colorfinder_c = ColorFinder('C', self.c_cf.getrgb(), self.c_cf.getacc(), self.c_cf.getkernel(),
                                             self.c_cf.getcrop(),
                                             self.lang_cb.currentText(),
                                             lambda contours: max(contours, key=lambda c: cv2.contourArea(c)))
            self.colorfinder_d = ColorFinder('D', self.d_cf.getrgb(), self.d_cf.getacc(), self.d_cf.getkernel(),
                                             self.d_cf.getcrop(),
                                             self.lang_cb.currentText(),
                                             lambda contours: max(contours, key=lambda c: cv2.contourArea(c)))
            self.chatter = Chatter(self.lang_cb.currentText())

            # Connect signals
            self.start_b.clicked.connect(self.snapper.snap)
            self.snap_s.rect.connect(self.snapper.rect)
            self.chatter.prompt_same.connect(self.snapper.snap)  # loop
            self.chatter.prompt_invalid.connect(self.snapper.snap)  # loop
            self.chatter.result.connect(self.snapper.snap)  # loop

            self.snapper.image.connect(self.snap_s.image)
            self.colorfinder_q.mark.connect(self.snap_s.mark)
            self.colorfinder_a.mark.connect(self.snap_s.mark)
            self.colorfinder_b.mark.connect(self.snap_s.mark)
            self.colorfinder_c.mark.connect(self.snap_s.mark)
            self.colorfinder_d.mark.connect(self.snap_s.mark)
            self.chatter.prompt.connect(self.snap_s.result_clear)
            self.chatter.result.connect(self.snap_s.result)

            self.snapper.image.connect(self.colorfinder_q.image)
            self.q_cf.rgb.connect(self.colorfinder_q.rgb)
            self.q_cf.acc.connect(self.colorfinder_q.acc)
            self.q_cf.kernel.connect(self.colorfinder_q.kernel)
            self.q_cf.crop.connect(self.colorfinder_q.crop)
            self.lang_cb.currentTextChanged.connect(self.colorfinder_q.lang)
            self.snapper.image.connect(self.colorfinder_a.image)
            self.a_cf.rgb.connect(self.colorfinder_a.rgb)
            self.a_cf.acc.connect(self.colorfinder_a.acc)
            self.a_cf.kernel.connect(self.colorfinder_a.kernel)
            self.a_cf.crop.connect(self.colorfinder_a.crop)
            self.lang_cb.currentTextChanged.connect(self.colorfinder_a.lang)
            self.snapper.image.connect(self.colorfinder_b.image)
            self.b_cf.rgb.connect(self.colorfinder_b.rgb)
            self.b_cf.acc.connect(self.colorfinder_b.acc)
            self.b_cf.kernel.connect(self.colorfinder_b.kernel)
            self.b_cf.crop.connect(self.colorfinder_b.crop)
            self.lang_cb.currentTextChanged.connect(self.colorfinder_b.lang)
            self.snapper.image.connect(self.colorfinder_c.image)
            self.c_cf.rgb.connect(self.colorfinder_c.rgb)
            self.c_cf.acc.connect(self.colorfinder_c.acc)
            self.c_cf.kernel.connect(self.colorfinder_c.kernel)
            self.c_cf.crop.connect(self.colorfinder_c.crop)
            self.lang_cb.currentTextChanged.connect(self.colorfinder_c.lang)
            self.snapper.image.connect(self.colorfinder_d.image)
            self.d_cf.rgb.connect(self.colorfinder_d.rgb)
            self.d_cf.acc.connect(self.colorfinder_d.acc)
            self.d_cf.kernel.connect(self.colorfinder_d.kernel)
            self.d_cf.crop.connect(self.colorfinder_d.crop)
            self.lang_cb.currentTextChanged.connect(self.colorfinder_d.lang)
            #
            self.colorfinder_q.txt.connect(self.chatter.question)
            self.colorfinder_a.txt.connect(self.chatter.answer_a)
            self.colorfinder_b.txt.connect(self.chatter.answer_b)
            self.colorfinder_c.txt.connect(self.chatter.answer_c)
            self.colorfinder_d.txt.connect(self.chatter.answer_d)
            #
            self.colorfinder_q.txt.connect(self.q_l.setText)
            self.colorfinder_q.mark_color_not_found.connect(self.q_l.clear)
            self.colorfinder_q.mark_too_small.connect(self.q_l.clear)
            self.colorfinder_a.txt.connect(self.a_l.setText)
            self.colorfinder_a.mark_color_not_found.connect(self.a_l.clear)
            self.colorfinder_a.mark_too_small.connect(self.a_l.clear)
            self.colorfinder_b.txt.connect(self.b_l.setText)
            self.colorfinder_b.mark_color_not_found.connect(self.b_l.clear)
            self.colorfinder_b.mark_too_small.connect(self.b_l.clear)
            self.colorfinder_c.txt.connect(self.c_l.setText)
            self.colorfinder_c.mark_color_not_found.connect(self.c_l.clear)
            self.colorfinder_c.mark_too_small.connect(self.c_l.clear)
            self.colorfinder_d.txt.connect(self.d_l.setText)
            self.colorfinder_d.mark_color_not_found.connect(self.d_l.clear)
            self.colorfinder_d.mark_too_small.connect(self.d_l.clear)
            #
            self.chatter.prompt.connect(self.prompt_l.setText)
            self.chatter.prompt_invalid.connect(self.prompt_l.clear)
            #
            self.chatter.result.connect(self.result_l.setText)
            self.chatter.prompt_invalid.connect(self.result_l.clear)

            # Move to threads
            self.snapper.moveToThread(self.snapper_thread)
            self.colorfinder_q.moveToThread(self.colorfinder_q_thread)
            self.colorfinder_a.moveToThread(self.colorfinder_a_thread)
            self.colorfinder_b.moveToThread(self.colorfinder_b_thread)
            self.colorfinder_c.moveToThread(self.colorfinder_c_thread)
            self.colorfinder_d.moveToThread(self.colorfinder_d_thread)
            self.chatter.moveToThread(self.chatter_thread)

            # Start threads
            self.snapper_thread.start()
            self.colorfinder_q_thread.start()
            self.colorfinder_a_thread.start()
            self.colorfinder_b_thread.start()
            self.colorfinder_c_thread.start()
            self.colorfinder_d_thread.start()
            self.chatter_thread.start()

    def closeEvent(self, event: QCloseEvent):
        with open("kahoot.json", "wt") as f:
            json.dump({"snapper": self.snap_s.serialize(),
                       "lang": self.lang_cb.currentText(),
                       "q_cf": self.q_cf.serialize(),
                       "a_cf": self.a_cf.serialize(),
                       "b_cf": self.b_cf.serialize(),
                       "c_cf": self.c_cf.serialize(),
                       "d_cf": self.d_cf.serialize()
                       }, f)


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())
