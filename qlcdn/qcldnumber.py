import sys
import requests
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from bs4 import BeautifulSoup


class Demo(QWidget):
    def __init__(self):
        super(Demo, self).__init__()
        self.resize(600, 250)
        self.subcribers = None

        self.lcd_1 = QLCDNumber(self)
        self.lcd_1.setDigitCount(10)
        self.url = 'https://www.youtube.com/channel/UCIKWgTXmHu5V_YvEg6OXvqg'

        label = QLabel(self)
        label.setFont(QFont('Arial', 50))
        label.setGeometry(QRect(10, 10, 600, 200))
        label.setText(f"頻道名稱:{self.get_yt_title()}")

        self.v_layout = QVBoxLayout()
        self.v_layout.addWidget(label)
        self.v_layout.addWidget(self.lcd_1)

        self.setLayout(self.v_layout)
        self.timer = QTimer(self)
        self.getsub_count()
        self.timer.timeout.connect(self.getsub_count)
        self.timer.start(10000000)

    def get_yt_title(self):
        page = requests.get(self.url)
        soup = BeautifulSoup(page.text, 'html.parser')
        for meta in soup.findAll("meta"):
            # print(meta)
            name = meta.get('name', '')
            tmp = []
            if name == "title":
                title = meta.get('content', '')
        return title

    def getsub_count(self):

        page = requests.get(self.url)
        soup = BeautifulSoup(page.text, 'html.parser')
        pew = soup.findAll("span",
                           {
                               "class": "yt-subscription-button-subscriber-count"
                                        "-branded-horizontal subscribed yt-uix-tooltip"}
                           )
        did = {"萬": 10000, "千": 1000, "百萬": 1000000}
        subcribers = 0
        for subs in pew:
            text = subs.get_text()
            try:
                value = did[text[-1]]
                subcribers = float(text[:-1]) * value
            except KeyError:
                subcribers = int(text)

        print(subcribers)
        t = "%10d" % subcribers
        self.lcd_1.display(t)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = Demo()
    demo.show()
    sys.exit(app.exec_())
