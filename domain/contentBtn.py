from PySide6.QtWidgets import QPushButton
from PySide6.QtCore import Signal
class Status:
        myname = "tree"
        def __init__(self, name = "tree"):
            self.myname=name
class contentBtn2(QPushButton):
        sig = Signal(str)

        def __init__(self, parent=None):
            super().__init__(parent)
            self.st = Status()
            self.clicked.connect(self.emitp)

        def setStatusName(self, name: str):
            self.st = Status(name)

        def emitp(self):
            self.sig.emit(self.st.myname)