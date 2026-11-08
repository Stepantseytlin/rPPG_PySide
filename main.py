import multiprocessing as mp
import sys

from PySide6.QtWidgets import QApplication

from domain.config import Config
from ui.ui import Window


if __name__ == "__main__":
    mp.freeze_support()
    appConfig = Config(
        cad=30 - 5,
        fMa=240,
        fMi=40+0*26,
        segLength=1024,
        straplen=20,
        source_type="txt",

    )

    app = QApplication(sys.argv)
    win = Window(config=appConfig)
    win.show()
    sys.exit(app.exec())
