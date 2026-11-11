import PySide6

from PySide6.QtCore import QSize, Qt, QPoint, Signal, Slot, QRect, QTimer, QThread, QObject
from PySide6 import QtWidgets,QtCore , QtGui
from PySide6.QtGui import QIcon, QPixmap, QImage,QColor,QPainter
from PySide6.QtWidgets import QProgressBar,QSlider,QApplication, QMainWindow, QMessageBox,QPushButton, QBoxLayout,QAbstractButton, QLabel,QVBoxLayout,QHBoxLayout, QWidget, QMenu,QGridLayout, QCheckBox,QLineEdit
import cv2 as cv
from PySide6.QtGui import QMouseEvent
import sys, os
from pathlib import Path
import time
from typing import List, Tuple
from .ui_form import Ui_Widget
import numpy as np
from domain.stat import Stat
from domain.FIFO import FIFO
from domain.IF import IF
from domain.config import Config
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from src.sourceHandler import Handler
import scipy
from src.FIFOWorker import FIFOWorker
from src.CalcRunner import CalculateRunner
from domain.Cap import Cap

from domain.Frame_ import Frame_
class ConfigMenu(QWidget):
    def __init__(self,parent:QWidget|None = None)->None:
        super().__init__(parent, Qt.WindowType.Tool)


        self.setStyleSheet("""
            QWidget#PopUp {
                background: palette(--blueTheme 100%);
                border: 1px solid palette(--blueTheme 100%);
            }
        """)
        self.changingFromInput = 0
        rect = parent.geometry()

        parentGeomX,parentGeomY,parentGeomW,parentGeomH = rect.x(), rect.y(),rect.width(),rect.height()

        self.setGeometry(parentGeomX+parentGeomW//2,parentGeomY+parentGeomH//2,parentGeomW//2,parentGeomH//2)

        #Лэйоут с настройками
        # self.HomeWidget = QWidget(self)
        self.HomeLayout = QGridLayout(self)
        # self.fillerwidget = QWidget(self.HomeWidget)
        # self.fillerwidget.setMinimumSize(QSize(400,100))
        self.setStyleSheet("""QWidget {
                                background-color: palette(in lch, --yellowTheme 100%);
                                border: 1px solid palette(mid);
                                color : 000000;
                            }
                        """)

        # self.HomeLayout.addWidget(self.fillerwidget, 0,1)
        #self.fillerwidget.show()

        self.SegmentConfigWidget = QWidget(self)
        self.HomeLayout.addWidget(self.SegmentConfigWidget,1,0)
        self.SegmentConfigWidget.setMinimumSize(QSize(300,200))
        self.SegmentConfigWidget.setMaximumSize(QSize(420,240))
        self.SegmentConfigWidget.setSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding,QtWidgets.QSizePolicy.Policy.MinimumExpanding)
        self.SegmentConfigWidget.setStyleSheet("""QLabel{
                                                background-color: palette(in lch, --yellowTheme 100%);
                                                border: 0px solid palette(mid);
                                                color : 000000;
                                            }
                                        """)

        self.SegmentConfigLayout = QVBoxLayout(self.SegmentConfigWidget)
        self.SegmentConfigLayout.setSpacing(1)
        self.SegmentConfigLayout.setContentsMargins(0,0,0,0)

        """Настраиваем виджет длины сегмента"""
        self.SegLengthSlider = QSlider(orientation = QtCore.Qt.Orientation.Horizontal,tickInterval=1,tickPosition=QSlider.TickPosition.TicksBelow )
        self.SegLengthSlider.setMinimum(128)
        self.SegLengthSlider.setMaximum(2048)

        self.SegLengthSlider.setToolTip("Задать длину сегмента")

        self.SegLenlabel = QWidget(self.SegmentConfigWidget)
        self.SegLenlabel.setMaximumHeight(100)
        self.SegLenlayout = QHBoxLayout(self.SegLenlabel)
        self.SegLenNameLabel = QLabel("Длина Сегмента:")
        self.SegmentConfigLayout.addWidget(self.SegLenlabel,alignment=QtCore.Qt.AlignmentFlag.AlignTop)
        self.SegmentConfigLayout.addWidget(self.SegLengthSlider,alignment=QtCore.Qt.AlignmentFlag.AlignTop)

        self.fpsWidget = QWidget(self.SegmentConfigWidget)
        self.fpsLabel = QLabel("Частота дискретизации: ",)
        self.fpsconfigInput = QLineEdit(f"{30}",parent = self.fpsWidget)
        self.fpsLayout = QHBoxLayout(self.fpsWidget)
        self.fpsLayout.addWidget(self.fpsLabel)
        self.fpsLayout.addWidget(self.fpsconfigInput)
        self.fpsconfigInput.textChanged.connect(self.OnfpsChanged)
        self.SegmentConfigLayout.addWidget(self.fpsWidget)
        
        
        self.SegLenlayout.addWidget(self.SegLenNameLabel)

        self.SegLenInput = QLineEdit("Введите длину сегмента",parent = self.SegLenlabel)
        self.SegLenInput.textChanged.connect(self.SegLenInputChanged)
        self.SegLenInput.setToolTip("По умолчанию, 1024")
        self.SegLenlayout.addWidget(self.SegLenInput)

        self.SegLengthSlider.valueChanged.connect(self.SegLengthChanged)
        self.SegLengthSlider.setValue(1024)

        self.changingFromInput = 0


        """Настраиваем виджет перекрытия"""
        self.StrapLengthSlider = QSlider(orientation = QtCore.Qt.Orientation.Horizontal,tickInterval=1,tickPosition=QSlider.TickPosition.TicksBelow )
        self.StrapLengthSlider.setMinimum(1)
        self.StrapLengthSlider.setMaximum(99)

        self.StrapLenlabel = QWidget(self.SegmentConfigWidget)
        self.StrapLenlayout = QHBoxLayout(self.StrapLenlabel)
        self.StrapLenNameLabel = QLabel("Процент Перекрытия:")
        self.StrapLenlayout.addWidget(self.StrapLenNameLabel)

        self.StrapLenInput = QLineEdit("Введите длину перекрытия",parent = self.StrapLenlabel)
        self.StrapLenInput.textChanged.connect(self.StrapLenInputChanged)
        self.StrapLenInput.setToolTip("По умолчанию, 20")

        self.StrapLenlayout.addWidget(self.StrapLenInput)
        self.StrapLenlayout.addWidget(QLabel("%"))
        self.SegmentConfigLayout.addWidget(self.StrapLenlabel,alignment=QtCore.Qt.AlignmentFlag.AlignTop)
        self.SegmentConfigLayout.addWidget(self.StrapLengthSlider,alignment=QtCore.Qt.AlignmentFlag.AlignTop)

        self.StrapLengthSlider.valueChanged.connect(self.StrapLengthChanged)

        self.StrapLengthSlider.setValue(20)
        """-----------------------------"""
        self.Eiglabel = QWidget(self.SegmentConfigWidget)
        self.Eigs = QHBoxLayout(self.Eiglabel)
        self.feig = QLabel(text="Первая компонента:")
        self.leig = QLabel(text="Последняя компонента:")
        self.EigTripletsF = QLineEdit("0",parent = self.StrapLenlabel,)
        self.EigTripletsL = QLineEdit("2",parent = self.StrapLenlabel,)
        self.EigTripletsF.setMaximumWidth(15)
        self.EigTripletsL.setMaximumWidth(15)
        self.Eigs.addWidget(self.feig)
        self.Eigs.addWidget(self.EigTripletsF)
        self.Eigs.addWidget(self.leig)
        self.Eigs.addWidget(self.EigTripletsL)
        self.SegmentConfigLayout.addWidget(self.Eiglabel)
        
        

        self.HilbertCheckBox = QCheckBox("Использовать Преобразование Гильберта",parent = self.SegmentConfigWidget)
        self.SegmentConfigLayout.addWidget(self.HilbertCheckBox)
        self.setButton = QPushButton("Задать",parent = self.SegmentConfigWidget )
        self.SegmentConfigLayout.addWidget(self.setButton)
        rec = self.rect().height(),self.rect().width()
        self.setButton.setGeometry(0,0,rec[1]//2,rec[0]//2)
        self.setButton.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
    
    
        
    def OnfpsChanged(self,text):
        QMessageBox(text = "Внимание. Изменение ФПС влияет на точность оценки чсс").show()
        pass

    def SegLenInputChanged(self, text):
        if self.changingFromInput:
            self.changingFromInput = 0
            return
        if not text:
            return
        if int(text) < 128:
            return
        newText = int(text)
        self.changingFromInput = 1
        self.SegLengthSlider.setValue(newText)
    def SegLengthChanged(self,value):
        if self.changingFromInput:
            self.changingFromInput = 0
            return
        newValue = value
        self.changingFromInput = 1
        self.SegLenInput.setText(f"{newValue}")
    def StrapLengthChanged(self,value):
            if self.changingFromInput:
                self.changingFromInput = 0
                return
            newValue = value
            self.changingFromInput=1
            self.StrapLenInput.setText(f"{newValue}")
    def StrapLenInputChanged(self,text):
        if not text:
            return
        if self.changingFromInput:
            self.changingFromInput = 0
            return
        if int(text) < 1:
            return
        newText = int(text)
        self.changingFromInput = 1
        self.StrapLengthSlider.setValue(newText)
    
class PopUpConfigMenu(QWidget):
    def __init__(self,parent):
        super().__init__(parent)
        self.popTimer = QTimer(singleShot=True)
        self.btn = QPushButton("Конфигурация",self)
        self.popTimer.setInterval(5000)
        self.popTimer.timeout.connect(self.ontimer)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._color = QColor("green")
        self.setMouseTracking(True)
        
        self.popup_menu = ConfigMenu(self.window())
        self.btn.clicked.connect(self.popup_menu.show)
        # self.setGeometry(0,0,300,50)
        self.setToolTip("Настройки конфигураций")
        # widget = QWidget(self)
        # self.Nametext = QLabel("Конфигурация",self.btn)

        # self.Nametext.setObjectName("NameBtn")
        self.baselayout = QGridLayout(self)
        self.baselayout.setContentsMargins(0, 0, 0, 0)
        self.baselayout.setSpacing(0)
        self.baselayout.addWidget(self.btn)
        
        # self.baselayout.setColumnMinimumWidth(0,100)
        # self.baselayout.addWidget(self.Nametext)
        self.setStyleSheet("""
                    QLabel#NameBtn {
                        background-color: palette(in lch, --yellowTheme 100%);
                        border: 2px solid palette(mid);
                    }
                """)
        self.show()
    
    
    
    def enterEvent(self,event):
        # self.popup_menu.show()
        pass

    def ontimer(self):
        
        cursor_position = self.cursor().pos()

        over_panel = self.rect().contains(
            self.mapFromGlobal(cursor_position)
        )

        over_popup = self.popup_menu.rect().contains(
            self.popup_menu.mapFromGlobal(cursor_position)
        )

        if not over_panel and not over_popup:
            self.popup_menu.hide()
        return

    def leaveEvent(self, event):
        self.popTimer.start()
        # hide()
        return 0#super().leaveEvent(event)



class Window(QWidget,Ui_Widget, QObject):
    HandlerStart : Signal = Signal()
    HandlerConfig : Signal = Signal(object, object)

    def __init__(self, config : Config):
        super(Window, self).__init__()

        self.PGMScatterplots=[]

        self.config = config
        self.setupUi(self)
        self._last_frame_pixmap = None
        self._configure_interface()

        self.project_root = Path(__file__).resolve().parents[1]
        self.default_source_path = self.project_root / "sources" / "txt" / "RGB" / "P1H1Labnet.txt"
        self.stat = Stat()
        self.faces = []
        self.IF = IF()
        self.Frame = Frame_()
        self.cap = Cap()
        self.FRAMES = []
        self.vidpath = str(self.default_source_path)
        self._last_roi = None

        self.fifo = FIFO(seglength=config.segLength, strapLen=config.strapLen, IF=self.IF, stat=self.stat, config=self.config)
        self.fifoworker = FIFOWorker(self.fifo)

      
        self.plot = pg.PlotWidget()
        self.plotCR = pg.PlotWidget()
        self.plotPGM = pg.PlotWidget(self)


        self.frq = QVBoxLayout(self.Frequency)
        self.frq.setContentsMargins(12, 10, 12, 12)
        self.frq.setSpacing(8)
        self.frq.addWidget(self._plot_title("Периодограмма Томсона"))
        # self.frq.addWidget(self.canvas, 1)

        self.PlotLabel = QLabel("График: ", self)
        self.PlotLabel.setObjectName("PlotTitle")
        self.PlotSource = QtWidgets.QComboBox(self)
        self.PlotSource.setObjectName("SourceCombo")
        self.PlotSource.addItem("Пульсовая волна", "Пульсовая волна")
        self.PlotSource.addItem("ДФПГ","ДФПГ")
        self.PlotSource.addItem("АПГ","АПГ")
        self.PlotSource.addItem("Периодограмма Томсона","ПГМ")
        self.PlotSource.addItem("Собственные значения", "eigvls")
        self.PlotSource.setCurrentIndex(0)
        self.PlotSource.currentIndexChanged.connect(self.on_plot_changed)


        self.cr = QVBoxLayout(self.PulseWave)

        self.cr.addWidget(self.PlotLabel)
        self.cr.addWidget(self.PlotSource,0, Qt.AlignmentFlag.AlignLeft)#(self._plot_title("Пульсовая волна"))жж
        self.cr.addWidget(self.plotCR, 1)


        self.cr.setContentsMargins(12, 10, 12, 12)
        self.cr.setSpacing(8)

        # self.horizontalLayoutfloor2.addWidget(self.PlotLabel, 0, Qt.AlignmentFlag.AlignHCenter)
        # self.horizontalLayoutfloor2.addWidget(self.PlotSource, 0, Qt.AlignmentFlag.AlignHCenter)
        self.horizontalLayoutfloor2.setContentsMargins(50,50,50,50)
        self.horizontalLayoutfloor2.setStretch(1,0)

        self.med_fr = QVBoxLayout(self.Fr)
        self.med_fr.setContentsMargins(12, 10, 12, 12)
        self.med_fr.setSpacing(8)
        self.med_fr.addWidget(self._plot_title("ЧСС"))
        self.med_fr.addWidget(self.plot, 1)
        self._configure_plot_widget(self.plotCR, "Время", "CR")
        self._configure_plot_widget(self.plot, "Время", "ЧСС, уд/мин")
        self.if_curve = self.plot.plot([], [], pen=pg.mkPen("#2563eb", width=2))
        self.confidence_if=[]
        self.confidence_if.append(self.plot.plot([], [], pen=pg.mkPen({'color':"#2563eb", 'width':1,'style':PySide6.QtCore.Qt.PenStyle.DashLine})))
        self.confidence_if.append(self.plot.plot([], [], pen=pg.mkPen({'color':"#2563eb", 'width':1,'style':PySide6.QtCore.Qt.PenStyle.DashLine})))
        self.EigScatters = self.plotCR.scatterPlot( [], [], size = 1.5, pen = QtGui.QPen( "blue"))

        self.cr_curve : pg.PlotDataItem  = self.plotCR.plot([], [], pen=pg.mkPen("#0f766e", width=2))

        # self.sfc = gl.GLSurfacePlotItem(z=100 * np.random.random((1, 1)))
        # self.canvas.addItem(self.sfc)
        # self.canvas.setBackgroundColor((255, 255, 255))
        # self.canvas.setCameraPosition(distance=600, elevation=22, azimuth=225-180)

        # self.axis = gl.GLAxisItem()
        # self.axis.setSize(x=399, y=513//4, z=100)
        # self.canvas.addItem(self.axis)
        # font = QtGui.QFont("Arial", 14)
        # self.canvas.addItem(gl.GLTextItem(pos=(150.2, 0, 0), text="Сегмент", color=(0, 0, 0, 255), font=font))
        # self.canvas.addItem(gl.GLTextItem(pos=(0, 150.2, 0), text="Гц", color=(0, 0, 0, 255), font=font))
        # self.canvas.addItem(gl.GLTextItem(pos=(0, 0, 55.2), text="P", color=(0, 0, 0, 255), font=font))

        # ticks = np.arange(0, (self.config.segLength//2+1)//4+1, self.config.segLength/self.config.cad)
        # for i, t in enumerate(ticks):
        #     line = gl.GLLinePlotItem(pos=np.array([[0, t, 0], [0.1, t, 0]]), color=(0, 0, 0, 255), width=1, antialias=True)
        #     self.canvas.addItem(line)
        #     self.canvas.addItem(gl.GLTextItem(pos=(-0.3, t, 0), text=str(t * self.config.cad/self.config.segLength), color=(0, 0, 0, 255)))

        self.TomsGMS = QVBoxLayout()

        self.frq.addLayout(self.TomsGMS)
        self.TomsGMS.addWidget(self.plotPGM)
        self.PGMScatter = self.plotPGM.scatterPlot([1,1,5,0],[4,5,7,5], pen=pg.mkPen("#d7eb25", alphaF=0.2),size = 0)
        self.PGMLine = self.plotPGM.plot([],[],pen=pg.mkPen("#eb2825", width=7, alphaF=1))

        self._configure_plot_widget(self.plotPGM, "Частота", "Мощность")



        self.timer = QTimer(self)
        self.timer.setInterval(60*5)
        self.timer.timeout.connect(self.on_timer)
        self.timer.start()

      
        self.FIFOth = QThread()
     

        self.fifoworker.moveToThread(self.FIFOth)
        self.fifo.threadmngr.moveToThread(self.FIFOth)
        self.FIFOth.start()

        self.ROIth = QThread()
        default_source = str(self.default_source_path) if self.default_source_path.exists() else None
        self.srcHandler = Handler(self.config, self.VSlot, self.fifoworker.PSlot, filepath=default_source, stat=self.stat, IS=self.fifoworker.InstantSlot)
        self.srcHandler.moveToThread(self.ROIth)
        self.HandlerConfig.connect(self.srcHandler.ChangeConfig)
        self.HandlerStart.connect(self.srcHandler.read)
        self.ROIth.start()
        self.HandlerStart.emit()
        self.srcHandler.batchStateChanged.connect(
            self.batchProgress.setVisible
        )
        self.toolButton.sig.connect(self.choosefile)
        self.toolButton_4.clicked.connect(self.PausePlay)

        self.timerPlot = QTimer(self)
        self.timerPlot.setInterval(300)#(int((self.config.strapLen *1000)//self.config.cad ))
        self.timerPlot.timeout.connect(self.displayPlot)
        self.timerPlot.start()

        self.toolButton_5.clicked.connect(self.reset)
        self.radioButton.toggled.connect(self._toggle_camera)
        self.radioButtonINST.toggled.connect(self._toggleINST)




        # self.configMenu = QMenu("Конфигурация", self.plots)
        # self.configMenu.popup(QPoint(300,200),QtGui.QAction(text = "bababa",parent = self.plots,))

    def _toggleINST(self, checked):
        self.config.instant = 0 if checked else 1
        if hasattr(self, "srcHandler"):
            self._request_handler_config()
        mode = "\u0420\u0435\u0436\u0438\u043c \u0440\u0435\u0430\u043b\u044c\u043d\u043e\u0433\u043e \u0432\u0440\u0435\u043c\u0435\u043d\u0438" if checked else "\u041f\u0430\u043a\u0435\u0442\u043d\u044b\u0439 \u0440\u0430\u0441\u0447\u0435\u0442 TXT"
        self._set_status(mode)

    @Slot(np.ndarray,np.ndarray)
    def VSlot(self, frame, roi):
        self.Frame.value = frame
        self._last_roi = roi
        if frame is not None:
            if not self.stat.Pause:
                self.displayFace()
            else:
                ##
                pass

    @Slot(np.ndarray)
    def PSlot(self, roi):
        self.fifo.insertFrame(roi)

    def _configure_interface(self):
        self.setWindowTitle("rPPG Monitor")
        self.resize(1280, 760)
        self.setMinimumSize(1040, 640)

        self.gridLayout.setContentsMargins(14, 14, 14, 14)
        self.gridLayout.setHorizontalSpacing(14)
        self.gridLayout.setColumnStretch(0, 0)
        self.gridLayout.setColumnStretch(1, 1)
        self.PulseWave_2.setSpacing(10)
        self.verticalLayout_7.setSpacing(10)
        self.horizontalLayout_2.setSpacing(8)

        self.toolButton.setObjectName("PrimaryButton")
        self.toolButton_4.setObjectName("RunButton")
        self.toolButton_5.setObjectName("SecondaryButton")
        self.figvid.setObjectName("VideoPreview")
        self.plots.setObjectName("PlotsArea")
        self.PulseWave.setObjectName("PlotPanel")
        self.Fr.setObjectName("PlotPanel")
        self.Frequency.setObjectName("PlotPanel")

        self.toolButton_2.hide()
        self.toolButton.setStatusName("*.txt")
        self.toolButton.setToolTip("Выбрать файл для обработки")
        self.toolButton_4.setText("Старт")
        self.toolButton_4.setToolTip("Запустить или поставить обработку на паузу")
        self.toolButton_5.setText("Сброс")
        self.toolButton_5.setToolTip("Очистить накопленные данные")
        self.radioButton.setText("Камера")
        self.radioButton.setAutoExclusive(False)
        self.radioButton.setCursor(Qt.CursorShape.PointingHandCursor)
        self.radioButtonINST.setText("Режим Реального Времени")
        self.radioButtonINST.setAutoExclusive(False)
        self.radioButtonINST.setCursor(Qt.CursorShape.PointingHandCursor)
        self.radioButtonINST.setChecked(1)
        self.sourceLabel = QLabel("Источник:", self)
        self.sourceLabel.setObjectName("ControlLabel")
        self.sourceCombo = QtWidgets.QComboBox(self)
        self.sourceCombo.setObjectName("SourceCombo")
        self.sourceCombo.addItem("Видео", "*.mp4")
        self.sourceCombo.addItem("TXT", "*.txt")
        self.sourceCombo.setCurrentIndex(1)
        self.sourceCombo.setToolTip("Тип входных данных")
        self.horizontalLayout_2.insertWidget(0, self.sourceLabel, 0, Qt.AlignmentFlag.AlignTop)
        self.horizontalLayout_2.insertWidget(1, self.sourceCombo, 0, Qt.AlignmentFlag.AlignTop)
        self.sourceCombo.currentIndexChanged.connect(self.on_source_changed)

        self.configpopup = PopUpConfigMenu(self)
        self.configpopup.setObjectName("ConfigBtn")
        self.configpopup.show()
        
        self.horizontalLayout_2.insertWidget(2,self.configpopup)
        self.configpopup.popup_menu.setObjectName("PopUp")
        self.configpopup.popup_menu.setButton.clicked.connect(self.changeConfig)
        


        self.figvid.setText("\u041e\u0442\u043a\u0440\u043e\u0439\u0442\u0435 TXT \u0438\u043b\u0438 \u0432\u0438\u0434\u0435\u043e, \u043b\u0438\u0431\u043e \u0432\u043a\u043b\u044e\u0447\u0438\u0442\u0435 \u043a\u0430\u043c\u0435\u0440\u0443")
        self.figvid.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.figvid.setMinimumSize(QSize(420, 320))
        self.figvid.setMaximumSize(QSize(420, 320))
        # self.figvid.setScaledContents(False)
        self.figvid.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        # self.figvid.installEventFilter(self)
        self.verticalLayout_6.setAlignment(self.figvid, Qt.AlignmentFlag.AlignTop)

        self.RPPGWidget.hide()
        self.PulseWave.setMinimumHeight(260)
        self.Fr.setMinimumHeight(240)
        self.Frequency.setMinimumHeight(320)
        self.plots.setMaximumSize(QSize(16777215, 16777215))
        self.plots.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        self.label.hide()
        self.label_2.hide()
        self.label_3.hide()

        self.metricsLayout = QtWidgets.QHBoxLayout()
        self.metricsLayout.setSpacing(8)
        self.metricSource = self._add_metric(self.metricsLayout, "\u0418\u0441\u0442\u043e\u0447\u043d\u0438\u043a", "TXT")
        self.metricFace = self._add_metric(self.metricsLayout, "ROI", "ожидание")
        self.metricHeartRate = self._add_metric(self.metricsLayout, "ЧСС", "--")
        self.verticalLayout_7.addLayout(self.metricsLayout)

        self.statusLabel = QLabel("Ожидание источника", self)
        self.statusLabel.setObjectName("StatusLabel")
        self.verticalLayout_7.addWidget(self.statusLabel)

        self.batchProgress = QProgressBar(self)
        self.batchProgress.setRange(0, 0)
        self.batchProgress.setTextVisible(False)
        self.batchProgress.hide()
        self.verticalLayout_7.addWidget(
            self.batchProgress
        )
        self.setStyleSheet("""
            QWidget {
                background: #f5f7fb;
                color: #1f2937;
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 13px;
            }
            QLabel#ControlLabel {
                color: #475569;
                padding: 8px 0 0 0;
                background: transparent;
            }
            QPushButton, QComboBox {
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 7px 12px;
            }
            QPushButton:hover, QComboBox:hover {
                border-color: #64748b;
            }
            QPushButton:pressed {
                background: #e2e8f0;
            }
            QPushButton#PrimaryButton {
                background: #2563eb;
                border-color: #2563eb;
                color: #ffffff;
                font-weight: 600;
            }
            QPushButton#RunButton {
                background: #0f766e;
                border-color: #0f766e;
                color: #ffffff;
                font-weight: 600;
            }
            QLabel#VideoPreview {
                background: #111827;
                border: 1px solid #334155;
                border-radius: 8px;
                color: #cbd5e1;
                padding: 12px;
            }
            QWidget#PlotPanel {
                background: #ffffff;
                border: 1px solid #d8dee9;
                border-radius: 8px;
            }
            QScrollArea#PlotsArea {
                background: transparent;
                border: none;
            }
            QLabel#PlotTitle {
                background: transparent;
                color: #0f172a;
                font-size: 14px;
                font-weight: 600;
            }
            QWidget#MetricBox {
                background: #ffffff;
                border: 1px solid #d8dee9;
                border-radius: 8px;
            }
            QLabel#MetricTitle {
                background: transparent;
                color: #64748b;
                font-size: 11px;
            }
            QLabel#MetricValue {
                background: transparent;
                color: #0f172a;
                font-size: 15px;
                font-weight: 600;
            }
            QLabel#StatusLabel {
                background: #eef2ff;
                border: 1px solid #c7d2fe;
                border-radius: 8px;
                color: #3730a3;
                padding: 8px 10px;
            }
            QLabel#PlotLabel {
                background: #ffffff;
                color: #0f172a;
                font-size: 14px;
                font-weight: 600;
                        }

        """)
    def ResizeFIFO(self):
        segLen = self.configpopup.popup_menu.SegLengthSlider.value()
        strapLen = self.configpopup.popup_menu.StrapLengthSlider.value()
        cad = int(self.configpopup.popup_menu.fpsconfigInput.text())
        print(f" Eigentriplets fifo = {self.fifo.config.nEt}")
        self.config.cad = cad
        self.stat.Pause = 1
        self.fifo.resize(segLen,strapLen)
        self.config.strapLen = strapLen
        self.config.segLength = segLen
        self.srcHandler.ChangeConfig(self.config)
    def changeConfig(self):
        previousState = self.stat.Pause
        feig = self.configpopup.popup_menu.EigTripletsF.text()
        leig = self.configpopup.popup_menu.EigTripletsL.text()
        self.config.nEt = list((int(feig),int(leig)))
        print(f" Eigentriplets = {self.config.nEt}")
        self.configpopup.popup_menu.hide()
        self.ResizeFIFO()
        self.config.Hilbert = bool(self.configpopup.popup_menu.HilbertCheckBox.checkState())
        self.stat.Pause = previousState
         
        
    def _plot_title(self, text):
        title = QLabel(text, self)
        title.setObjectName("PlotTitle")
        return title

    def _add_metric(self, target_layout, title, value):
        box = QWidget(self)
        box.setObjectName("MetricBox")
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(10, 7, 10, 7)
        box_layout.setSpacing(2)
        title_label = QLabel(title, box)
        title_label.setObjectName("MetricTitle")
        value_label = QLabel(value, box)
        value_label.setObjectName("MetricValue")
        box_layout.addWidget(title_label)
        box_layout.addWidget(value_label)
        target_layout.addWidget(box)
        return value_label

    def _configure_plot_widget(self, plot_widget, bottom_label, left_label):
        plot_widget.setBackground("#ffffff")
        plot_widget.showGrid(x=True, y=True, alpha=0.25)
        plot_widget.setLabel("bottom", bottom_label)
       
        plot_widget.setLabel("left", left_label)
        plot_widget.getAxis("bottom").setPen(pg.mkPen("#64748b"))
        plot_widget.getAxis("left").setPen(pg.mkPen("#64748b"))
        plot_widget.getAxis("bottom").setTextPen(pg.mkPen("#475569"))
        plot_widget.getAxis("left").setTextPen(pg.mkPen("#475569"))

    def _set_status(self, text):
        if hasattr(self, "statusLabel"):
            self.statusLabel.setText(text)

    def on_plot_changed(self,index=None):
        plotType = self.PlotSource.currentData()
        if plotType=="Пульсовая волна" :
            self.config.plotSource = "PW"
        elif plotType=="ДФПГ":
            self.config.plotSource = "rPPG"
        elif plotType=="АПГ":
                    self.config.plotSource = "deriv"
        elif plotType == "ПГМ":
            self.config.plotSource = "pgm"
        elif plotType == "eigvls":
            self.config.plotSource = "eigvls"
        print(f"changed plot for {plotType}")
    def on_source_changed(self, index=None):
        fmt = self.sourceCombo.currentData()
        self.toolButton.setStatusName(fmt)
        self.config.SourceType = "txt" if fmt == "*.txt" else "video"
        self._set_camera_checked(False)
        if fmt == "*.txt":
            print(f"new source is from .txt {self.config.SourceType, fmt}")
            self._request_handler_config(str(self.default_source_path) if self.default_source_path.exists() else self.vidpath)
            # self.toolButton.setText("Открыть файл")
            self.metricSource.setText("TXT")
            self._set_status("\u0418\u0441\u0442\u043e\u0447\u043d\u0438\u043a: \u0442\u0435\u043a\u0441\u0442\u043e\u0432\u044b\u0439 \u0444\u0430\u0439\u043b")
        else:
            print(f"new source is {self.config.SourceType, fmt,self.config.SourceType == "video",
                            isinstance(self.vidpath, str),
                            Path(self.vidpath).suffix.lower() in {".mp4", ".avi", ".mov"},
                            Path(self.vidpath).exists()}")
            if (
                self.config.SourceType == "video"
                and isinstance(self.vidpath, str)
                and Path(self.vidpath).suffix.lower() in {".mp4", ".avi", ".mov"}
                and Path(self.vidpath).exists()
            ):
                print(f"new source is {self.config.SourceType, fmt}")
                self._request_handler_config(self.vidpath)
            # self.toolButton.setText("Открыть файл")
            self.metricSource.setText("\u0412\u0438\u0434\u0435\u043e")
            self._set_status("\u0418\u0441\u0442\u043e\u0447\u043d\u0438\u043a: \u0432\u0438\u0434\u0435\u043e")

    @Slot(np.ndarray)
    def _show_frame(self, frame):
        if frame is None:
            return False
        frame = cv.Mat(frame)
        if len(frame.shape) == 2:
            rgb = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
        else:
             
            rgb = frame
            # rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        # rgb = np.array(rgb)

        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w,QImage.Format_RGB888)
        # print(f"showing face {rgb.max()}")
        self._last_frame_pixmap = QPixmap.fromImage(img.copy())
        self._update_video_pixmap()
        return True

    def _update_video_pixmap(self):
        if self._last_frame_pixmap is None or self._last_frame_pixmap.isNull():
            return
        target_size = self.figvid.contentsRect().size()
        if not target_size.isValid() or target_size.isEmpty():
            return
        # print("resizing PixMap")
        self.figvid.setPixmap(
            self._last_frame_pixmap.scaled(
                target_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def eventFilter(self, watched, event):
        if watched is self.figvid and event.type() == QtCore.QEvent.Type.Resize:
            self._update_video_pixmap()
        return super().eventFilter(watched, event)

    def _update_heart_rate_metric(self):
        if not hasattr(self, "metricHeartRate") or len(self.IF.values) == 0:
            return
        latest = np.asarray(self.IF.values[-1], dtype=float)[0,0]
         
        value_hz = float(np.nanmean(latest))
        self.metricHeartRate.setText(f"{value_hz * 60:.1f} \u0443\u0434/\u043c\u0438\u043d")

    def _odd_filter_size(self, size, preferred):
        if size <= 1:
            return 1
        value = min(size, preferred)
        if value % 2 == 0:
            value -= 1
        return max(1, value)

    def _periodogram_surface(self):

        if len(self.fifo.pgm) == 0:
            return None

        pgm = np.asarray(self.fifo.pgm, dtype=float)
         
        if pgm.ndim == 4:
            surface = pgm[:, 0, :, 0]
        elif pgm.ndim == 3:
            if pgm.shape[1] == 1:
                surface = pgm[:, 0, :]
            elif pgm.shape[-1] == 1:
                surface = pgm[:, :, 0]
            else:
                surface = pgm[:, :, 0]
        elif pgm.ndim == 2:
            surface = pgm
        elif pgm.ndim == 1:
            surface = pgm.reshape(1, -1)
        else:
            return None

        surface = np.nan_to_num(surface, nan=0.0, posinf=0.0, neginf=0.0)
        width = surface.shape[1]
        return surface[:, :max(1, width // 6 + 1)]

    def _update_periodogram_surface(self):
        # print(f"cond : {not self.stat.Pause and not self.stat.stopPlotting}")
        if not self.stat.Pause or not self.stat.stopPlotting:

            pgm = self._periodogram_surface()
            if pgm is None:
                return

            rows, cols = pgm.shape
            kernel = (
                self._odd_filter_size(rows, 45),
                self._odd_filter_size(cols, 21),
            )
            pgm = scipy.signal.medfilt2d(pgm, kernel)

            cmap = pg.colormap.get("viridis")
            palette = cmap.getLookupTable(0.0, 1.0, 256, alpha=False)
            min_value = float(np.min(pgm))
            max_value = float(np.max(pgm))
            if max_value > min_value:
                normalized = (pgm - min_value) / (max_value - min_value)
            else:
                normalized = np.zeros_like(pgm)
            color_indexes = np.clip(normalized * 255, 0, 255).astype(np.uint8)
            colors = palette[color_indexes]
            colors = np.concatenate(
                [colors, 255 * np.ones((*colors.shape[:2], 1), dtype=np.uint8)],
                axis=2,
            )
            self.sfc.setData(z=100 * pgm, colors=colors)

    def _set_camera_checked(self, checked):
        if not hasattr(self, "radioButton"):
            return
        blocked = self.radioButton.blockSignals(True)
        self.radioButton.setChecked(checked)
        self.radioButton.blockSignals(blocked)

    def _toggle_camera(self, checked):
        if checked:
            self.turnoncamera()
        elif self.config.SourceType == "cam":
            source_type, source_path = self._source_before_camera

            self.stat.Pause = 1
            self.config.SourceType = source_type
            self.vidpath = source_path
            self.fifo.changeConfig(self.config)
            self._request_handler_config(source_path)

            self.toolButton_4.setText("Старт")
            self._set_status("Камера отключена")

    def turnoncamera(self):
        preview = cv.VideoCapture(0)
        if not preview.isOpened():
            self.metricSource.setText("\u041a\u0430\u043c\u0435\u0440\u0430")
            self.metricFace.setText("\u043d\u0435\u0442 \u0441\u0438\u0433\u043d\u0430\u043b\u0430")
            self._set_status("\u041a\u0430\u043c\u0435\u0440\u0430 \u043d\u0435\u0434\u043e\u0441\u0442\u0443\u043f\u043d\u0430")
            self._set_camera_checked(False)
            return

        ret, frame = preview.read()
        ret, frame = preview.read()
        frame= cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        preview.release()
        if not ret or frame is None:
            self.metricSource.setText("\u041a\u0430\u043c\u0435\u0440\u0430")
            self.metricFace.setText("\u043d\u0435\u0442 \u043a\u0430\u0434\u0440\u0430")
            self._set_status("\u041a\u0430\u043c\u0435\u0440\u0430 \u043e\u0442\u043a\u0440\u044b\u0442\u0430, \u043d\u043e \u043a\u0430\u0434\u0440 \u043d\u0435 \u043f\u043e\u043b\u0443\u0447\u0435\u043d")
            self._set_camera_checked(False)
            return

        # self.vidpath = 0
        self.Frame.value = frame
        self.metricSource.setText("\u041a\u0430\u043c\u0435\u0440\u0430")
        self.metricFace.setText("\u043e\u0436\u0438\u0434\u0430\u043d\u0438\u0435 ROI")
        self._show_frame(frame)
        self._set_status("\u041a\u0430\u043c\u0435\u0440\u0430 \u0433\u043e\u0442\u043e\u0432\u0430, \u043d\u0430\u0436\u043c\u0438\u0442\u0435 \u0421\u0442\u0430\u0440\u0442")
        if self.config.SourceType != "cam":
            self._source_before_camera = (
                self.config.SourceType,
                self.vidpath,
            )

        self.config.SourceType = "cam"
        self.fifo.changeConfig(self.config)

        self._request_handler_config(0)


    def reset(self):
        self.stat.Pause = 1
        self.stat.reset = 0
        self.fifo.reset()
        self.toolButton_4.setText("\u0421\u0442\u0430\u0440\u0442")
        self.metricFace.setText("\u043e\u0436\u0438\u0434\u0430\u043d\u0438\u0435")
        self.metricHeartRate.setText("--")
        self._set_status("\u0414\u0430\u043d\u043d\u044b\u0435 \u0441\u0431\u0440\u043e\u0448\u0435\u043d\u044b")
        self.cr_curve.setData([], [])
        self.if_curve.setData([], [])
        self.confidence_if[0].setData([], [])
        self.confidence_if[1].setData([], [])
        self.PGMLine.setData([], [])
        self.PGMScatter.setData([], [])

        for item in self.PGMScatterplots:
            self.plotPGM.removeItem(item)

        self.PGMScatterplots.clear()
        self.EigScatters.clear()

    def on_timer(self):
        if not self.fifo.sta.Pause:
            if self.fifo.frameCounter<self.config.segLength-1:
                # print(f"processing {self.fifo.frameCounter}/{self.config.segLength}\n\t\t{self.fifo.frameCounter/self.config.segLength *100:.1f}%, {self.srcHandler.handlerFinished}")
                self._set_status(f"Обработка: {self.fifo.frameCounter}/{self.config.segLength}\n\t\t{self.fifo.frameCounter/self.config.segLength *100:.1f}%")
                
            else:
                self._set_status(f"Обработка завершена\n\t\t{self.fifo.frameCounter/self.config.segLength *100:.1f}%")



        if self.stat.reset:
            self.fifo.reset()
            self.stat.reset = 0

    def displayPlot(self):
        if self.fifo.sta.Pause and self.stat.stopPlotting:#and not (self.stat.freqready or self.stat.pwready):
            return

        if self.stat.freqready and len(self.fifo.pgm) > 0 and len(self.IF.values) > 0:
            

            y = np.asarray(self.IF.values[-len(self.IF.values):], dtype=float).reshape(len(self.IF.values),2)*60
            x = np.arange(y.shape[0]) * self.config.strapLen / self.config.cad
            self.if_curve.setData(x, y[:,0])
            self.confidence_if[1].setData(x, y[:,0]- y[:,1]) 
            self.confidence_if[0].setData(x, y[:,0]+ y[:,1]) 
            self._update_heart_rate_metric()

            try:
                pass#self._update_periodogram_surface()
            except Exception as exc:
                self._set_status(f"\u041e\u0448\u0438\u0431\u043a\u0430 \u043e\u0442\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u044f \u043f\u0435\u0440\u0438\u043e\u0434\u043e\u0433\u0440\u0430\u043c\u043c\u044b: {exc}")
            self.stat.freqready = 0

        if self.stat.pwready and len(self.fifo.CR) > 0:
            self.stat.pwready = 0
             
            if self.config.plotSource=="pgm":
                print(self.plotCR.getAxis("bottom").labelText)
                if self.plotCR.getAxis("bottom").labelText!="Частота, Гц":
                    self.plotCR.setLabel("bottom", "Частота, Гц")
                    self.plotCR.setLabel("left", "Мощность")
                 
                CR = np.asarray(self.fifo.CR[-1], dtype=float).reshape(-1)[:int(self.config.fMa/60/(self.config.cad/self.config.segLength))+20]
                x = np.arange(CR.shape[0]) * (self.config.cad / self.config.segLength)
            elif self.config.plotSource=="eigvls":
                CR = np.asarray(
                                    self.fifo.CR[-1],
                                    dtype=float,
                                ).reshape(-1)
                # print(f"EIG shape = {CR.shape}")
                x = np.arange(CR.shape[0])  
                
            else:
                if self.plotCR.getAxis("bottom").labelText!="Время, сек":
                    self.plotCR.setLabel("bottom", "Время, сек")
                    self.plotCR.setLabel("left", "Амплитуда")
                CR = np.asarray(
                    self.fifo.CR[-1],
                    dtype=float,
                ).reshape(-1)[:self.config.segLength]

                segment_index = len(self.fifo.CR) - 1
                segment_start_s = (
                    segment_index
                    * self.config.strapLen
                    / self.config.cad
                )

                # APG обрезается на 10 отсчётов в get_IF().
                sample_offset = (
                    10 if self.config.plotSource == "deriv" else 0
                )

                x = (
                    segment_start_s
                    + (sample_offset + np.arange(CR.size))
                    / self.config.cad
                )

            self.cr_curve.setData(x, CR)
            if self.config.plotSource=="eigvls":
                self.EigScatters.setData(x,CR)
                self.plotCR.setLabel("bottom", "Номер")
            else:
                self.EigScatters.clear()
            x = np.arange(self.fifo.pgm[-1].shape[1]//2+1) * (self.config.cad / self.config.segLength)
            y = self.fifo.pgm[-1][0,:self.config.segLength//2+1]
            self.PGMScatterplots.append(self.plotPGM.scatterPlot(x,y, size = 1.5,pen = QtGui.QPen( "blue")))
            self.PGMScatterplots[-1].setOpacity(0.2)
            self.PGMLine.setData(x,y)
            if len(self.PGMScatterplots)<50:
                return
            for plot in self.PGMScatterplots[:-50]:
                self.PGMScatterplots.pop(0)
                plot.clear()
                self.plotPGM.removeItem(plot)

    def displayFace(self):
        
        frame = self.Frame.value
        if frame is None:
            return

        roi = self._last_roi
        has_face = (
            roi is not None
            and len(roi) > 0
            and roi[0] is not None
        )

        self.metricFace.setText(
            "найдено" if has_face else "лицо не найдено"
        )
        
        self._show_frame(frame)

    def PausePlay(self):
        self.stat.Pause ^= 1
        self.stat.stopPlotting ^= 1
         
        if self.stat.Pause:
            # self.configpopup.show()
            self.toolButton_4.setText("\u0421\u0442\u0430\u0440\u0442")
            self._set_status("\u041f\u0430\u0443\u0437\u0430")
        else:
            # self.configpopup.hide()
            self.toolButton_4.setText("\u041f\u0430\u0443\u0437\u0430")
            self._set_status("\u041e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0430")
             
            if self.srcHandler.handlerFinished:
                if len(self.fifo.threadmngr._pending_segments)>0:
                    QtWidgets.QMessageBox(self,text="Обработка завершена, идет дообработка сегментов").show()

                    return

    def on_click(self, event):
        azim, elev = self._ax.azim, self._ax.elev
 
    def set_canvas_table_configuration(self, row_count, data):
        self.fig.set_canvas(self.canvas)
        self._ax = self.canvas.figure.add_subplot(projection="3d")

        
     
    def _request_handler_config(self, filepath=None):
   
        effective_path = (
        self.vidpath if filepath is None else filepath
    )
        self.srcHandler.ChangeConfig(
            self.config,
            effective_path,
        )
             

    def _changeConfig(self, newConfig: Config = None, newFilepath=None):
        self.stat.Pause = 1
        if newConfig is not None:
            self.config = newConfig
        if newFilepath is not None:
            self.vidpath = newFilepath
        self.fifo.changeConfig(self.config)
        self._request_handler_config(self.vidpath)

    def choosefile(self, fmt: str = "*.mp4"):
        fmt = self.toolButton.st.myname
        caption = "\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0432\u0438\u0434\u0435\u043e" if fmt == "*.mp4" else "\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0442\u0435\u043a\u0441\u0442\u043e\u0432\u044b\u0439 \u0444\u0430\u0439\u043b"
        file_filter = (
            "\u0412\u0438\u0434\u0435\u043e (*.mp4 *.avi *.mov);;\u0412\u0441\u0435 \u0444\u0430\u0439\u043b\u044b (*)"
            if fmt == "*.mp4"
            else "\u0422\u0435\u043a\u0441\u0442\u043e\u0432\u044b\u0435 \u0444\u0430\u0439\u043b\u044b (*.txt *.csv);;\u0412\u0441\u0435 \u0444\u0430\u0439\u043b\u044b (*)"
        )
        self.vidpath = QtWidgets.QFileDialog(self).getOpenFileName(
            self,
            caption,
            str(self.project_root),
            filter=file_filter,
        )[0]
        if not self.vidpath:
            self._set_status("\u0412\u044b\u0431\u043e\u0440 \u0444\u0430\u0439\u043b\u0430 \u043e\u0442\u043c\u0435\u043d\u0435\u043d")
            return

        self._set_camera_checked(False)
        self.config.SourceType = "video" if fmt == "*.mp4" else "txt"
        self._changeConfig(newConfig=self.config, newFilepath=self.vidpath)
         
        self.on_source_changed()
        self.toolButton_4.setText("\u0421\u0442\u0430\u0440\u0442")

        if fmt == "*.mp4":
            preview = cv.VideoCapture(self.vidpath)
            if not preview.isOpened():
                preview.release()
                self.metricFace.setText("\u043d\u0435\u0442 \u0441\u0438\u0433\u043d\u0430\u043b\u0430")
                self._set_status("\u041d\u0435 \u0443\u0434\u0430\u043b\u043e\u0441\u044c \u043e\u0442\u043a\u0440\u044b\u0442\u044c \u0432\u0438\u0434\u0435\u043e")
                return
            ret, self.Frame.value = preview.read()
            ret, self.Frame.value = preview.read()

            preview.release()
            if not ret or self.Frame.value is None:
                self.metricFace.setText("\u043d\u0435\u0442 \u043a\u0430\u0434\u0440\u0430")
                self._set_status("\u0412\u0438\u0434\u0435\u043e \u043e\u0442\u043a\u0440\u044b\u0442\u043e, \u043d\u043e \u043f\u0435\u0440\u0432\u044b\u0439 \u043a\u0430\u0434\u0440 \u043d\u0435 \u043f\u043e\u043b\u0443\u0447\u0435\u043d")
                return
            self._request_handler_config(self.vidpath)
            self.metricSource.setText("\u0412\u0438\u0434\u0435\u043e")
            self.metricFace.setText("\u043e\u0436\u0438\u0434\u0430\u043d\u0438\u0435 ROI")
             
            self._show_frame(self.Frame.value)
            self._set_status("\u0412\u0438\u0434\u0435\u043e \u0437\u0430\u0433\u0440\u0443\u0436\u0435\u043d\u043e, \u043d\u0430\u0436\u043c\u0438\u0442\u0435 \u0421\u0442\u0430\u0440\u0442")
        elif fmt == "*.txt":
            try:
                try:
                    PW = np.loadtxt(self.vidpath, delimiter=",")
                except ValueError:
                    PW = np.loadtxt(self.vidpath)
            except Exception as exc:
                self.metricFace.setText("\u043e\u0448\u0438\u0431\u043a\u0430 TXT")
                self._set_status(f"\u041d\u0435 \u0443\u0434\u0430\u043b\u043e\u0441\u044c \u043f\u0440\u043e\u0447\u0438\u0442\u0430\u0442\u044c TXT: {exc}")
                return
            self.metricSource.setText("TXT")
            self.metricFace.setText("\u043d\u0435 \u0442\u0440\u0435\u0431\u0443\u0435\u0442\u0441\u044f")
            self._last_frame_pixmap = None
            self.figvid.clear()
            self.figvid.setText("\u0417\u0430\u0433\u0440\u0443\u0436\u0435\u043d \u0442\u0435\u043a\u0441\u0442\u043e\u0432\u044b\u0439 \u0441\u0438\u0433\u043d\u0430\u043b")
            self._set_status(f"TXT \u0437\u0430\u0433\u0440\u0443\u0436\u0435\u043d: {PW.shape[0]} \u043e\u0442\u0441\u0447\u0435\u0442\u043e\u0432. \u041d\u0430\u0436\u043c\u0438\u0442\u0435 \u0421\u0442\u0430\u0440\u0442")

    def closeEvent(self, event):
        self.stat.Pause = 1
        if hasattr(self, "srcHandler"):
            self.srcHandler.handlerFinished = 1
        if self.cap.value is not None:
            self.cap.value.release()
        if hasattr(self, "fifo") and hasattr(self.fifo, "threadmngr"):
            self.fifo.threadmngr.stop()
        for thread in (getattr(self, "ROIth", None), getattr(self, "FIFOth", None)):
            if thread is not None:
                thread.quit()
                thread.wait(1500)
        super().closeEvent(event)

    def changest(self):
        next_index = 1 if self.sourceCombo.currentData() == "*.mp4" else 0
        self.sourceCombo.setCurrentIndex(next_index)
