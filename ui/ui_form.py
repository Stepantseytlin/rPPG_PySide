# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.11.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QLabel,
    QPushButton, QRadioButton, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget)

from domain.contentBtn import contentBtn2

class Ui_Widget(object):
    def setupUi(self, Widget):
        if not Widget.objectName():
            Widget.setObjectName(u"Widget")
        Widget.resize(1149, 609)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Widget.sizePolicy().hasHeightForWidth())
        Widget.setSizePolicy(sizePolicy)
        self.gridLayout = QGridLayout(Widget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.PulseWave_2 = QVBoxLayout()
        self.PulseWave_2.setObjectName(u"PulseWave_2")
        self.verticalLayout_7 = QVBoxLayout()
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.toolButton_4 = QPushButton(Widget)
        self.toolButton_4.setObjectName(u"toolButton_4")

        self.horizontalLayout_2.addWidget(self.toolButton_4, 0, Qt.AlignmentFlag.AlignTop)

        self.toolButton = contentBtn2(Widget)
        self.toolButton.setObjectName(u"toolButton")
        self.toolButton.setText("Открыть файл")
        self.horizontalLayout_2.addWidget(self.toolButton, 0, Qt.AlignmentFlag.AlignTop)

        self.toolButton_5 = QPushButton(Widget)
        self.toolButton_5.setObjectName(u"toolButton_5")

        self.horizontalLayout_2.addWidget(self.toolButton_5, 0, Qt.AlignmentFlag.AlignTop)

        self.toolButton_2 = QPushButton(Widget)
        self.toolButton_2.setObjectName(u"toolButton_2")

        self.horizontalLayout_2.addWidget(self.toolButton_2, 0, Qt.AlignmentFlag.AlignTop)

        self.horizontalLayoutfloor2 = QHBoxLayout()

        self.verticalConfigLayout = QVBoxLayout()

        self.verticalConfigLayout.addLayout(self.horizontalLayout_2)

        self.verticalConfigLayout.addLayout(self.horizontalLayoutfloor2)


        self.verticalLayout_3.addLayout(self.verticalConfigLayout)

        self.verticalLayout_5.addLayout(self.verticalLayout_3)


        self.verticalLayout_7.addLayout(self.verticalLayout_5)

        self.radioButton = QRadioButton(Widget)
        self.radioButton.setObjectName(u"radioButton")
        self.radioButton.setCursor(QCursor(Qt.CursorShape.CrossCursor))

        self.radioButtonINST = QRadioButton(Widget)
        self.radioButtonINST.setObjectName(u"radioButtonINST")
        self.radioButtonINST.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self.radioButtonINST.setChecked(1)
        
        self.horLayout_radbut = QHBoxLayout()
        self.horLayout_radbut.addWidget(self.radioButton, 0, Qt.AlignmentFlag.AlignHCenter)
        self.horLayout_radbut.addWidget(self.radioButtonINST, 1, Qt.AlignmentFlag.AlignRight)

        self.verticalLayout_7.addLayout(self.horLayout_radbut)

        # self.verticalLayout_7.addWidget(self.radioButton, 0, Qt.AlignmentFlag.AlignHCenter)

        self.verticalLayout_6 = QVBoxLayout()
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.figvid = QLabel(Widget)
        self.figvid.setObjectName(u"figvid")
        self.figvid.setMinimumSize(QSize(200, 200))
        self.figvid.setMaximumSize(QSize(200, 200))
        self.figvid.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.figvid.setMargin(3)

        self.verticalLayout_6.addWidget(self.figvid, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)


        self.verticalLayout_7.addLayout(self.verticalLayout_6)


        self.PulseWave_2.addLayout(self.verticalLayout_7)

        self.RPPGWidget = QWidget(Widget)
        self.RPPGWidget.setObjectName(u"RPPGWidget")
        self.RPPGWidget.setMinimumSize(QSize(400, 400))
        self.RPPGWidget.setMaximumSize(QSize(600, 600))

        self.PulseWave_2.addWidget(self.RPPGWidget)


        self.horizontalLayout.addLayout(self.PulseWave_2)


        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)

        self.plots = QScrollArea(Widget)
        self.plots.setObjectName(u"plots")
        self.plots.setMaximumSize(QSize(900, 900))
        self.plots.setWidgetResizable(True)
        self.plots1 = QWidget()
        self.plots1.setObjectName(u"plots1")
        self.plots1.setGeometry(QRect(0, 0, 714, 595))
        self.verticalLayout = QVBoxLayout(self.plots1)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.PulseWave = QWidget(self.plots1)
        self.PulseWave.setObjectName(u"PulseWave")
        self.label = QLabel(self.PulseWave)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(330, 10, 37, 12))

        self.verticalLayout.addWidget(self.PulseWave)

        self.Fr = QWidget(self.plots1)
        self.Fr.setObjectName(u"Fr")
        palette = QPalette()
        brush = QBrush(QColor(0, 0, 0, 255))
        brush.setStyle(Qt.BrushStyle.SolidPattern)
        palette.setBrush(QPalette.ColorGroup.Active, QPalette.ColorRole.BrightText, brush)
        palette.setBrush(QPalette.ColorGroup.Inactive, QPalette.ColorRole.BrightText, brush)
        palette.setBrush(QPalette.ColorGroup.Disabled, QPalette.ColorRole.BrightText, brush)
        self.Fr.setPalette(palette)
        self.label_2 = QLabel(self.Fr)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(340, 10, 71, 16))
        palette1 = QPalette()
        palette1.setBrush(QPalette.ColorGroup.Active, QPalette.ColorRole.WindowText, brush)
        palette1.setBrush(QPalette.ColorGroup.Active, QPalette.ColorRole.Button, brush)
        palette1.setBrush(QPalette.ColorGroup.Active, QPalette.ColorRole.Light, brush)
        palette1.setBrush(QPalette.ColorGroup.Active, QPalette.ColorRole.Midlight, brush)
        palette1.setBrush(QPalette.ColorGroup.Active, QPalette.ColorRole.Dark, brush)
        palette1.setBrush(QPalette.ColorGroup.Active, QPalette.ColorRole.Mid, brush)
        brush1 = QBrush(QColor(255, 255, 255, 255))
        brush1.setStyle(Qt.BrushStyle.SolidPattern)
        palette1.setBrush(QPalette.ColorGroup.Active, QPalette.ColorRole.Text, brush1)
        palette1.setBrush(QPalette.ColorGroup.Active, QPalette.ColorRole.BrightText, brush1)
        palette1.setBrush(QPalette.ColorGroup.Active, QPalette.ColorRole.ButtonText, brush1)
        palette1.setBrush(QPalette.ColorGroup.Active, QPalette.ColorRole.Base, brush)
        palette1.setBrush(QPalette.ColorGroup.Active, QPalette.ColorRole.Window, brush)
        palette1.setBrush(QPalette.ColorGroup.Active, QPalette.ColorRole.Shadow, brush)
        palette1.setBrush(QPalette.ColorGroup.Active, QPalette.ColorRole.AlternateBase, brush)
        brush2 = QBrush(QColor(255, 255, 220, 255))
        brush2.setStyle(Qt.BrushStyle.SolidPattern)
        palette1.setBrush(QPalette.ColorGroup.Active, QPalette.ColorRole.ToolTipBase, brush2)
        palette1.setBrush(QPalette.ColorGroup.Active, QPalette.ColorRole.ToolTipText, brush)
        brush3 = QBrush(QColor(255, 255, 255, 127))
        brush3.setStyle(Qt.BrushStyle.SolidPattern)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette1.setBrush(QPalette.ColorGroup.Active, QPalette.ColorRole.PlaceholderText, brush3)
#endif
#if QT_VERSION >= QT_VERSION_CHECK(6, 6, 0)
        palette1.setBrush(QPalette.ColorGroup.Active, QPalette.ColorRole.Accent, brush)
#endif
        palette1.setBrush(QPalette.ColorGroup.Inactive, QPalette.ColorRole.WindowText, brush)
        palette1.setBrush(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Button, brush)
        palette1.setBrush(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Light, brush)
        palette1.setBrush(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Midlight, brush)
        palette1.setBrush(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Dark, brush)
        palette1.setBrush(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Mid, brush)
        palette1.setBrush(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Text, brush1)
        palette1.setBrush(QPalette.ColorGroup.Inactive, QPalette.ColorRole.BrightText, brush1)
        palette1.setBrush(QPalette.ColorGroup.Inactive, QPalette.ColorRole.ButtonText, brush1)
        palette1.setBrush(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Base, brush)
        palette1.setBrush(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Window, brush)
        palette1.setBrush(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Shadow, brush)
        palette1.setBrush(QPalette.ColorGroup.Inactive, QPalette.ColorRole.AlternateBase, brush)
        palette1.setBrush(QPalette.ColorGroup.Inactive, QPalette.ColorRole.ToolTipBase, brush2)
        palette1.setBrush(QPalette.ColorGroup.Inactive, QPalette.ColorRole.ToolTipText, brush)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette1.setBrush(QPalette.ColorGroup.Inactive, QPalette.ColorRole.PlaceholderText, brush3)
#endif
#if QT_VERSION >= QT_VERSION_CHECK(6, 6, 0)
        palette1.setBrush(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Accent, brush)
#endif
        palette1.setBrush(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, brush)
        palette1.setBrush(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Button, brush)
        palette1.setBrush(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Light, brush)
        palette1.setBrush(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Midlight, brush)
        palette1.setBrush(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Dark, brush)
        palette1.setBrush(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Mid, brush)
        palette1.setBrush(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, brush)
        palette1.setBrush(QPalette.ColorGroup.Disabled, QPalette.ColorRole.BrightText, brush1)
        palette1.setBrush(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, brush)
        palette1.setBrush(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, brush)
        palette1.setBrush(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Window, brush)
        palette1.setBrush(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Shadow, brush)
        palette1.setBrush(QPalette.ColorGroup.Disabled, QPalette.ColorRole.AlternateBase, brush)
        palette1.setBrush(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ToolTipBase, brush2)
        palette1.setBrush(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ToolTipText, brush)
        brush4 = QBrush(QColor(0, 0, 0, 127))
        brush4.setStyle(Qt.BrushStyle.SolidPattern)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette1.setBrush(QPalette.ColorGroup.Disabled, QPalette.ColorRole.PlaceholderText, brush4)
#endif
#if QT_VERSION >= QT_VERSION_CHECK(6, 6, 0)
        palette1.setBrush(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Accent, brush)
#endif
        self.label_2.setPalette(palette1)

        self.verticalLayout.addWidget(self.Fr)

        self.Frequency = QWidget(self.plots1)
        self.Frequency.setObjectName(u"Frequency")
        self.label_3 = QLabel(self.Frequency)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(290, 10, 151, 16))

        self.verticalLayout.addWidget(self.Frequency)

        self.plots.setWidget(self.plots1)

        self.gridLayout.addWidget(self.plots, 0, 1, 1, 1)


        self.retranslateUi(Widget)

        QMetaObject.connectSlotsByName(Widget)
    # setupUi

    def retranslateUi(self, Widget):
        Widget.setWindowTitle(QCoreApplication.translate("Widget", u"Widget", None))
        self.toolButton_4.setText(QCoreApplication.translate("Widget", u"\u041f\u0430\u0443\u0437\u0430", None))
        # self.toolButton.setText(QCoreApplication.translate("Widget", u"\u0412\u044b\u0431\u0440\u0430\u0442\u044c \u0412\u0438\u0434\u0435\u043e", None))
        self.toolButton_5.setText(QCoreApplication.translate("Widget", u"\u0421\u0431\u0440\u043e\u0441\u0438\u0442\u044c", None))
        self.toolButton_2.setText(QCoreApplication.translate("Widget", u"\u041f\u043e\u043c\u0435\u043d\u044f\u0442\u044c \u0442\u0438\u043f \u0434\u0430\u043d\u043d\u044b\u0445", None))
        self.radioButton.setText(QCoreApplication.translate("Widget", u"\u0418\u0441\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u044c \u043a\u0430\u043c\u0435\u0440\u0443", None))
        self.figvid.setText("")
        self.label.setText(QCoreApplication.translate("Widget", u"Cr", None))
        self.label_2.setText(QCoreApplication.translate("Widget", u"\u0427\u0421\u0421", None))
        self.label_3.setText(QCoreApplication.translate("Widget", u"\u041f\u0435\u0440\u0438\u043e\u0434\u043e\u0433\u0440\u0430\u043c\u043c\u0430 \u0422\u043e\u043c\u0441\u043e\u043d\u0430", None))
    # retranslateUi

