# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'driver_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1100, 908)
        palette1 = QtGui.QPalette()
        palette1.setColor(palette1.Background, QtGui.QColor(255, 255, 255))
        MainWindow.setPalette(palette1)
        # MainWindow.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        # MainWindow.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # 设置窗口背景透明
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.rail_view = QtWidgets.QLabel(self.centralwidget)
        self.rail_view.setGeometry(QtCore.QRect(30, 80, 651, 731))
        self.rail_view.setObjectName("rail_view")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(700, 20, 20, 841))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.flag_view = QtWidgets.QLabel(self.centralwidget)
        self.flag_view.setGeometry(QtCore.QRect(730, 70, 351, 341))
        self.flag_view.setObjectName("flag_view")
        self.backgruond = QtWidgets.QLabel(self.centralwidget)
        self.backgruond.setGeometry(QtCore.QRect(0, 0, 1100, 900))
        # self.backgruond.setAutoFillBackground(False)
        self.backgruond.setText("")
        self.backgruond.setPixmap(QtGui.QPixmap("bg.png"))
        self.backgruond.setScaledContents(True)
        # self.backgruond.setAlignment(QtCore.Qt.AlignJustify|QtCore.Qt.AlignTop)
        # self.backgruond.setWordWrap(False)
        # self.backgruond.setOpenExternalLinks(False)
        self.backgruond.setObjectName("backgruond")
        self.output = QtWidgets.QLabel(self.centralwidget)
        self.output.setGeometry(QtCore.QRect(720, 490, 361, 281))
        self.output.setStyleSheet("QLabel{background:white;}\n"
"QLabel{color:rgb(100,100,100,250);font-size:40px;font-weight:bold;font-family:Roman times;}\n"
"QLabel:hover{color:rgb(100,100,100,120);}")
        self.output.setObjectName("output")
        self.output.setAlignment(QtCore.Qt.AlignCenter) #
        self.bbox = QtWidgets.QLabel(self.centralwidget)
        self.bbox.setGeometry(QtCore.QRect(0, 0, 16, 16))
        self.bbox.setStyleSheet("QLabel{border:5px solid rgb(255,0,0)}")
        self.bbox.setObjectName("bbox")
        self.backgruond.raise_()
        self.rail_view.raise_()
        self.line.raise_()
        self.flag_view.raise_()
        self.output.raise_()
        self.bbox.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.rail_view.setText(_translate("MainWindow", "TextLabel"))
        self.flag_view.setText(_translate("MainWindow", "TextLabel"))
        self.output.setText(_translate("MainWindow", "out"))
        self.bbox.setText(_translate("MainWindow", "TextLabel"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

