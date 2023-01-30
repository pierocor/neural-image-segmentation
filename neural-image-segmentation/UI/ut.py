# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def __init__(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.showMaximized()
        MainWindow.setMinimumSize(1100, 750)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralWidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(20, 10, 161, 170))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.preButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.preButton.setObjectName("preButton")
        self.verticalLayout.addWidget(self.preButton)
        self.pushButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout.addWidget(self.pushButton)
        self.analyzeButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.analyzeButton.setObjectName("analyzeButton")
        self.verticalLayout.addWidget(self.analyzeButton)
        self.clearButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.clearButton.setObjectName("clearButton")
        self.verticalLayout.addWidget(self.clearButton)
        self.graphicsView = QtWidgets.QGraphicsView(self.centralWidget)
        self.graphicsView.setEnabled(True)
        self.graphicsView.setGeometry(QtCore.QRect(190, 40, 853, 640))
        self.graphicsView.setObjectName("graphicsView")
        # self.progressBar = QtWidgets.QProgressBar(self.centralWidget)
        # self.progressBar.setGeometry(QtCore.QRect(370, 390, 118, 23))
        # self.progressBar.setProperty("value", 24)
        # self.progressBar.setObjectName("progressBar")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralWidget)
        self.textBrowser.setGeometry(QtCore.QRect(190, 10, 800, 23))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textBrowser.sizePolicy().hasHeightForWidth())
        self.textBrowser.setSizePolicy(sizePolicy)
        self.textBrowser.setObjectName("textBrowser")
        self.toolButton = QtWidgets.QToolButton(self.centralWidget)
        self.toolButton.setGeometry(QtCore.QRect(1000, 10, 41, 22))
        self.toolButton.setObjectName("toolButton")
        MainWindow.setCentralWidget(self.centralWidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 667, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Neural Image Segmentation"))
        self.label.setText(_translate("MainWindow", "  Input Image Select:"))
        self.analyzeButton.setText(_translate("MainWindow", "Analyze"))
        self.preButton.setText(_translate("MainWindow", "Image Process"))
        self.pushButton.setText(_translate("MainWindow", "Process"))
        self.clearButton.setText(_translate("MainWindow", "Reset"))
        self.toolButton.setText(_translate("MainWindow", "..."))
