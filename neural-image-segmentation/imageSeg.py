# This is the main script to start the application for our Neural Image Segmentation project
import warnings
import sys
import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
from PIL import Image, ImageQt
from UI.ut import Ui_MainWindow
from unet.unet_utils import gamma_correction, unet_predict
from postImgProc.utils import *
from postImgProc.alg import *

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class ImgSeg(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow(self)
        self.ui.toolButton.clicked.connect(self.tool_button_callback)
        self.ui.pushButton.clicked.connect(self.process_button_callback)
        self.ui.preButton.clicked.connect(self.pre_button_callback)
        self.ui.analyzeButton.clicked.connect(self.analyze_button_callback)
        self.ui.clearButton.clicked.connect(self.clear_button_callback)
        self.file_name = ""
        self.img = None
        self.gamma_image = None
        self.seg_image = None
        self.show()

    def display_img(self, image):
        img = Image.fromarray(image, mode='RGB')
        qt_img = ImageQt.ImageQt(img)
        pix = QtGui.QPixmap.fromImage(qt_img)
        item = QtWidgets.QGraphicsPixmapItem(pix)
        scene = QtWidgets.QGraphicsScene(self)
        scene.addItem(item)
        self.ui.graphicsView.setPhotoByScnen(scene)

    def tool_button_callback(self):
        self.file_name = QtWidgets.QFileDialog.getOpenFileName(self)[0]
        self.ui.textBrowser.setText(self.file_name)
        if self.file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            self.img = readImg(self.file_name)
            pix = QtGui.QPixmap(self.file_name)
            self.ui.graphicsView.setPhoto(pix)
        else:
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Failed")
            msg.setText("Please check the input file")
            msg.show()

    def pre_button_callback(self):
        if self.img is not None:
            # conduct the process
            if self.gamma_image is None:
                self.gamma_image = gamma_correction(self.img)

            # display the result
            self.display_img(self.gamma_image)
        else:
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Failed")
            msg.setText("Please check the input file")
            msg.show()

    def process_button_callback(self):
        if self.gamma_image is not None:
            # conduct the process
            if self.seg_image is None:
                self.seg_image = unet_predict(self.gamma_image)

            # display the result
            self.display_img(self.seg_image.astype(np.uint8))
        else:
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Failed")
            msg.setText("Warning: Gamma Correction not applied!")
            msg.show()

    def clear_button_callback(self):
        self.ui.textBrowser.setText("")
        self.ui.graphicsView.setPhotoByScnen(None)
        self.img = None
        self.gamma_image = None
        self.seg_image = None

    def analyze_button_callback(self):
        labeled_axon, labeled_cell, nr_cell = separate_axon_and_cell(self.img)
        axons_to_cells = get_touching_dict(labeled_axon, labeled_cell)
        fil = filter_axon(self.img)
        info_list = analyze_axons(fil, labeled_axon, axons_to_cells)
        segmented_axons = getSegmentedAxons(fil, info_list, nr_cell)
        img_annotated = self.img.copy()
        print(img_annotated)
        # for item in info_list:
        #     touch_points = item["touch_points"]
        #     intersect_points = item["intersect_points"]
        #     end_points = item["end_points"]
        #     for coord in touch_points:
        #         img_annotated[coord[0]][coord[1]] = [255, 0, 0]
        #     for coord in intersect_points:
        #         if not type(coord) is tuple:
        #             continue
        #         img_annotated[coord[0]][coord[1]] = [0, 0, 255]
        #     for coord in end_points:
        #         if not type(coord) is tuple:
        #             continue
        #         img_annotated[coord[0]][coord[1]] = [0, 255, 0]
        # display the result
        self.display_img(img_annotated)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    myapp = ImgSeg()
    sys.exit(app.exec())
