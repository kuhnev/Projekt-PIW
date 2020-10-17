import cv2
import numpy as np
import detectors
from PyQt5.QtWidgets import  QWidget, QLabel, QVBoxLayout, QPushButton
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QFont, QDoubleValidator
from capture import MyVideoCapture

class Data:
    def __init__(self):
        self.scale = 1.6
        self.neighboor = 5

    def getScale(self):
        return self.scale

    def getNeighboor(self):
        return self.neighboor

    def setScale(self,scale):
        self.scale = scale

    def setNeighboor(self,neighboor):
        self.neighboor = neighboor


class Thread(QThread):
    changePixmap = pyqtSignal(np.ndarray)

    def __init__(self, parent, capture_source):
        super().__init__(parent)
        self.capture_source = capture_source
        self.get_next_frame = True

    def read_next_frame(self):
        self.get_next_frame = True

    def run(self):
        cap = MyVideoCapture(self.capture_source)
        while True:
            if self.get_next_frame:
                ret, frame = cap.get_frame()
                if ret:
                    self.get_next_frame = False
                    self.changePixmap.emit(frame)


class App(QWidget):

    update_params = pyqtSignal()
    request_next_frame = pyqtSignal()
    reload_frame = pyqtSignal(np.ndarray)


    def __init__(self, capture_source = 0):
        super().__init__()
        self.face_detector = detectors.FaceDetector(scale_factor=1.3, min_neighbors=5)
        self.smile_detector = detectors.SmileDetector(scale_factor=1.3, min_neighbors=60)

        self.last_processed_frame = None

        self.update_params.connect(self.setParams)

        self.initUI()

        th = Thread(self, capture_source)
        self.request_next_frame.connect(th.read_next_frame)

        th.changePixmap.connect(self.setImage)
        self.reload_frame.connect(self.setImage)

        th.start()

    def setImage(self, raw_frame):
        self.last_processed_frame = np.copy(raw_frame)
        frame = np.copy(raw_frame)

        faces = self.face_detector.detect_on_frame(frame)
        self.face_detector.label_on_frame(frame, faces, (0,0,255))

        for (x, y, w, h) in faces:
            frame_face_part = frame[y:y+h, x:x+w]
            smiles = self.smile_detector.detect_on_frame(frame_face_part)
            smiles = self.abandon_impossible_smiles(frame_face_part, smiles)
            for smile in smiles:
                smile[0] = smile[0] + x
                smile[1] = smile[1] + y

            self.smile_detector.label_on_frame(frame, smiles, (255,0,0))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytesPerLine = ch * w
        rgb_frame_qt_format = QImage(rgb_frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
        rgb_frame_qt_format_scaled = rgb_frame_qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        self.label.setPixmap(QPixmap.fromImage(rgb_frame_qt_format_scaled))

    def setParams(self):
        self.face_factor_text.setText("Face detection scale factor: " 
                                    + str(self.face_detector.get_scale_factor()))
        self.face_neighboor_text.setText("Face detection min neighboor: " 
                                    + str(self.face_detector.get_min_neighbors()))
        self.smile_factor_text.setText("Smile detection scale factor: " 
                                    + str(self.smile_detector.get_scale_factor()))
        self.smile_neighboor_text.setText("Smile detection min neighboor: " 
                                    + str(self.smile_detector.get_min_neighbors()))

        if self.last_processed_frame is not None:
            self.reload_frame.emit(self.last_processed_frame)

    def initUI(self):
        self.setWindowTitle("Smile detector")

        self.vbox = QVBoxLayout(self)

        self.label = QLabel(self)
        self.vbox.addWidget(self.label)
        self.vbox.addStretch()

        self.btn_next_frame = QPushButton(self)
        self.btn_next_frame.setText("-> Next frame ->")
        self.btn_next_frame.clicked.connect(self.show_next_frame)
        self.vbox.addWidget(self.btn_next_frame)

        self.face_factor_text = QLabel(self)
        self.face_factor_text.setAlignment(Qt.AlignCenter)
        self.vbox.addWidget(self.face_factor_text)

        self.face_neighboor_text = QLabel(self)
        self.face_neighboor_text.setAlignment(Qt.AlignCenter)
        self.vbox.addWidget(self.face_neighboor_text)

        self.smile_factor_text = QLabel(self)
        self.smile_factor_text.setAlignment(Qt.AlignCenter)
        self.vbox.addWidget(self.smile_factor_text)

        self.smile_neighboor_text = QLabel(self)
        self.smile_neighboor_text.setAlignment(Qt.AlignCenter)
        self.vbox.addWidget(self.smile_neighboor_text)

        self.update_params.emit()

        self.btn_face_factor_increase = QPushButton(self)
        self.btn_face_factor_increase.setText("+++ Face scale factor +++")
        self.btn_face_factor_increase.clicked.connect(self.increase_facial_scale_factor)
        self.vbox.addWidget(self.btn_face_factor_increase)

        self.btn_face_factor_decrease = QPushButton(self)
        self.btn_face_factor_decrease.setText("- - - Face scale factor - - -")
        self.btn_face_factor_decrease.clicked.connect(self.decrease_facial_scale_factor)
        self.vbox.addWidget(self.btn_face_factor_decrease)

        self.btn_face_neighboor_increase = QPushButton(self)
        self.btn_face_neighboor_increase.setText("+++ Face neighboor +++")
        self.btn_face_neighboor_increase.clicked.connect(self.increase_facial_neighboor)
        self.vbox.addWidget(self.btn_face_neighboor_increase)

        self.btn_face_neighboor_decrease = QPushButton(self)
        self.btn_face_neighboor_decrease.setText(" - - - Face neighboor - - - ")
        self.btn_face_neighboor_decrease.clicked.connect(self.decrease_facial_neighboor)
        self.vbox.addWidget(self.btn_face_neighboor_decrease)


        self.btn_smile_factor_increase = QPushButton(self)
        self.btn_smile_factor_increase.setText("+++ Smile scale factor +++")
        self.btn_smile_factor_increase.clicked.connect(self.increase_smile_scale_factor)
        self.vbox.addWidget(self.btn_smile_factor_increase)

        self.btn_smile_factor_decrease = QPushButton(self)
        self.btn_smile_factor_decrease.setText("- - - Smile scale factor - - -")
        self.btn_smile_factor_decrease.clicked.connect(self.decrease_smile_scale_factor)
        self.vbox.addWidget(self.btn_smile_factor_decrease)

        self.btn_smile_neighboor_increase = QPushButton(self)
        self.btn_smile_neighboor_increase.setText("+++ Smile neighboor +++")
        self.btn_smile_neighboor_increase.clicked.connect(self.increase_smile_neighboor)
        self.vbox.addWidget(self.btn_smile_neighboor_increase)

        self.btn_smile_neighboor_decrease = QPushButton(self)
        self.btn_smile_neighboor_decrease.setText(" - - - Smile neighboor - - - ")
        self.btn_smile_neighboor_decrease.clicked.connect(self.decrease_smile_neighboor)
        self.vbox.addWidget(self.btn_smile_neighboor_decrease)



        self.show()

    def show_next_frame(self):
        self.request_next_frame.emit()

    def abandon_impossible_smiles(self, img_face_part, smiles):
        height, width, layers = img_face_part.shape

        filtered_smiles = list()

        for smile in smiles:
            if (smile[1] > int(height/2)):
                filtered_smiles.append(smile)
        
        return tuple(filtered_smiles)

    def increase_facial_scale_factor(self):
        self.face_detector.set_scale_factor(self.face_detector.get_scale_factor() + 0.05)
        self.update_params.emit()
    
    def decrease_facial_scale_factor(self):
        self.face_detector.set_scale_factor(self.face_detector.get_scale_factor() - 0.05)
        self.update_params.emit()       

    def increase_facial_neighboor(self):
        self.face_detector.set_min_neighbors(self.face_detector.get_min_neighbors() + 1)
        self.update_params.emit()
    
    def decrease_facial_neighboor(self):
        self.face_detector.set_min_neighbors(self.face_detector.get_min_neighbors() - 1)
        self.update_params.emit()
        
    def increase_smile_scale_factor(self):
        self.smile_detector.set_scale_factor(self.smile_detector.get_scale_factor() + 0.05)
        self.update_params.emit()
        
    def decrease_smile_scale_factor(self):
        self.smile_detector.set_scale_factor(self.smile_detector.get_scale_factor() - 0.05)
        self.update_params.emit()
    
    def increase_smile_neighboor(self):
        self.smile_detector.set_min_neighbors(self.smile_detector.get_min_neighbors() + 1)
        self.update_params.emit()
    
    def decrease_smile_neighboor(self):
        self.smile_detector.set_min_neighbors(self.smile_detector.get_min_neighbors() - 1)
        self.update_params.emit()

