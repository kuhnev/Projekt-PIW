import cv2
from abc import ABC, abstractmethod

FACE_HAAR_CASCADE  = "./cascades/haarcascade_frontalface_default.xml"
SMILE_HAAR_CASCADE = "./cascades/haarcascade_smile.xml"

class Detector(ABC):
    def __init__(self, scale_factor, min_neighbors, cascade_path):
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
    
    @abstractmethod
    def detect_on_frame(self, frame):
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect objects
        objects_detected = self.face_cascade.detectMultiScale(gray_frame, self.scale_factor, self.min_neighbors)

        return objects_detected

    @abstractmethod
    def label_on_frame(self, frame, objects, color = (0, 255, 0)):
        for (x, y, w, h) in objects:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    def get_scale_factor(self):
        return self.scale_factor

    def get_min_neighbors(self):
        return self.min_neighbors

    def set_scale_factor(self, scale_factor):
        self.scale_factor = scale_factor

    def set_min_neighbors(self, min_neighbors):
        self.min_neighbors = min_neighbors

class FaceDetector(Detector):
    def __init__(self, scale_factor = 1.3, min_neighbors = 5, cascade_path = FACE_HAAR_CASCADE):
        super().__init__(scale_factor, min_neighbors, cascade_path)

    def detect_on_frame(self, frame):
        faces = super().detect_on_frame(frame)
        return faces

    def label_on_frame(self, frame, faces, color = (0, 255, 0)):
        frame = super().label_on_frame(frame, faces, color)



class SmileDetector(Detector):
    def __init__(self, scale_factor = 1.8, min_neighbors = 20, cascade_path = SMILE_HAAR_CASCADE):
        # self.face_cascade = cv2.CascadeClassifier(cascade_path)
        super().__init__(scale_factor, min_neighbors, cascade_path)

    def detect_on_frame(self, frame):
        smiles = super().detect_on_frame(frame)
        return smiles

    def label_on_frame(self, frame, smiles, color):
        frame = super().label_on_frame(frame, smiles, color)
