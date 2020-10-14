import detectors
import capture
import cv2

# TODO Implement this
def abandon_impossible_smiles(img_face_part, smiles):
    height, width, layers = img_face_part.shape

    filtered_smiles = list()

    for smile in smiles:
        if (smile[1] > int(height/2)):
            filtered_smiles.append(smile)
    
    return tuple(filtered_smiles)


def main():

    face_detector = detectors.FaceDetector(scale_factor=1.1, min_neighbors=10)
    smile_detector = detectors.SmileDetector(scale_factor=1.3, min_neighbors=60)
    frame_capture = capture.MyVideoCapture("dataset/dataset.avi")

    running = True
    while running:
        ret, img = frame_capture.get_frame()
        if (ret):
            faces = face_detector.detect_on_frame(img)
            face_detector.label_on_frame(img, faces, (0,0,255))

            for (x, y, w, h) in faces:
                img_face_part = img[y:y+h, x:x+w]
                smiles = smile_detector.detect_on_frame(img_face_part)
                smiles = abandon_impossible_smiles(img_face_part, smiles)
                for smile in smiles:
                    smile[0] = smile[0] + x
                    smile[1] = smile[1] + y

                smile_detector.label_on_frame(img, smiles, (255,0,0))
            
            cv2.imshow('img', img)
            cv2.waitKey()
        else:
            running = False


if __name__ == "__main__":
    main()