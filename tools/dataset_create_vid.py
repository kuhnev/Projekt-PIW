import cv2
FPS = 60
SCALE_FACTOR = 4

def dataset_jpg_to_vid():
    img_array = []
    size = (0, 0)
    for i in range(25):
        for j in range(2):
            filename = "../dataset/" + str(i + 1) + "/TD_RGB_E_" + str(j + 1) + ".jpg"
            print(filename)
            img = cv2.imread(filename)
            height, width, layers = img.shape
            img = cv2.resize(img, (int(width/SCALE_FACTOR), int(height/SCALE_FACTOR)))
            height, width, layers = img.shape
            size = (width,height)
            print(size)
            img_array.append(img)


    out = cv2.VideoWriter('../dataset/dataset.avi',cv2.VideoWriter_fourcc(*'DIVX'), FPS, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def main():
    dataset_jpg_to_vid()

if __name__ == "__main__":
    main()