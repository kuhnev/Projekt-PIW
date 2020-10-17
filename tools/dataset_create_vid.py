import cv2
import glob

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

def dataset_jpg_to_vid_2():
    img_array = []
    shape = (1024, 1024)
    jpgs = glob.glob("../dataset/dataset_2/19--Couple/*.jpg")
    for jpg in jpgs:
        jpg = jpg.replace("\\", "/")
        print(jpg)
        img = cv2.imread(jpg)
        img_resized = cv2.resize(img, shape)
        img_array.append(img_resized)

    size = shape

    out = cv2.VideoWriter('../dataset/dataset_2_1.avi',cv2.VideoWriter_fourcc(*'DIVX'), FPS, size)
    print(len(img_array))
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def main():
    # dataset_jpg_to_vid()
    dataset_jpg_to_vid_2()

if __name__ == "__main__":
    main()