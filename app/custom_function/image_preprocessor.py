import cv2
import numpy as np
import os

class ImagePreprocessor:
    def __init__(self, output_dir="/tmp"):
        self.output_dir = output_dir

    def preprocess(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (640, 640))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        print(f"Image brightness: {brightness:.2f}")

        if brightness < 80:
            gamma = 1.5
        elif brightness > 180:
            gamma = 0.7
        else:
            gamma = 1.0

        if gamma != 1.0:
            lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(256)]).astype("uint8")
            image = cv2.LUT(image, lut)
            print(f"Applied gamma correction with gamma = {gamma}")

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_gray = clahe.apply(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        image = cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2BGR)
        print("Applied CLAHE for contrast enhancement.")

        image = cv2.GaussianBlur(image, (3, 3), 0)
        print("Applied Gaussian Blur.")

        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image = cv2.filter2D(image, -1, sharpen_kernel)
        print("Applied sharpening filter.")

        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"Laplacian variance (blur score): {blur_score:.2f}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        filename = os.path.basename(image_path).replace(".jpg", "_pre.jpg")
        preprocessed_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(preprocessed_path, image)

        return preprocessed_path , blur_score
    

class ObstacleImagePreprocessor:
    def __init__(self, output_dir="/tmp"):
        self.output_dir = output_dir

    def preprocess(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (640, 640))

        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # brightness = np.mean(gray)

        # if brightness < 80:
        #     gamma = 1.5
        # elif brightness > 180:
        #     gamma = 0.7
        # else:
        #     gamma = 1.0

        # if gamma != 1.0:
        #     lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 
        #                     for i in np.arange(256)]).astype("uint8")
        #     image = cv2.LUT(image, lut)

        # lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # l, a, b = cv2.split(lab)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # cl = clahe.apply(l)
        # limg = cv2.merge((cl, a, b))
        # image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # image = cv2.GaussianBlur(image, (3, 3), 0)

        # blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        filename = os.path.basename(image_path).replace(".jpg", "_pre.jpg")
        preprocessed_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(preprocessed_path, image)

        return preprocessed_path #, blur_score