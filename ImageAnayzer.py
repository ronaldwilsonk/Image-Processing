import os
import numpy as np
from scipy.stats import linregress
import cv2


class ImageAnalyzer:
    def __init__(self, target_image, face_detector_path, eye_detector_path, display_processed_image=False):
        """
        Program to pre-process images in the Morph Dataset. Images corrected for size, contrast and
        rotation.
        Author: Ronald Wilson
        :param target_image: Image to be corrected, array like
        :param face_detector_path: Absolute path to the Cascaded Haar Detector for face detection
        :param eye_detector_path: Absolute path to the Cascaded Haar Detector for eye(both) detection
        :param display_processed_image: Flag to display processed image.
                True: Display Image, False: Do not display image
                Set to true on single image and False for batch processing
        """
        self.cascade_face_detector_location = face_detector_path
        self.cascade_eye_detector_location = eye_detector_path
        # Provides the quality of the image post processing
        # Quality Code 0: Image cannot be processed
        # Quality Code 0.5: Only face was localized. Failed to detect eyes for image rotation
        # Quality Code 1: Rotation succeeded. This code also includes images that doesn't need to be rotated
        self.quality_code = 0

        self.color_image = cv2.resize(target_image, (400, 480))
        self.grey_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)
        self.processed_image = None

        # Contrast equalization using histogram equalization
        self.grey_scale_intensity_histogram = self._get_intensity_histogram()
        self.grey_image = self._histogram_equalization()

        # Calculating the best angle for rotating the image
        face, slope, self.quality_code = self._get_rotation_angle(self.grey_image)
        if slope != np.inf:
            if self.quality_code == 0.5 or self.quality_code == 1:
                self._rotate_image(rotation_angle=np.rad2deg(np.arctan(slope)))
                self._extract_face(face)
                # Code to display image if required
                if display_processed_image is True:
                    cv2.imshow('Processed Image', self.processed_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        else:
            # Save failed images
            self.processed_image = self.color_image
            self.quality_code = 0

    def _get_intensity_histogram(self):
        """
        Obtain the intensity level histogram with bin size = 1
        :return: intensity vector, array-like
        """
        rows, cols = np.shape(self.grey_image)
        intensity_space = np.zeros(256)
        for i in range(0, rows):
            for j in range(0, cols):
                intensity_space[self.grey_image[i, j]] += 1
        return intensity_space

    def _histogram_equalization(self):
        """
        Apply Histogram Equalization on image
        :return: Histogram equalized image, array-like. Same dimension as source image
        """
        norm_cum_sum = np.cumsum(np.divide(self.grey_scale_intensity_histogram, sum(self.grey_scale_intensity_histogram)))
        equalized_intensity_vector = np.floor(np.multiply(norm_cum_sum, 255))
        row, col = np.shape(self.grey_image)
        histogram_equalized_image = np.zeros((row, col), dtype=np.uint8)
        for i in range(0, row):
            for j in range(0, col):
                histogram_equalized_image[i, j] = equalized_intensity_vector[self.grey_image[i, j]]
        return histogram_equalized_image

    def _get_rotation_angle(self, grey_image):
        """
        Obtain the best angle for rotating image using slope of the line connecting both eyes
        :param grey_image: Grayscale image
        :return: best_face_detect, best_slope, exec_code
                 best_face_detect: Coordinates of the detected face
                 best_slope: Slope between the eyes
                 exec_code: Quality code for the execution [0, 0.5, 1]
        """
        exec_code = 0
        rows, cols = np.shape(grey_image)
        possible_angles = np.arange(-45, 45, 0.5)
        best_face_detect, best_slope = None, np.inf
        for angle in possible_angles:
            rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            rotated_grey_image = cv2.warpAffine(grey_image, rotation_matrix, (cols, rows))
            flag, face, slope = self._init_viola_jones(rotated_grey_image)
            if flag == 0.5 and max([exec_code, 0.5]) == 0.5:
                best_face_detect = face
            if flag == 1 and min([best_slope, abs(slope)]) == slope and slope != 0:
                best_face_detect = face
                best_slope = slope
                exec_code = 1
        return best_face_detect, best_slope, exec_code

    def _init_viola_jones(self, rotated_grey_image):
        """
        Apply Viola-Jones to detect face and eyes in the image
        :param rotated_grey_image:
        :return: flag, face, slope
                 flag: 0: for no face detected
                       0.5: face detected
                       1: face and eyes detected
                 face: coordinates of detected face
                 slope: slope between eyes if eyes found else 0(default)
        """
        flag = 0
        face_cascade = cv2.CascadeClassifier(self.cascade_face_detector_location)
        eye_cascade = cv2.CascadeClassifier(self.cascade_eye_detector_location)
        face, eyes, slope = face_cascade.detectMultiScale(rotated_grey_image), [], 0
        if len(face) == 1:
            flag = 0.5
            [[x, y, w, h]] = face
            roi_gray = rotated_grey_image[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) == 2:
                x, y = [], []
                for (ex, ey, ew, eh) in eyes:
                    x.append(np.int(ex + ew / 2))
                    y.append(np.int(ey + eh / 2))
                slope, _, _, _, _ = linregress(x, y)
                if np.isnan(slope):
                    slope = 0
                    flag = 0.5
                else:
                    flag = 1
        return flag, face, slope

    def _rotate_image(self, rotation_angle):
        """
        Rotate image at a given angle
        :param rotation_angle: Angle to rotate at, double
        :return: None
        """
        rows, cols, ch = np.shape(self.color_image)
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
        self.grey_image = cv2.warpAffine(self.grey_image, rotation_matrix, (cols, rows))
        for i in range(0, ch):
            self.color_image[:, :, i] = cv2.warpAffine(self.color_image[:, :, i], rotation_matrix, (cols, rows))

    def _extract_face(self, face_coordinates):
        """
        Extract the face from the image
        :param face_coordinates: Coordinates of the face
        :return: None
        """
        [[x, y, w, h]] = face_coordinates
        self.processed_image = cv2.resize(self.color_image[y:y + h, x:x + w, :], (200, 240))

    def get_processed_image(self):
        """
        Return the processed image and quality code
        :return: processed_image, quality_code
                 processed_image: Post-processed image, array-like dim(200, 240)
                 quality_code: Quality of processing performed.
                 Quality Code 0: Image cannot be processed
                 Quality Code 0.5: Only face was localized. Failed to detect eyes for image rotation
                 Quality Code 1: Rotation succeeded. This code also includes images that doesn't need to be rotated
        """
        return self.processed_image, self.quality_code


if __name__ == "__main__":
    # Path to dataset
    root = 'D:/Test/'

    # This is typically found in ...\Users\XXX\AppData\Local\Programs\Python\PythonXX\Lib\site-packages\cv2\data
    # if opencv_python is installed. To install: pip install opencv_python
    face_detector = 'D:/haarcascade_frontalface_default.xml'
    eye_detector = 'D:/haarcascade_eye.xml'

    quality_0_output_path = root + "Failed Images/"
    os.mkdir(quality_0_output_path)
    quality_5_output_path = root + "Unrotated Images/"
    os.mkdir(quality_5_output_path)
    quality_10_output_path = root + "Rotated Images/"
    os.mkdir(quality_10_output_path)

    file_list = os.listdir(root)
    for idx, file in enumerate(file_list):
        print("File " + str(idx + 1) + " of " + str(len(file_list)))
        print("File Name : ", file)
        im = cv2.imread(root + file)
        IA = ImageAnalyzer(im, face_detector, eye_detector)
        processed_image, quality_code = IA.get_processed_image()
        if quality_code == 0:
            cv2.imwrite(quality_0_output_path + file, processed_image)
            print("Status: Failed\n")
        elif quality_code == 0.5:
            cv2.imwrite(quality_5_output_path + file, processed_image)
            print("Status: Unrotated\n")
        else:
            cv2.imwrite(quality_10_output_path + file, processed_image)
            print("Status: Rotated\n")
