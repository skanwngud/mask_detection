import glob
import cv2
import numpy as np
import os

from file_manager import FileManager

class FaceCrop:
    def __init__(self):
        super(FaceCrop).__init__()
        self.__face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.__eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'

        self.face_cascade = cv2.CascadeClassifier(self.__face_cascade_path)
        self.eye_cascade = cv2.CascadeClassifier(self.__eye_cascade_path)

        self.res_dict = {}

        self.file_manager = FileManager()
        
    def find_face(self, img):
        face = self.face_cascade.detectMultiScale(img, 1.3, 5)
        return face
    
    def find_eye(self, img):
        eye = self.eye_cascade.detectMultiScale(img, 1.5, 5)
        return eye

    def vis_info(self, img, key, face = False, eye = False):
        face_xywh = self.res_dict[key]["face"]
        eye_xywh = self.res_dict[key]["eye"]

        if face:
            for idx in range(len(face_xywh)):
                img = cv2.rectangle(img, (face_xywh[idx][0], face_xywh[idx][1]), (face_xywh[idx][2], face_xywh[idx][3]), (0, 0, 255), 2)
        if eye:
            for idx in range(len(eye_xywh)):
                img = cv2.rectangle(img, (eye_xywh[idx][0], eye_xywh[idx][1]), (eye_xywh[idx][2], eye_xywh[idx][3]), (255, 0, 0), 2)
        return img

    def crop_face(self, path, save, visualize):
        img_list = self.file_manager.get_images_list(path + "*.jpg")

        for img in img_list:
            current_img = img.split('/')[-1]
            self.res_dict[current_img] = {}
            image = self.file_manager.load_img(path + img)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = self.find_face(image_gray)
            eyes = self.find_eye(image_gray)

            self.res_dict[current_img]["face"] = faces
            self.res_dict[current_img]["eye"] = eyes

            if len(faces) != 0 and save:
                for (x, y, w, h) in faces:
                    cropped = image[x:x+w, y:y+h]
                    cv2.imwrite(path + img.split('.')[0] + "_crop.jpg", cropped)

            if visualize:
                vis_img = self.vis_info(image, current_img, True, True)
                cv2.imwrite(path + img.split('.')[0] + "_vis.jpg", vis_img)

        return self.res_dict

if __name__ == "__main__":
    face_crop = FaceCrop()
    print(face_crop.crop_face("./", True, True))

