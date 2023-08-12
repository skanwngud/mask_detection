from glob import glob

import cv2
import numpy as np
import random

class MakeDataset:
    def __init__(self, save=""):
        self.save = save
        self.save_cnt = 0

    # Brightness
    def __chg_brightness(self, image=None, type=None, save=True):
        """
        :param image: original image
        :param type: "b" is brighter, "d" is darker
        :return: changed image
        """
        if image is None:
            return

        assert type in ["b", "d", None], "please choose 'b' or 'd'"
        brightness = tuple([np.random.randint(0, 128)] * 3)
        temp_array = np.full(image.shape, brightness, dtype=np.uint8)

        if type == "b":
            chg_img = cv2.add(image, temp_array)
        elif type == "d":
            chg_img = cv2.subtract(image, temp_array)
        elif type is None:
            chg_img = self.__chg_brightness(image, random.choice(["b", "d"]))

        if save:
            cv2.imwrite(self.save + f"{self.save_cnt}.jpg", chg_img)
            self.save_cnt += 1

        return chg_img

    # Flip horizontal, vertical
    def __flip_hroizontal(self, image=None, type=None, save=True):
        """
        :param image: original image
        :param type: "h" is horizontal, "v" is vertical
        :return: changed image
        """
        if image is None:
            return

        assert type in ["h", "v", None], "please input 'h' or 'v'"

        if type == "h":
            flip_img = cv2.flip(image, 1)
        elif type == "v":
            flip_img = cv2.flip(image, 0)
        elif type is None:
            ran = random.randint(0, 2)
            flip_img = cv2.flip(image, ran)

        if save:
            cv2.imwrite(self.save + f"{self.save_cnt}.jpg", flip_img)
            self.save_cnt += 1

        return flip_img

    # Translation
    def __translation(self, image=None, x=0, y=0, save=True):
        """

        :param image: original image
        :param x: translate to x
        :param y: translate to y
        :return: changed image
        """
        if image is None:
            return

        if x == 0 and y == 0:
            x = random.randint(0, image.shape[0] // 2)
            y = random.randint(0, image.shape[1] // 2)

        rows, cols = image.shape[:2]
        M = np.float64([[1, 0, x], [0, 1, y]])

        translation_img = cv2.warpAffine(image, M, (cols, rows))

        if save:
            cv2.imwrite(self.save + f"{self.save_cnt}.jpg", translation_img)
            self.save_cnt += 1

        return translation_img

    # main run
    def main(self, images):
        making_dict = {
            "brightness": self.__chg_brightness,
            "flip": self.__flip_hroizontal,
            "translation": self.__translation
        }

        for image in images:
            image = cv2.imread(image)

            type = random.choice(["brightness", "flip", "translation"])
            print(type)

            making_func = making_dict[type]
            making_func(image)


if __name__ == "__main__":
    imgs = glob('../*.jpg')

    md = MakeDataset("./")
    md.main(imgs)
