import cv2
import glob

import numpy as np
from PIL import Image

input_path = " "
output_path = " "


for img in glob.glob(input_path):

    subpath = img.split("/")[-1]
    img_name = subpath.split(".")[0]

    originalImage = cv2.imread(img)
    img = np.array(Image.open(img))


    flipVertical = cv2.flip(originalImage, 0)
    flipHorizontal = cv2.flip(originalImage, 1)
    flipBoth = cv2.flip(originalImage, -1)

    cv2.imwrite(output_path + img_name + "_1.jpg",flipVertical)
    cv2.imwrite(output_path + img_name + "_2.jpg", flipHorizontal)
    cv2.imwrite(output_path + img_name + "_3.jpg",flipBoth)



    #  rotate 90
    Image.fromarray(np.rot90(img)).save(output_path + img_name + "_4.jpg")
    # rotate 180
    Image.fromarray(np.rot90(img, 2)).save(output_path + img_name + "_5.jpg")
    # rotate 270
    Image.fromarray(np.rot90(img, 3)).save(output_path + img_name + "_6.jpg")

