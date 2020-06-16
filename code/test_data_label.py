import cv2
import glob
import csv

import numpy as np
from PIL import Image



for i in range(0,6):

    input_path = ' '

    for img in glob.glob(input_path):

        subpath = img.split("/")[-1]

        with open(' ', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([subpath, str(i)])
        file.close()




