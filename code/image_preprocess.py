import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

import csv

# read image
all_path = ' '

#  use kmeans to do cluster
#  k =2 the high pixel value represent the damage and white lines
# the low value represent the blue background
def kmeans(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = img.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    k = 2
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # convert back to 8 bit values
    centers = np.uint8(centers)
    labels = labels.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(img.shape)

    return(segmented_image)

# for gray image to calculate the most common pixel value
#  if the max_index > 50, then level = 6
def hist_calculate(img_gray):

    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    # count the most common value portion
    number_pixel = []
    for h in hist:
        number_pixel.append(int(h))
    max_index = number_pixel.index(max(number_pixel))
    return(max_index)

def get_rectangle_points(closing):
    cont = []
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if (cv2.contourArea(cnt) > 1200):
            rect = cv2.minAreaRect(cnt)
            angle = abs(rect[2])
            if (angle < 5) or (angle > 85):
                box = cv2.boxPoints(rect)
                # print(box[0][0], box[0][1])
                # cv2.line(segmented_image, (241, 0), (241, segmented_image_gray.shape[0] -1), (255, 0, 0), 5)
                # roi = img_gray[0:segmented_image_gray.shape[0] - 1,235: 241]
                # (mean, stddev) = cv2.meanStdDev(roi)
                box = np.int0(box)
                cv2.drawContours(segmented_image, [box], 0, (0, 0, 255), 2)
                cont.append(cnt)
                # print(angle)
    # print(len(cnt))
    cv2.imwrite('', segmented_image)
    return(cont)



def white_line_points(img_gray,cont):

    x_list = []
    y_list = []
    for c in cont:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        for point in box:
            x_list.append(int(point[0]))
            y_list.append(int(point[1]))
    #  remove the same element
    x_list = sorted(list(set(x_list)))
    y_list = sorted(list(set(y_list)))

    x_list_remove_similar = []
    height, width  = img_gray.shape
    for x in x_list:
        if x < (width -1) and (x > 0 ):
            if x_list_remove_similar ==  [] :
                x_list_remove_similar.append(x)
                new = x
            else:
                if x in range(new-10, new + 10):
                    pass
                else:
                    x_list_remove_similar.append(x)
                    new = x

    # print(x_list_remove_similar)

    y_list_remove_similar = []

    for y in y_list:
        if y < (height - 1) and (y > 0 ):
            if y_list_remove_similar == []:
                y_list_remove_similar.append(y)
                new = y
            else:
                if y in range(new - 10, new + 10):
                    pass
                else:
                    y_list_remove_similar.append(y)
                    new = y


    return(x_list_remove_similar, y_list_remove_similar)


def find_white_lines(x_list, y_list, img,closing):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (img_mean, stddev) = cv2.meanStdDev(img_gray)

    width_left = 0
    width_right = 0
    mean_max_right = 0
    mean_max_left = 0
    image = img.copy()


    for x in x_list:
        for i in range(1,11):
            if (x - i) <= 0 or x >= (img_gray.shape[1] -1):
                pass
            else:
                roi = img_gray[0:img_gray.shape[0]-1,x - i :x]
                (mean, stddev) = cv2.meanStdDev(roi)
                if mean > mean_max_left :
                    mean_max_left = mean
                    width_left = i

        for i in range(1, 11):
            if (x + i) >= img_gray.shape[1] -1:
                pass
            else:
                roi = img_gray[0:img_gray.shape[0]-1, x :x + i ]
                (mean, stddev) = cv2.meanStdDev(roi)
                if mean > mean_max_right:
                    mean_max_right = mean
                    width_right = i

        roi = img_gray[ 0:img_gray.shape[0]-1, x - width_left:x + width_right]
        (mean, stddev) = cv2.meanStdDev(roi)
        roi_mean = mean
        if (x - width_left) < 290 and (x - width_left) != 170:
            cv2.rectangle(image, (x - width_left + 10 , 0), (x + width_right+10, img_gray.shape[0]), (0, 0, 255), 3)

            if ((roi_mean > img_mean) and stddev < 60):
                for i in range(x - width_left , x + width_right):
                    for j in range(0,img_gray.shape[0] -1):
                        if i >= img_gray.shape[1]:
                            pass
                        else:
                            closing[j ,i+10] = 0


    height_up = 0
    height_down = 0
    mean_max_up = 0
    mean_max_down = 0
    for y in y_list:
        for i in range(1, 11):
            if ((y - i) <= 0 ) or (y >=(img_gray.shape[0] -1)):
                pass
            else:
                roi = img_gray[y-i:y, 0:img_gray.shape[1]-1 ]
                (mean, stddev) = cv2.meanStdDev(roi)
                if mean > mean_max_up:
                    mean_max_up = mean
                    height_up = i
        for i in range(1, 11):
            if (y + i) >= img_gray.shape[0] -1:
                pass
            else:
                roi = img_gray[ y:y+i, 0:img_gray.shape[1]-1]
                (mean, stddev) = cv2.meanStdDev(roi)
                if mean > mean_max_down:
                    mean_max_down = mean
                    height_down = i
        print(y - height_up,y + height_down)
        roi = img_gray[y - height_up:y + height_down, 0:img_gray.shape[1]-1]
        (mean, stddev) = cv2.meanStdDev(roi)
        roi_mean = mean
        if (y - height_up) < 290 and (y - height_up) != 41:
            cv2.rectangle(image, (0, y - height_up), (img_gray.shape[1]-1,y + height_down), (0, 0, 255), 3)
            if ((roi_mean > img_mean) and stddev < 60):
                for i in range(0,img_gray.shape[1] -1):
                    for j in range(y - height_up , y + height_up):
                        if j >= img_gray.shape[0]:
                            pass
                        else:
                            closing[j,i] = 0
    return(closing)



for path in glob.glob(all_path + '*.jpg'):
    # initialize the level = 0
    level = 0

    img_name = path.split("/") [-1]
    # print(img_name)
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # calculate the hist value
    max_pixel = hist_calculate(img_gray)
    if max_pixel >= 50:
        level = 6

    #  Kmeans cluster
    segmented_image = kmeans(img)
    segmented_image_gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('', segmented_image_gray)
    blur = cv2.GaussianBlur(segmented_image_gray, (5, 5), 0)
    cv2.imwrite('',blur)
    ret, thresh = cv2.threshold(segmented_image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((1, 1), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cv2.imwrite('',closing)
    reverse = cv2.bitwise_not(closing)

    cont = get_rectangle_points(reverse)

    x_list, y_list = white_line_points(img_gray, cont)
    black_white = find_white_lines(x_list, y_list, img.copy(),closing)
    height, width = black_white.shape
    white_pixel = 0
    black_pixel = 0
    all = 0
    for i in range(height):
        for j in range(width):
            if black_white[i, j] == 255:
                white_pixel =  white_pixel + 1
            if black_white[i, j] == 0:
                black_pixel = black_pixel + 1
            all = all + 1


    white_ratio = white_pixel / all


























