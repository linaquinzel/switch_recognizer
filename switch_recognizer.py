import cv2
import numpy as np

def find_contours_of_switchers(image_path):
    """ Находим контуры. """
    # получаем и читаем картинку
    image = cv2.imread(image_path)
    # делаем изображение серым
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([22, 93, 0], dtype="uint8")
    upper = np.array([45, 255, 255], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)
    b, g, r = cv2.split(image)
    dst = cv2.addWeighted(r, 0.5, g, 0.5, 0.0)
    cv2.imshow("window_name", mask)
    # блюрим изображение (картинка, (размер ядра(матрицы), стандартное отклонение ядра))
    blurred = cv2.GaussianBlur(mask, (3, 3), 0)
    # преобразуем картинку в чб, всем значениям >127 присваиваем 255, остальным-0
    # threshold возвращает два значения - второй переданный в функцию аргумент и картинку
    T, thresh_img = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    print(thresh_img)
    # находим контуры интересных точек
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (255,0,0), 3, cv2.LINE_AA, hierarchy, 1)
    cv2.imshow('contours', image)
    cv2.waitKey(0) 
    #closing all open windows 
    cv2.destroyAllWindows() 
    return contours, gray_image

def find_coordinates_of_switchers(contours, gray_image):
    """ Находим координаты. """
    # словарь вида {выключатель: значение}
    switchers_coordinates = {}
    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if w > 20 and h > 30:
            img_crop = gray_image[y - 15:y + h + 15, x - 15:x + w + 15]
            cards_name = cv2.find_features(img_crop)
            switchers_coordinates[cards_name] = (x - 15, 
                     y - 15, x + w + 15, y + h + 15)
    return switchers_coordinates

find_contours_of_switchers("switcher.jpg")