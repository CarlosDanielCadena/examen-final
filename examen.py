
'''
/////////////////////////////////////////////
//    PONTIFICIA UNIVERSIDAD JAVERIANA     //
//                                         //
//  Carlos Daniel Cadena Cahvarro          //
//                                         //
//  Procesamiento de imagenes y vision     //
//  Examen final                           //
/////////////////////////////////////////////
'''

import cv2
import numpy as np
import sys
import os

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from time import time
from hough import hough
from orientation_estimate import *


class bandera:

    def __init__(self, path):                   # Constructor
        self.image = cv2.imread(path, 1)        # Imagen original
        self.image1 = cv2.imread(path, 1)  # Imagen original
        self.labels = 0
        #cv2.imshow("Original", self.image)     # Impresion de imagen original

    def colores(self):

        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        n_colors = 4
        method = 'kmeans'
        select = 1
        image = np.array(image, dtype=np.float64) / 255
        rows, cols, ch = image.shape
        assert ch == 3
        image_array = np.reshape(image, (rows * cols, ch))
        image_array_sample = shuffle(image_array, random_state=0)[:10000]

        model = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
        self.labels = model.predict(image_array)
        labels = self.labels
        centers = model.cluster_centers_

        colors = np.max(labels)+1
        print(f"El numero de colores de la bandera es de: {colors}")
        #print(labels)
        #return labels

    def porcentaje(self):
        primero = 0
        segundo = 0
        tercero = 0
        cuarto = 0
        for i in self.labels:
            if i == 1:
                primero += i
                porcentaje1 = primero / len(self.labels)
            if i == 2:
                segundo += i
                porcentaje2 = segundo / len(self.labels)
            if i == 3:
                tercero += i
                porcentaje3 = tercero / len(self.labels)
            if i == 0:
                cuarto += i
                porcentaje4 = cuarto / len(self.labels)

        print(f"El porcentaje para el color 1 de la bandera es de: {porcentaje1}")
        print(f"El porcentaje para el color 2 de la bandera es de: {porcentaje1}")
        print(f"El porcentaje para el color 3 de la bandera es de: {porcentaje1}")
        print(f"El porcentaje para el color 4 de la bandera es de: {porcentaje4}")

    def orientacion (self):

        image = self.image1
        high_thresh = 300
        bw_edges = cv2.Canny(image, high_thresh * 0.3, high_thresh, L2gradient=True)

        houghs = hough(bw_edges)
        accumulator = houghs.standard_HT()

        acc_thresh = 50
        N_peaks = 11
        nhood = [25, 9]
        peaks = houghs.find_peaks(accumulator, nhood, acc_thresh, N_peaks)

        [_, cols] = image.shape[:2]
        image_draw = np.copy(image)
        for i in range(len(peaks)):
            rho = peaks[i][0]
            theta_ = houghs.theta[peaks[i][1]]

            theta_pi = np.pi * theta_ / 180
            theta_ = theta_ - 180
            a = np.cos(theta_pi)
            b = np.sin(theta_pi)
            x0 = a * rho + houghs.center_x
            y0 = b * rho + houghs.center_y
            c = -rho
            x1 = int(round(x0 + cols * (-b)))
            y1 = int(round(y0 + cols * a))
            x2 = int(round(x0 - cols * (-b)))
            y2 = int(round(y0 - cols * a))

            if np.abs(theta_) < 80:
                image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 255, 255], thickness=2)
            elif np.abs(theta_) > 100:
                image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [255, 0, 255], thickness=2)
            else:
                if theta_ > 0:
                    image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 255, 0], thickness=2)
                else:
                    image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 0, 255], thickness=2)

        # print(theta_)
        if theta_ == 0:
            print('la bandera tiene lineas mixtas')
        elif theta_ <= 90:
            print('la bandera tiene lineas horizontales')
        elif theta_ > 90:
            print('la bandera tiene lineas veritcales')



