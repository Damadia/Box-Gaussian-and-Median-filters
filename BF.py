# Box Filter

import cv2
import numpy as np
import math 
from skimage.util import random_noise


def filterBlur(neighborhood, k): # neighborhood = matriz[k,k], k = entero positivo
    r = 0 
    g = 0 
    b = 0 
    
    for i in range(k):
        for j in range(k):
            b += neighborhood[i][j][0]
            g += neighborhood[i][j][1]
            r += neighborhood[i][j][2]


    return [b/(k*k), g/(k*k),r/(k*k)] #Los tres canales

def checkForER(x,y,k, size):
    h, w, c = size
    flags = np.zeros(4, dtype=np.uint8)
    if (x - (math.floor(k/2)) < 0): #Desborde en lo ancho izq
        flags[0] = 1
    if (x + (math.floor(k/2)) > w - 1): #Desborde en lo ancho der
        flags[1] = 1
        
    if (y - (math.floor(k/2)) < 0): #Desborde en lo alto arriba
        flags[2] = 1
    if (y + (math.floor(k/2)) > h - 1): #Desborde en lo alto abajo
        flags[3] = 1

    return flags 


def edgeReplication(img, x, y, k): #Para los bordes (openCV usa mirror border y quería probar con otro método)
    nh = np.full((k,k,3), -1)
    for i in range(k):
        for j in range(k):
            nh[i,j] = img[np.clip(y + i - (math.floor(k/2)), 0 , img.shape[0] - 1)  , np.clip(x + j - (math.floor(k/2)), 0 , img.shape[1] - 1) ] #numpy.clip es equivalente a la función clamp 
    return nh

def createNeighborhood(img,x, y, k):
    nh = np.full((k,k,3),-1) #matiz kxk que almacena arreglos de tamaño 3 por los tres canales 
    for i in range(k):
        for j in range(k):
            nh[i,j] = img[y+i - (math.floor(k/2)), x + j - (math.floor(k/2))]
    return nh


def boxFilter(imgOg, k):
    #La imagen tiene que ser estritamente definida en RGB, de lo contrario aquí dará error porque una imagen en escala de grises no tiene channels
    imgBF = imgOg.copy()
    h, w, c = imgBF.shape # ancho, largo y canales  


    neighborhood = np.full((k,k,3), -1) #Matriz llena de -1 del tamaño del kernel, setteo todo en -1 para usar el edgeReplication en caso haga falta

    for y in range(h):
        for x in range(w):
            neighborhood = np.full((k,k,3), -1)
            if any(checkForER(x,y,k, imgOg.shape)): #Al menos una bandera está en 1 (hay desbordamiento)
                neighborhood = edgeReplication(imgOg, x, y, k)
            else:
                neighborhood = createNeighborhood(imgOg, x, y, k)
            channelsBlur = filterBlur(neighborhood, k)
            imgBF[y][x][0] = channelsBlur[0] #b
            imgBF[y][x][1] = channelsBlur[1] #g
            imgBF[y][x][2] = channelsBlur[2] #r

    
    return imgBF



def main():
    rgImg = cv2.imread('./imagenes/drive-gosling-hallwayl.jpg')
    cv2.imshow("Imagen original", rgImg)
    rgImgBF = rgImg.copy()
    
    rgImgBF = boxFilter(rgImgBF, 3)
    rgImgBFCV = rgImg.copy()

    rgImgBFCV = cv2.boxFilter(rgImgBFCV, -1, (3,3), normalize=True)



    
    rgImgGray = cv2.cvtColor(rgImg, cv2.COLOR_BGR2GRAY)

    #cv2.imshow("RG gray", rgImgGray)
    cv2.imshow("Mi implementeación de Box Filter", rgImgBF)

    cv2.imshow("Implementeación de Box Filter de openCV", rgImgBF)



    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()