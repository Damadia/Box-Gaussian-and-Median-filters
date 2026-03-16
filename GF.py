# Gaussian Filter

import cv2
import numpy as np
import math 
from skimage.util import random_noise


def gaussianBlur(neighborhood, k, GK, kSum, sigma = 1): # neighborhood = matriz[k,k], k = entero positivo    
    r = 0 
    g = 0 
    b = 0 

    
    for i in range(k):
        for j in range(k):
            b += (neighborhood[i][j][0] * GK[i][j])
            g += (neighborhood[i][j][1] * GK[i][j]) 
            r += (neighborhood[i][j][2] * GK[i][j])


    return [b/kSum,g/kSum,r/kSum] #Los tres canales

def gaussianKernel(k, sigma = 1):
    #Formula de la matriz gausiana, también puede ser definida por un arreglo 1D y multiplicada por su transpuesta.
    #Los valores también ser truncados de acuerdo a una cosntante para aproximar los valores como la pirade de pascal 
    #para dar una matriz con enteros (mi constante será 32)

    #kernel = np.full((k,k), -1, dtype = np.float64)
    kSum = 0

    #Calcular en la matriz 1D
    kernel1D = np.full(k, -1, dtype = np.float64)

    kernel2D = np.zeros((k,k), dtype = np.uint16)

    
    # (1/(2*np.pi*np.power(sigma,2))) este factor no es necesario incluirlo porque se cancela en la normalización 
    for i in range(k):
        disX = i - math.floor(k/2)
        kernel1D[i] = np.exp(-1*(np.power(disX,2) / 2*np.power(sigma,2)))
        kernel1D[i] = np.round((kernel1D[i] * 32))#Para aproximar a valores enteros más fáciles de calcular
    

    #Si con la matriz K hacemos K^T * K nos da una matriz 2D que es el kernel de convulsión gaussiano
    kernel2D = np.outer(kernel1D, kernel1D)
    kSum = np.sum(kernel2D)

    print(kernel2D)
    return kernel2D, kSum




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


def gaussianFilter(imgOg, k, sigma = 1):
    #La imagen tiene que ser estritamente definida en RGB, de lo contrario aquí dará error porque una imagen en escala de grises no tiene channels
    imgGF = imgOg.copy()
    h, w, c = imgGF.shape # ancho, largo y canales  


    neighborhood = np.full((k,k,3), -1) #Matriz llena de -1 del tamaño del kernel, setteo todo en -1 para usar el edgeReplication en caso haga falta
    GK, kSum = gaussianKernel(k, sigma)

    for y in range(h):
        for x in range(w):
            neighborhood = np.full((k,k,3), -1)
            if any(checkForER(x,y,k, imgOg.shape)): #Al menos una bandera está en 1 (hay desbordamiento)
                neighborhood = edgeReplication(imgOg, x, y, k)
            else:
                neighborhood = createNeighborhood(imgOg, x, y, k)
            channelsBlur = gaussianBlur(neighborhood, k, GK, kSum, sigma)
            imgGF[y][x][0] = channelsBlur[0] #b
            imgGF[y][x][1] = channelsBlur[1] #g
            imgGF[y][x][2] = channelsBlur[2] #r

    
    return imgGF



def main():
    perlas = cv2.imread('./imagenes/joven_perlas.jpg')
    cv2.imshow("Imagen original", perlas)
    perlasGF = perlas.copy()
    
    perlasGF = gaussianFilter(perlasGF,5,0.75) #El valor de sigma tiene que ser menor cuanto mayor es el kernel, de lo contrario la matriz serán muchos ceros y valore similares
                                            # Algunos valores que me funcionaron:
                                            # k = 3 y sigma = 1; k = 5 y sigma = 0.75; k = 7 y sigma = 0.6;
                                            # En la salida de la consola se puede ver cuando fue una buena aproximación
                                            # de acuerdo a como están distribuidos los valores    

    perlasGFCV = perlas.copy()

    perlasGFCV = cv2.GaussianBlur(perlasGFCV,(5,5), 0)



    
    perlasGray = cv2.cvtColor(perlas, cv2.COLOR_BGR2GRAY)

    #cv2.imshow("RG gray", perlasGray)

    cv2.imshow("Mi implementeación de Gaussian Filter", perlasGF)

    cv2.imshow("Implementeación de Gaussian Filter de openCV", perlasGFCV)




    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()