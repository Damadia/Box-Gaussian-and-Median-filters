# Median filter

import cv2
import numpy as np
import math
from skimage.util import random_noise



#Nota de Svein:
#Esta función le dije a la IA que la generará, pero me tomé la molestía de documentarla lo que hacen las líneas
#importantes porque la sintaxis de Python me confude un poco al darse tantas libertades.
def add_sp_noise(image, prob = 0.05):
    output = np.copy(image)

    # Salt mode
    num_salt = np.ceil(prob * image.size * 0.5) #El .5 es porque la mitad es sal
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in  image.shape[:2]] #np.random.randint = genera en el intervalo [0, i-1) una cantidad de num_salt de números aleatorios
    output[coords[0], coords[1], :] = 255 #Esto dice, para todas las coordenadas (x,y) de coords (las cuales son pares ordenados), pon la intensidad en 255 = blanco
    # cords luce algo así [[0,1,2], [4,5,6]] donde los dos arreglos siempre son del mismo tamaño. Python tiene la particularidad de que si le mandas
    # dos arreglos del mismo tamaño, empata por índice cada uno de los elementos de los dos arreglos como un par ordenado

    # Pepper mode
    num_pepper = np.ceil(prob * image.size * 0.5) # Y la otra mitad es pimienta
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]] #Este for crea dos arreglos porque image.shape = ancho, largo, excluimos el último elemento porque son los canales
    output[coords[0], coords[1], :] = 0 #Similarmente, para todas las coordenadas (x,y) de coords, pon la intensidad en 0 = negro
    return output




def medianBlur(neighborhood, k): # neighborhood = matriz[k,k], k = entero positivo    
    r = 0 
    g = 0 
    b = 0 

    medianArr = np.zeros((k*k, 3), dtype=np.uint8) #En un mismo arreglo, definir los elementos como arreglos de 3 elementos para los tres canales
    # i = index // k correspondiente fila
    # j = index % k correspondiente columna

    for i in range(k*k):
        row = i // k
        col = i % k
        medianArr[i] = neighborhood[row][col]
        #insertion sort y para los tres canales
        for channelPos in range(3):
            if (i > 0):
                j = i
                while j > 0 and medianArr[j][channelPos] < medianArr[j-1][channelPos]:
                    swap(channelPos, medianArr[j], medianArr[j-1]) #Cambio de índices pero, dos índices de un solo arreglo, sino el mismo índice de dos arreglos distintos
                    j-=1


    #print(medianArr)
    mid = k*k//2
    b = medianArr[mid][0]
    g = medianArr[mid][1]
    r = medianArr[mid][2]
    return [b,g,r] #Los tres canales

def swap(channel, arr1, arr2):
    temp = arr1[channel]
    arr1[channel] = arr2[channel]
    arr2[channel] = temp


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


def medianFilter(imgOg, k):
    #La imagen tiene que ser estritamente definida en RGB, de lo contrario aquí dará error porque una imagen en escala de grises no tiene channels
    imgGF = imgOg.copy()
    h, w, c = imgGF.shape # ancho, largo y canales  


    neighborhood = np.full((k,k,3), -1) #Matriz llena de -1 del tamaño del kernel, setteo todo en -1 para usar el edgeReplication en caso haga falta

    for y in range(h):
        for x in range(w):
            neighborhood = np.full((k,k,3), -1)
            if any(checkForER(x,y,k, imgOg.shape)): #Al menos una bandera está en 1 (hay desbordamiento)
                neighborhood = edgeReplication(imgOg, x, y, k)
            else:
                neighborhood = createNeighborhood(imgOg, x, y, k)
            channelsBlur = medianBlur(neighborhood, k)
            imgGF[y][x][0] = channelsBlur[0] #b
            imgGF[y][x][1] = channelsBlur[1] #g
            imgGF[y][x][2] = channelsBlur[2] #r

    
    return imgGF



def main():
    perrito = cv2.imread('./imagenes/perrito_pelon.jpg')
    cv2.imshow("Imagen original", perrito)

    perritoSP = perrito.copy()
    perritoSP = add_sp_noise(perritoSP) 
    cv2.imshow("Imagen con Sal y Pimienta", perritoSP)

    
    perritoMF = perrito.copy()
    

    perritoMF = medianFilter(perritoSP,3) 
    perritoMFCV = perrito.copy()

    perritoMFCV = cv2.medianBlur(perritoSP,3) #Implementación de openCV para ver la comparasión



    
    perritoGray = cv2.cvtColor(perrito, cv2.COLOR_BGR2GRAY)

    #cv2.imshow("RG gray", perritoGray)

    cv2.imshow("Mi implementeación de Median Filter", perritoMF)

    cv2.imshow("Implementeación de Median Filter de openCV", perritoMFCV)
    



    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()
