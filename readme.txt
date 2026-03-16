1. Después de descargar el proyecto además de descargarlo por el zip se puede hacer
   git clone https://github.com/Damadia/Box-Gaussian-and-Median-filters.git

2. Una vez que se tenga el proyecto, asegurarse de tener los archivos binarios de python descargados y tener el PATH definido en las variables de entorno del sistema desde
   https://www.python.org/downloads/release/python-3143/ 

3. Hacer un enviroment en la terminal con "python3 -m venv .venv" (verificar donde se está bien la ruta antes de hacerlo)

4. Descargar las dependencias  
    4.1. "pip install opencv-python"
    4.2. "pip install numpy"
    4.3. "pip install scikit-learn"

5. Yo trabajé el proyecto desde VS Code, asegurarse de descargar la extensión oficial de Microsoft de python

6. Si todo fue hecho correctamente y no faltan dependencia, se puede ejecutar cada programa con python file.py (e.g "python BF.py" ejecuta el programa de Box filter)

7. En la función main de cada archivo se define una imagen y es leída con openCV (las imagenes están en la ruta "../imagenes/[nombre_imagen].jpg")

    7.1. En box filter solo hace falta llamar la función boxFilter(img, k), con img siendo la imagen abierta y k el tamaño del kernel (el cual tiene que ser un número impar > 1)

    7.2. En Gaussian filter además de lo anterio, hay que definir un sigma el cual tiene que ser más pequeño en relación a que tan grande es el kernel k. 
         Se tiene que llamar a gaussianFilter(img,k,sigma)
    
    7.3. En Median filter hay que definir la imagen de sal y pimienta y usar esa en vez de la original para ver los efectos de median filter. Llamar a la función medianBlur(img, k)

NOTA: las depencias no están incluidas en el git, por lo que hay que hacer los pasos 3 y 4 antes de poder ejecutar los archivos .py