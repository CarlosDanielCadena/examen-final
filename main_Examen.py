from examen import *         # Importar

if __name__ == "__main__":       # Main
    path = input("Por favor ingrese la ruta de la imagen: ")    # Ingreso de la ruta de la imagen
    imagen = bandera(path)
    imagen.colores()
    imagen.porcentaje()
    imagen.orientacion()    
