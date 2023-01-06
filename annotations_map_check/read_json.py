import json
import matplotlib.pyplot as plt
import numpy as np

'''
Lectura del json con los datos de los mapas para generar visualizaciones
'''


def read_json():

    # Open a json file for reading
    jsonFile = 'Resultados2.json'

    with open(jsonFile) as f:
        data = json.load(f)

    datos = data['video'][0:]
    print(len(datos))
    i, a = 0, 0
    frame, conteo_anotaciones, conteo_mapa, precision, head_frame, euclidean_d = [], [], [], [], [], []
    error, b, errorsi, average = [], [], [], []
    promedio = 0
    w = 0
    new, new1, new2 = [], [], []

    for i in range(218952):
        frame.append(datos[i]['frame'])
        conteo_anotaciones.append(datos[i]['h_xml'])
        conteo_mapa.append(datos[i]['h_map'])
        euclidean_d.append(datos[i]['d_prom'])
        if head_frame[i] == 1:
            error.append(abs(conteo_anotaciones[i] - conteo_mapa[i]))
            if error[a] != 0:
                new.append(frame[i])
                new1.append(conteo_anotaciones[i])
                new2.append(conteo_mapa[i])
            a += 1
        promedio = promedio + euclidean_d[i]
        w = w + 1
        if ((frame[i] == 1801 or frame[i] == 6801 or frame[i] == 26801 or frame[i] == 12801
             or frame[i] == 5801 or frame[i] == 16801 or frame[i] == 9801 or frame[i] == 200) and w > 1800):
            average.append(promedio / w)
            w = 0
            promedio = 0

    print(len(error))
    for i in range(len(error)):
        if error[i] != 0:
            b.append(error[i])
            errorsi.append(i)

    print(errorsi)

    print(average)
    print(new)
    print(new1)
    print(new2)

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    plt.stem(x, average)
    plt.xlim(0, 20)
    plt.xticks(np.arange(0, 20, 1))
    plt.ylim(1, 11)
    plt.xlabel('Número de clip utilizado')
    plt.ylabel('Distancia euclidiana promedio cabezas (pixeles)')
    plt.title('Distancia euclidiana promedio por video')
    plt.grid()

    plt.show()


if __name__ == '__main__':
    read_json()


