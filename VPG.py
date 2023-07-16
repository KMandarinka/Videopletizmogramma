import os #В данном коде используется для работы с файловой системой, для чтения списка файлов в указанной директории
import cv2 #Библиотека компьютерного зрения, которая предоставляет множество функций для обработки изображений и видео, используется для загрузки видео, преобразования кадров в оттенки серого, детекции лиц на кадрах и извлечения регионов интереса (ROI) для анализа интенсивности пикселей
import numpy as np #Библиотека для работы с многомерными массивами данных, используется для сохранения и обработки временных изменений интенсивности, а также для создания массива времени и округления значений амплитуды сигнала.
import matplotlib.pyplot as plt #Библиотека для визуализации данных на графиках, используется для построения графика временных изменений интенсивности и отображения сигнала на графике


# Функция для извлечения ВПГ из видеоизображения
def extract_vpg(video_path):
    # Загрузка видео
    cap = cv2.VideoCapture(video_path) #открывает видеофайл для чтения

    # Инициализация массива для сохранения временных изменений интенсивности
    intensity_values = []

    # Чтение видеокадров до конца видео
    while(cap.isOpened()): #выполняется, пока видеофайл открыт и доступен для чтения
        ret, frame = cap.read() #значение ret указывает на успешность операции чтения, а frame содержит считанный кадр
        if not ret:
            break

        # Преобразование кадра в оттенки серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # frame - исходный кадр в цветном формате, а cv2.COLOR_BGR2GRAY указывает на преобразование из цветного BGR формата в оттенки серого

        # Детекция лица
        # в некоторых версиях необходимо использовать    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml') вместо строки 25
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #создает объект face_cascade, который содержит классификатор Хаара для детекции лиц
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(150, 150)) # использует классификатор лиц для обнаружения лиц на кадре в оттенках серого. Она возвращает координаты и размеры обнаруженных лиц в виде списка прямоугольников

        # Выбор ближайшего лица
        if len(faces) > 0: #Этот блок кода выполняется, если были обнаружены лица на кадре

            # Перебор всех обнаруженных лиц
            for (x, y, w, h) in faces:
                # Извлечение региона интереса (ROI) ближайшего лица
                roi = gray[y:y+h, x:x+w] #список прямоугольников (x, y, w, h), где (x, y) - координаты верхнего левого угла прямоугольника, а w и h - его ширина и высота


                intensity = np.mean(roi) #функция из библиотеки NumPy (np), она вычисляет среднее значение интенсивности пикселей в roi, что дает нам среднее значение интенсивности для данного лица на кадре
                intensity_values.append(intensity)

                # Отрисовка прямоугольника вокруг ближайшего лица на изображении frame; (0, 255, 0) -зеленый цвет; 2-толщина линии прямоугольника в пикселях;(x+w, y+h) координаты правого нижнего угла прямоугольника в пикселях
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Вывод кадра с прямоугольником, обозначающим ближайшее лицо
        cv2.imshow('Video', frame)

        # Обработка нажатия клавиши 'q' для выхода из цикла
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов и закрытие окон
    cap.release()
    cv2.destroyAllWindows()

    # возврат временных изменений интенсивности
    intensity_values = np.array(intensity_values)
    return intensity_values


    # Пример использования функции
source_dir = 'C:/Users/79778/PycharmProjects/Practics/AVIMeasurements'  # Путь до папки с видео
file_names = os.listdir(source_dir)
for file_name in file_names:
    video_path = f'C:/Users/79778/PycharmProjects/Practics/AVIMeasurements/{file_name}' # Путь до видео
    vpg_signal = extract_vpg(video_path)
    # Создание массива времени
    time = np.arange(len(vpg_signal))

    # Округление значений амплитуды до трех знаков после запятой
    rounded_vpg_signal = np.round(vpg_signal, 3)*10

    # Сохранение значений амплитуды сигнала по времени в созданный файл
    data = np.column_stack((time, rounded_vpg_signal))
    file = open(f'{file_name}.txt', "w")
    np.savetxt(file, data, delimiter=',', header='Time,Amplitude', comments='', fmt='%.3f')

    # Отображение сигнала на графике
    plt.plot(time, vpg_signal)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Video Plethysmography Signal')
    plt.grid(True)
    plt.show()

