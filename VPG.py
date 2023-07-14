import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Функция для извлечения ВПГ из видеоизображения
def extract_vpg(video_path):
    # Загрузка видео
    cap = cv2.VideoCapture(video_path)

    # Инициализация массива для сохранения временных изменений интенсивности
    intensity_values = []

    # Чтение видеокадров до конца видео
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        # Преобразование кадра в оттенки серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Детекция лица
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # в некоторых версиях необходимо использовать    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml') вместо строки 25
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Определение минимального значения стороны прямоугольника лица
        min_face_side = 150

        # Выбор ближайшего лица
        if len(faces) > 0:
            # Инициализация переменных для хранения индекса и стороны ближайшего лица
            closest_face_index = None
            closest_face_side = None

            # Перебор всех обнаруженных лиц
            for i, (x, y, w, h) in enumerate(faces):
                # Проверка, соответствует ли сторона прямоугольника минимальному значению
                if w >= min_face_side and h >= min_face_side:
                    # Проверка, является ли это первым лицом или оно ближе к центру кадра
                    if closest_face_side is None or (w + h) < closest_face_side:
                        closest_face_index = i
                        closest_face_side = w + h

            # Извлечение ближайшего лица
            if closest_face_index is not None:
                (x, y, w, h) = faces[closest_face_index]

                # Извлечение региона интереса (ROI) ближайшего лица
                roi = gray[y:y+h, x:x+w]

                # Извлечение среднего значения интенсивности пикселей в ROI
                intensity = np.mean(roi)
                intensity_values.append(intensity)

                # Отрисовка прямоугольника вокруг ближайшего лица
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
source_dir = 'C:/Users/79778/PycharmProjects/Practics/AVIMeasurements' # Путь до папки с видео 
file_names = os.listdir(source_dir)
for file_name in file_names:
    video_path = f'C:/Users/79778/PycharmProjects/Practics/AVIMeasurements/{file_name}' # путь жо папки с видео
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

