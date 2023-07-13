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
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Выбор ближайшего лица
        if len(faces) > 0:
            # Вычисление расстояний до всех лиц
            distances = []
            for (x, y, w, h) in faces:
                distance = np.sqrt((x - frame.shape[1]/2)**2 + (y - frame.shape[0]/2)**2)
                distances.append(distance)

            # Извлечение ближайшего лица
            closest_face_index = np.argmin(distances)
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
source_dir = 'C:/Users/79778/PycharmProjects/Practics/AVIMeasurements'
file_names = os.listdir(source_dir)
for file_name in file_names:
    video_path = f'C:/Users/79778/PycharmProjects/Practics/AVIMeasurements/{file_name}'
    vpg_signal = extract_vpg(video_path)
    # Создание массива времени
    time = np.arange(len(vpg_signal))

    # Округление значений амплитуды до трех знаков после запятой
    rounded_vpg_signal = np.round(vpg_signal, 3)*10

    # Сохранение значений амплитуды сигнала по времени в файл
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

