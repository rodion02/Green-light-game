import cv2
import imutils as imutils
import numpy as np
import time


def diffImg(a, b, c):
    """
    Функция простая - мы блюрим три входных серых кадра, а затем вычитаем из второго первый и из третьего второй, а
    результаты объединяем. Для чего это сделано? Всё просто, это улучшенный алгоритм считывания движений как по мне.
    В конце мы выводим результат объединения.
    """
    a = cv2.GaussianBlur(a, (21, 21), 0)
    b = cv2.GaussianBlur(b, (21, 21), 0)
    c = cv2.GaussianBlur(c, (21, 21), 0)
    d1 = cv2.absdiff(c, b)
    d2 = cv2.absdiff(b, a)
    return cv2.bitwise_and(d1, d2)


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    frame1 = cap.read()[1]  # Считываем сразу 3 кадра с камеры и сразу делаем их серые копии
    firstFrame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = cap.read()[1]
    secondFrame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    frame3 = cap.read()[1]
    thirdFrame = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)

    now = time.time()   # Задаём значения таймера, это нам понадобится чуть позже
    future1 = now + 13
    future2 = now - 1

    while True:
        frameDelta = diffImg(firstFrame, secondFrame, thirdFrame)   # Применяем функцию к трём серым кадрам
        text1 = "No motion"     # Опциональное остояние для нашей игры, когда нельзя двигаться
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]   # Тут идёт долгий процесс считывания контуров,
        thresh = cv2.dilate(thresh, None, iterations=2)     # но если кратко, то я получаю "карту движений" из функции
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # diffImg и перделываю всё в
        cnts = imutils.grab_contours(cnts)  # контуры. (В предыдущей строке я искал контуры, а тут я их помещаю в cnt)

        mask = np.zeros(frame1.shape[:2], frame1.dtype)     # Формирую поле нулей и размера кадра для создания маски
        mask = cv2.fillPoly(mask, cnts, (255,) * frame1.shape[2], )     # Заполняю пустое поле нулей контурами => маска
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Тут я задаю форму отображения контуров в маски,
        mask = cv2.dilate(mask, kernel, iterations=2)   # форма - квадрат 2х2 и резкость 2

        mask1 = cv2.bitwise_not(mask)   # Формирую маску противоположную предыдущей, ведь фон то должен быть зелёным
        green_Img = np.zeros(frame1.shape, frame1.dtype)    # Создаём зеленоватый фон
        green_Img[:, :] = (0, 30, 0)
        greenMask = cv2.bitwise_and(green_Img, green_Img, mask=mask1)   # И получаем зеленую маску для заднего фона

        if time.time() <= future1:  # В условии мы проверяем наш таймер (у нас их 2 - каждый для своего состояния)
            secs = int(future1 - time.time())   # Опциональная переменная, демонстрирующая сколько осталось секунд
            frame1 = cv2.addWeighted(greenMask, 1, frame1, 1, 0, frame1)    # Наносим опциональный зелёный фон
            greenMask = cv2.bitwise_and(green_Img, green_Img, mask=mask)    # Переделываем красную маску в зелёную
            frame1 = cv2.addWeighted(greenMask, 1, frame1, 1, 0, frame1)    # Закрашиваем движущийся объект в зелёный
            text = "Everything is OK"   # Статус игры - всё хорошо, можно двигаться
            if secs == 0:   # Когда таймер обнуляется, мы заводим таймер для второго состояния
                future2 = time.time() + 14

        elif time.time() <= future2:
            for c in cnts:  # Тут мы проверяеи наше изображениен на наличие движений, если они есть - меняем доп.статус
                text1 = "Something is happening"
            cv2.putText(frame1, "Status: {}".format(text1), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            secs = int(future2 - time.time())   # Всё та же опциональная переменная
            red_Img = np.zeros(frame1.shape, frame1.dtype)  # Делаем красную маску для движущихся контуров по аналогии как
            red_Img[:, :] = (0, 0, 255)  # и зелённую до этого
            redMask = cv2.bitwise_and(red_Img, red_Img, mask=mask)  # Конечный результат
            frame1 = cv2.addWeighted(redMask, 1, frame1, 1, 0, frame1)  # Накладываем красную маску на движущийся объект
            frame1 = cv2.addWeighted(greenMask, 1, frame1, 1, 0, frame1)    # Накладываем задний фон
            text = "Dangerous Situation!"   # Меняем состояние игры
            if secs == 0:   # Когда таймер обнуляется, мы заводим таймер для первого состояния
                future1 = time.time() + 5

        frame = imutils.resize(frame1, width=1000)
        cv2.putText(frame1, "Time left: {} seconds".format(secs), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame1, "Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow('Frame', frame1)     # Сверху был вывод ткста на кадр, а теперь и сам вывод кадра

        frame1 = frame2     # Чтобы не терять 2 кадра, мы их перепесываем => 1->2, 2->3, 3 мы считываем заново
        firstFrame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2 = frame3
        secondFrame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame3 = cap.read()[1]
        thirdFrame = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)

        key = cv2.waitKey(20) & 0xff    # Пишем завершабщую процес кнопку
        if key == 27:  # Esc
            break

    cv2.destroyAllWindows()     # В конце закрываем все окна
    cap.release()


if __name__ == '__main__':
    main()
