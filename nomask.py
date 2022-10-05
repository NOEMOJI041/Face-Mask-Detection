import tensorflow.keras
import numpy as np
import cv2

np.set_printoptions(suppress=False)
model = tensorflow.keras.models.load_model('FaceMD-07.h5')
data = np.ndarray(shape=(1, 128, 128, 3), dtype=np.float32)


cam = cv2.VideoCapture(0)

text = " "
while True:
    _, img = cam.read()
    img = cv2.resize(img, (128, 128))

    image_array = np.asarray(img)

    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)

    for i in prediction:
        if i[0] > 0.9:
            text = 'NoMask'
        elif i[0] < 0.00001:
            text = 'Mask'
        else:
            text = ''




        img = cv2.resize(img, (500, 500))
        cv2.putText(img, text, (10, 30),  cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break