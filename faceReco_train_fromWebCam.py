#pylint:disable=no-member

import os
import cv2 as cv
import numpy as np



haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

features = []
labels = []

webcam = cv.VideoCapture(1)

print("press q to Stop Treaning")

lebel=input("Enter Your Name : ")


while True:
    ret, frame = webcam.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h, x:x+w]
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)
        features.append(faces_roi)
        labels.append(0)

    cv.imshow('Training', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
print('Training done ---------------')