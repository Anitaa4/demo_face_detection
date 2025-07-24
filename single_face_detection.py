import cv2
import numpy as np
from keras_facenet import FaceNet

fname = 'sample_pic.jpg' # image path 

# read image
im = cv2.imread(fname) 
im = cv2.resize(im, (int(0.5 * im.shape[1]), int(0.5 * im.shape[0])))
gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# detect face using cascade classifier
cascade = 'haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(cascade)

faces = face_detector.detectMultiScale(gray_im, scaleFactor=1.3, minNeighbors=5)


# draw rectangle over detected faces
for (x, y,w,h) in faces:
    cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 3)

# crop only face
face = im[y:y+h, x:x+w]
face = cv2.resize(face, (160, 160))

cv2.imshow('messi', im)
cv2.imshow('face', face)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Embedding face
embedder = FaceNet()
face = face.astype('float32')
face = np.expand_dims(face, axis=0)
embedded_face = embedder.embeddings(face)
print(type(embedded_face))
print('shape {}'.format(embedded_face.shape))
print(embedded_face[0])
