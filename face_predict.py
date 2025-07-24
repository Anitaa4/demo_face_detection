import numpy as np
import pickle
from keras_facenet import FaceNet
import cv2
from sklearn.preprocessing import LabelEncoder

im_test = 'test_pic\messi_ronaldo_test_im.jpg'

facenet = FaceNet()
faces_embedding = np.load('face_dataset.npz')
y = faces_embedding['arr_1']
encoder = LabelEncoder()
encoder.fit(y)
cascade = 'haarcascade_frontalface_default.xml'
model = pickle.load(open('my_face_rec_model.pkl', 'rb'))

face_detector = cv2.CascadeClassifier(cascade)

image = cv2.imread(im_test)
rgb_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_detector.detectMultiScale(gray_im, scaleFactor=1.3, minNeighbors=5)
for x,y,w,h in faces:
    img = rgb_im[y:y+h, x:x+w]
    img = cv2.resize(img,(160,160))
    img = img.astype('float32')
    img = np.expand_dims(img, axis=0)
    y_pred = facenet.embeddings(img)
    target = model.predict_proba(y_pred)
    target_val = np.max(target)
    target_lable = [np.argmax(target)]
    if target_val > 0.7:
        final_name = encoder.inverse_transform(target_lable)[0]
    else:
        final_name = 'unknown'
    print(target)
    cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0), 3)
    cv2.putText(image, str(final_name), (x,y - 10),cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0,255, 0),
                 3, cv2.LINE_AA)
cv2.imshow('Face recognition', image)
cv2.waitKey(0) 
cv2.destroyAllWindows()