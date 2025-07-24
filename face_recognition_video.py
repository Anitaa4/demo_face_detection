import numpy as np
import pickle
from keras_facenet import FaceNet
import cv2
from sklearn.preprocessing import LabelEncoder

video_path = 'test_pic/neymar_messi_ronaldo.mp4'

facenet = FaceNet()
faces_embedding = np.load('face_dataset.npz')
y = faces_embedding['arr_1']
encoder = LabelEncoder()
encoder.fit(y)
cascade = 'haarcascade_frontalface_default.xml'
model = pickle.load(open('my_face_rec_model.pkl', 'rb'))

face_detector = cv2.CascadeClassifier(cascade)

cap = cv2.VideoCapture(video_path)
frame_counter = 0
while cap.isOpened():
    # get a frame of video
    ret, frame = cap.read()
    frame_counter += 1
    
    if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frame_counter = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for x,y,w,h in faces:
        img = frame[y:y+h, x:x+w]
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
        
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(frame, str(final_name), (x,y - 10),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 3, cv2.LINE_AA)

        cv2.imshow('Video', frame)

    if (cv2.waitKey(25) & 0xFF) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()