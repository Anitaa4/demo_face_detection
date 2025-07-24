import cv2
import os
import numpy as np
from keras_facenet import FaceNet
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pickle

# Create dataset to train Model
class CreateDataset:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160, 160)
        self.X = []
        self.y = []
        self.detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def extract_face(self, fname):
        img = cv2.imread(fname)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(img_gray,scaleFactor=1.3, minNeighbors=5)
        x, y, w, h = faces[0]
        face = img_rgb[y:y+h, x:x+w]
        face_arr = cv2.resize(face, self.target_size)
        return face_arr
    
    def load_faces(self, dir):
        FACES = []
        for im_name in os.listdir(dir):
            try:
                path = dir + im_name
                face = self.extract_face(path)
                FACES.append(face)
            except Exception as e:
                pass
        return FACES
    
    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = self.directory + '/' + sub_dir +'/'
            FACES = self.load_faces(path)
            labels = [sub_dir for _ in range(len(FACES))]
            self.X.extend(FACES)
            self.y.extend(labels)
        return np.asarray(self.X), np.asarray(self.y)


# FaceNet: faces --> vector
embedder = FaceNet()
def get_embedding(face_img):
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)
    face_vec = embedder.embeddings(face_img)
    return face_vec[0]

# Create dataset
im_dir = 'train_pic'
create_dataset = CreateDataset(im_dir)
X, y = create_dataset.load_classes()
EMBEDDED_X = []
for img in X:
    EMBEDDED_X.append(get_embedding(img))
EMBEDDED_X = np.asarray(EMBEDDED_X)
np.savez_compressed('face_dataset.npz', EMBEDDED_X, y)


# Train SMV model
encoder = LabelEncoder()
y = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(EMBEDDED_X, y, 
                                                    shuffle=True, test_size=0.2, 
                                                    random_state=1)
print(X_train)
print(y_train)
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print('score: {}'.format(score))
print('confusion metrix\n', confusion_matrix(y_test, y_pred))
print('classification report\n', classification_report(y_test, y_pred))

# save model
pickle.dump(model, open('my_face_rec_model.pkl', 'wb'))