from keras_facenet import FaceNet
import numpy as np
import cv2

embedder = FaceNet()

def get_embedding(face):
    face = cv2.resize(face, (160, 160))
    face = face.astype("float32")
    face = np.expand_dims(face, axis=0)
    embedding = embedder.embeddings(face)
    return embedding[0]
