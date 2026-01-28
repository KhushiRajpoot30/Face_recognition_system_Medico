from mtcnn import MTCNN

detector = MTCNN()

def detect_faces(frame):
    faces = detector.detect_faces(frame)
    return faces
