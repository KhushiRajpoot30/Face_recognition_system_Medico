import cv2, os
from utils.face_detector import detect_faces

name = input("Enter user name: ")
path = f"faces/{name}"
os.makedirs(path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while count < 30:
    ret, frame = cap.read()
    faces = detect_faces(frame)

    for face in faces:
        x, y, w, h = face['box']
        face_img = frame[y:y+h, x:x+w]
        cv2.imwrite(f"{path}/{count}.jpg", face_img)
        count += 1
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("Register Face", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Face registered successfully")
