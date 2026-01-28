import cv2
import os
import pandas as pd
from datetime import datetime
from time import sleep
from utils.face_detector import detect_faces
from utils.face_embedding import get_embedding
from utils.matcher import cosine_similarity

# Load registered faces
known_embeddings = []
known_names = []

for user in os.listdir("faces"):
    user_folder = os.path.join("faces", user)
    if not os.path.isdir(user_folder):
        continue
    for img_name in os.listdir(user_folder):
        img_path = os.path.join(user_folder, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        emb = get_embedding(image)
        known_embeddings.append(emb)
        known_names.append(user)

# Start camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit()

print("Camera will open. Get ready! Detecting face in 3 seconds...")
sleep(3)  # 3-second delay

punched_in = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_faces(frame)

    for face in faces:
        x, y, w, h = face['box']
        face_img = frame[y:y+h, x:x+w]
        emb = get_embedding(face_img)

        name = "Unknown"
        for k_emb, k_name in zip(known_embeddings, known_names):
            if cosine_similarity(emb, k_emb) > 0.7:
                name = k_name
                break

        # Punch-In once
        if name != "Unknown" and name not in punched_in:
            time = datetime.now().strftime("%H:%M:%S")
            pd.DataFrame([[name, time, "Punch-In"]],
                         columns=["Name","Time","Type"])\
                .to_csv("attendance.csv", mode="a", header=False, index=False)
            punched_in.add(name)
            print(f"{name} punched in at {time}")
            # Show the face for 2 more seconds
            cv2.imshow("Punch-In", frame)
            cv2.waitKey(2000)
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cv2.imshow("Punch-In Camera", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
