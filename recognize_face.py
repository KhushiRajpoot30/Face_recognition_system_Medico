import cv2
import os
import pandas as pd
from datetime import datetime
from utils.face_detector import detect_faces
from utils.face_embedding import get_embedding
from utils.matcher import cosine_similarity

# ---------------------------
# Track users already punched in
# ---------------------------
logged_in = set()

# ---------------------------
# Load all known faces
# ---------------------------
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

# ---------------------------
# Start camera
# ---------------------------
cap = cv2.VideoCapture(0)  # Change to 1,2 if you have multiple cameras
if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit()

# ---------------------------
# Real-time face recognition
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
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

        # Punch-In only once
        if name != "Unknown" and name not in logged_in:
            time = datetime.now().strftime("%H:%M:%S")
            pd.DataFrame([[name, time, "Punch-In"]],
                         columns=["Name","Time","Type"])\
                .to_csv("attendance.csv", mode="a", header=False, index=False)
            logged_in.add(name)
            print(f"{name} punched in at {time}")

        # Draw rectangle and name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Face Attendance System", frame)

    # Press ESC to exit manually
    if cv2.waitKey(1) & 0xFF == 27:
        print("Manual exit")
        break

    # --------------------------
    # Automatic exit if all users punched in
    # --------------------------
    registered_users = set(os.listdir("faces"))
    if registered_users <= logged_in:
        print("All users have punched in. Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
