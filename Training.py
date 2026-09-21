import os
import cv2
import numpy as np
import pickle
from PIL import Image
import pandas as pd  # Make sure pandas is installed

# === Configuration ===
image_dir = r"C:\Users\HP\PycharmProjects\Face\.venv\processed_faces"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
id_labels = {}
x_train = []
y_labels = []
log_data = []  # To store face detection log

# === Traverse Image Directory ===
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.lower().endswith(("png", "jpg", "jpeg")):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "_").lower()

            if label not in label_ids:
                label_ids[label] = current_id
                id_labels[current_id] = label
                current_id += 1

            id_ = label_ids[label]

            try:
                pil_image = Image.open(path).convert("L")
                image_array = np.array(pil_image, "uint8")
                faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.3, minNeighbors=5)

                print(f"🖼 Processing {file} for label '{label}' → Found {len(faces)} face(s)")

                log_data.append({
                    "Image": file,
                    "Label": label,
                    "Faces_Detected": len(faces)
                })

                for (x, y, w, h) in faces:
                    roi = image_array[y:y + h, x:x + w]
                    x_train.append(roi)
                    y_labels.append(id_)

            except Exception as e:
                print(f"⚠ Error processing {file}: {e}")

# === Save trained data and labels ===
if x_train and y_labels:
    os.makedirs("trainer", exist_ok=True)

    with open("trainer/labels.pickle", "wb") as f:
        pickle.dump(id_labels, f)

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("trainer/trainer.yml")
    print("✅ Training complete. Model and labels saved in 'trainer/' folder.")
else:
    print("❌ No faces found. Check your dataset structure or image quality.")

# === Save detection log to Excel ===
if log_data:
    df = pd.DataFrame(log_data)
    df.to_excel("trainer/face_detection_log.xlsx", index=False)
    print("📊 Detection log saved to 'trainer/face_detection_log.xlsx'")
