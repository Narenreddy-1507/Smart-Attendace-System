import cv2
import os

def preprocess_faces(dataset_dir="dataset_face", output_dir="processed_faces"):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Create output directory structure
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for person in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person)
        if not os.path.isdir(person_path):
            continue

        output_person_path = os.path.join(output_dir, person)
        if not os.path.exists(output_person_path):
            os.makedirs(output_person_path)

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠ Skipping {img_path}, failed to load.")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            if len(faces) == 0:
                print(f"⚠ No face found in {img_path}")
                continue

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                resized_face = cv2.resize(face, (200, 200))

                # Save to output folder
                output_img_path = os.path.join(output_person_path, img_name)
                cv2.imwrite(output_img_path, resized_face)
                print(f"✅ Processed and saved {output_img_path}")
                break  # Only process the first face

    print("✅ Dataset preprocessing complete.")

if __name__ == "__main__":
    preprocess_faces()
