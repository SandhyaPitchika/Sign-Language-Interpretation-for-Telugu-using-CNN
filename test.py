
import cv2
import numpy as np
import time
import json
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

# Load model and label map
model = load_model("./model/DeepCNN_f164_f264_d256.h5", compile=False)
with open("./model/label_map.json", "r") as f:
    label_map = json.load(f)
idx_to_label = {v: k for k, v in label_map.items()}

# Initialize
detector = HandDetector(maxHands=1)
cap = cv2.VideoCapture(0)

IMG_SIZE = 64
cam_size = 300
offset = 20

prev_label = ""
sentence = ""
last_pred_time = 0
prediction_delay = 1.0
repeat_gap = 2.5

prev_center = None
stable_start_time = None
stabilization_time = 1.0

last_display_label = ""
last_display_conf = 0.0
last_display_time = 0
display_duration = 2.0
last_hand_position = (0, 0, 0)

# Load and resize reference image (sign chart)
reference_img = cv2.imread("sign_reference.png")
reference_img = cv2.resize(reference_img, (300, 480))  # Resize to fit side-by-side

# Set full-screen mode
cv2.namedWindow("Sign Language Translator", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Sign Language Translator", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def is_stable(center, prev_center, threshold=15):
    if prev_center is None:
        return False
    return np.linalg.norm(np.array(center) - np.array(prev_center)) < threshold

while True:
    success, img = cap.read()
    if not success:
        break

    current_time = time.time()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        center = (x + w // 2, y + h // 2)
        last_hand_position = (x, y, h)

        if int(current_time * 2) % 2 == 0:
            cv2.rectangle(img, (x - 25, y - 25), (x + w + 25, y + h + 25), (0, 255, 0), 3)

        if is_stable(center, prev_center):
            if stable_start_time is None:
                stable_start_time = current_time
        else:
            stable_start_time = None

        if stable_start_time and (current_time - stable_start_time > stabilization_time):
            x1 = max(x - offset, 0)
            y1 = max(y - offset, 0)
            x2 = min(x + w + offset, img.shape[1])
            y2 = min(y + h + offset, img.shape[0])
            img_crop = img[y1:y2, x1:x2]

            img_white = np.ones((cam_size, cam_size, 3), np.uint8) * 255
            aspect_ratio = h / w

            try:
                if aspect_ratio > 1:
                    k = cam_size / h
                    w_cal = int(k * w)
                    img_resize = cv2.resize(img_crop, (w_cal, cam_size))
                    w_gap = (cam_size - w_cal) // 2
                    img_white[:, w_gap:w_gap + w_cal] = img_resize
                else:
                    k = cam_size / w
                    h_cal = int(k * h)
                    img_resize = cv2.resize(img_crop, (cam_size, h_cal))
                    h_gap = (cam_size - h_cal) // 2
                    img_white[h_gap:h_gap + h_cal, :] = img_resize

                if current_time - last_pred_time > prediction_delay:
                    input_img = cv2.resize(img_white, (IMG_SIZE, IMG_SIZE)) / 255.0
                    input_img = np.expand_dims(input_img, axis=0)

                    pred = model.predict(input_img, verbose=0)
                    class_id = np.argmax(pred)
                    confidence = float(np.max(pred))
                    label = idx_to_label[class_id]

                    if label == prev_label and current_time - last_pred_time > repeat_gap:
                        if label == "space":
                            sentence += " "
                        elif label == "delete":
                            sentence = sentence[:-1]
                        else:
                            sentence += label
                        last_pred_time = current_time

                    elif label != prev_label:
                        if label == "space":
                            sentence += " "
                        elif label == "delete":
                            sentence = sentence[:-1]
                        else:
                            sentence += label
                        prev_label = label
                        last_pred_time = current_time

                    last_display_label = label
                    last_display_conf = confidence
                    last_display_time = current_time

            except Exception as e:
                print("Prediction error:", e)

        prev_center = center

    # Display last predicted label
    if last_display_label and (time.time() - last_display_time < display_duration):
        lx, ly, lh = last_hand_position
        cv2.putText(img, f"{last_display_label} ({last_display_conf:.2f})",
                    (lx, ly + lh + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 180, 0), 2)

    # Draw sentence box
    cv2.rectangle(img, (0, 0), (640, 60), (245, 245, 245), -1)
    cv2.putText(img, "Text: " + sentence, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (50, 50, 200), 2)

    # Key controls
    cv2.putText(img, "Q: Quit | C: Clear", (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

    # Resize webcam frame and stack with reference image
    img_resized = cv2.resize(img, (640, 480))
    combined = np.hstack((img_resized, reference_img))
    cv2.imshow("Sign Language Translator", combined)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence = ""
        prev_label = ""

cap.release()
cv2.destroyAllWindows()
