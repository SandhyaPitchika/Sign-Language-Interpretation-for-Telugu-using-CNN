import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os
import tkinter as tk
from threading import Thread

# GUI state variables
label = "A"
counter = 0

# Create base folder
baseFolder = 'sign_language_data'
os.makedirs(baseFolder, exist_ok=True)

# Setup Tkinter GUI
def create_gui():
    global label_var, counter_var
    window = tk.Tk()
    window.title("Label Info")
    window.geometry("250x120")
    window.resizable(False, False)

    tk.Label(window, text="Current Label:", font=("Arial", 14)).pack()
    label_var = tk.StringVar()
    label_display = tk.Label(window, textvariable=label_var, font=("Arial", 16, 'bold'))
    label_display.pack()

    tk.Label(window, text="Image Count:", font=("Arial", 14)).pack()
    counter_var = tk.StringVar()
    counter_display = tk.Label(window, textvariable=counter_var, font=("Arial", 16, 'bold'))
    counter_display.pack()

    label_var.set(label)
    counter_var.set(counter)

    window.mainloop()

# Helper to get next image count
def get_next_count(folder_path):
    files = os.listdir(folder_path)
    return len([f for f in files if f.endswith('.jpg')])

# Start GUI in background
gui_thread = Thread(target=create_gui)
gui_thread.daemon = True
gui_thread.start()

# OpenCV and hand tracking
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

while True:
    success, img = cap.read()
    if not success:
        print("❌ Failed to read from camera.")
        break

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        height, width, _ = img.shape
        x1 = max(x - offset, 0)
        y1 = max(y - offset, 0)
        x2 = min(x + w + offset, width)
        y2 = min(y + h + offset, height)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size != 0:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = int(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = (imgSize - wCal) // 2
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = imgSize / w
                hCal = int(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = (imgSize - hCal) // 2
                imgWhite[hGap:hGap + hCal, :] = imgResize

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    # 🔵 Overlay current label and count
    cv2.putText(img, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.putText(img, f"Count: {counter}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord(' '):
        label = "space"
        print("▶ Label set to: SPACE")

    elif key == ord('d'):
        label = "delete"
        print("▶ Label set to: DELETE")

    elif 65 <= key <= 90:  # Capital A–Z
        label = chr(key)
        print(f"▶ Label set to: {label}")

    elif key == ord('s'):
        folder = os.path.join(baseFolder, label)
        os.makedirs(folder, exist_ok=True)
        counter = get_next_count(folder) + 1
        filename = f"{label}_{str(counter).zfill(3)}.jpg"
        filepath = os.path.join(folder, filename)
        cv2.imwrite(filepath, imgWhite)
        print(f"✅ Saved: {filepath}")

    # Update GUI label and count if available
    try:
        label_var.set(label)
        counter_var.set(counter)
    except:
        pass

cap.release()
cv2.destroyAllWindows()
