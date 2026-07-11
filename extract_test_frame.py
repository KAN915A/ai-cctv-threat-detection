import cv2

cap = cv2.VideoCapture(
    "Weapons-and-Knives-Detector-with-YOLOv8-main/Results/detected_objects_video.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 64)
ret, frame = cap.read()
cap.release()
cv2.imwrite("web/test_weapon.jpg", frame)
print("saved", frame.shape)
