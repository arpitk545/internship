import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)
frame_count = 0
analysis_result = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    frame_count += 1

    if frame_count % 30 == 0:
        try:
            analysis = DeepFace.analyze(frame, actions=['age', 'gender', 'emotion'], enforce_detection=False)
            analysis_result = analysis[0]
        except:
            analysis_result = None

    if analysis_result:
        text = f"{analysis_result['dominant_emotion']}, {analysis_result['gender']}, {analysis_result['age']}"
        cv2.putText(display_frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Webcam Face Analysis", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
