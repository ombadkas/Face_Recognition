import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


model = load_model("facial_recognition_model.keras")

classes = ["Swayam", "unknown"]

def predict_class(img):
    img = cv2.resize(img, (150, 150))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize pixel values
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
    return classes[predicted_class], confidence

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('frame', frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    predicted_class, confidence = predict_class(rgb_frame)

    print("Predicted Class:", predicted_class)
    print("Confidence:", confidence)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
