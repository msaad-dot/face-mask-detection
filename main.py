import cv2
import imutils
from imutils.video import VideoStream
from tensorflow.keras.models import load_model
from detection import detect_and_predict_mask
from gate_control import GateControl
from temperature_sensor import TemperatureSensor
from utils import send_notification

def apply_logic(label, temp_sensor, gate):
    temp = temp_sensor.get_temp_celsius()

    if temp >= 37:
        gate.buzz_on()
        gate.red_light()
    elif label == "No Mask":
        gate.red_light()
        gate.buzz_off()
        gate.close_gate()
        send_notification("Please wear a mask!")
    else:
        gate.buzz_off()
        gate.green_light()
        gate.open_gate()

def detect_mask(locs, preds, frame, temp_sensor, gate):
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        label_out = f"{label}: {max(mask, withoutMask) * 100:.2f}%"
        temp = temp_sensor.get_temp_celsius()
        person_temp = f"Temp: {temp:.1f}C"

        cv2.putText(frame, label_out, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.putText(frame, person_temp, (endX - 10, endY), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        if gate.detect_person():
            apply_logic(label, temp_sensor, gate)
        else:
            gate.close_gate()

if __name__ == "__main__":
    prototxtPath = "models/face_detector/deploy.prototxt"
    weightsPath = "models/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    maskNet = load_model("models/mask_detector.model")

    gate = GateControl()
    temp_sensor = TemperatureSensor()

    print("[INFO] Starting video stream...")
    vs = VideoStream(src=0, framerate=30).start()

    try:
        while True:
            frame = vs.read()
            frame = imutils.resize(frame, width=1000)
            locs, preds = detect_and_predict_mask(frame, faceNet, maskNet)
            detect_mask(locs, preds, frame, temp_sensor, gate)
            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        gate.cleanup()
        vs.stop()
