import cv2
import pyttsx3
from gtts import gTTS
import torch
from ultralytics import YOLO

# Load the YOLOv5 model
def load_yolo_model(model_path):
    model = YOLO(model_path)
    return model

# Perform object detection using the loaded YOLO model
# Perform object detection using the loaded YOLO model
def detect_objects(image, model, conf_threshold=0.4, iou_threshold=0.5):
    with torch.no_grad():
        results = model(image)

    # Extract bounding box coordinates from the 'boxes' attribute within the 'results' object
    bboxes = results.pred[0][:, :4].cpu().numpy()
    confidences = results.pred[0][:, 4].cpu().numpy()
    
    # Filter by confidence threshold
    valid_indices = confidences > conf_threshold
    bboxes = bboxes[valid_indices]

    # Perform non-maximum suppression
    keep = cv2.dnn.NMSBoxes(bboxes.tolist(), confidences.tolist(), conf_threshold, iou_threshold)
    
    # Extract class labels
    class_labels = results.names[results.pred[0][:, 5].cpu().numpy().astype(int)]

    final_predictions = [class_labels[i[0]] for i in keep]

    return final_predictions




# Non-maximum suppression
def non_max_suppression(predictions, iou_threshold):
    output = []

    if predictions is not None:
        keep = [True] * len(predictions)

        for i in range(len(predictions)):
            if keep[i]:
                for j in range(i + 1, len(predictions)):
                    if keep[j]:
                        box1 = predictions[i][:4]
                        box2 = predictions[j][:4]
                        iou = box_iou(box1, box2)
                        if iou > iou_threshold:
                            if predictions[i][4] > predictions[j][4]:
                                keep[j] = False
                            else:
                                keep[i] = False

        output = [predictions[i] for i in range(len(predictions)) if keep[i]]

    return output


# Convert (x, y, w, h) to (x1, y1, x2, y2)
def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


# Calculate IoU (Intersection over Union) for two boxes
def box_iou(box1, box2):
    tl = torch.max(box1[:, None, :2], box2[:, :2])
    br = torch.min(box1[:, None, 2:], box2[:, 2:])

    wh = br - tl
    wh[wh < 0] = 0
    inter = wh[:, 0] * wh[:, 1]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2 - inter

    return inter / union


# Extract object names from predictions
def get_object_names(predictions, class_names):
    object_names = [class_names[int(p[5])] for p in predictions] if predictions is not None else []
    return object_names


# Generate speech text based on detected objects
def generate_speech_text(object_names):
    if object_names:
        text = f"I see {', '.join(object_names)}"
    else:
        text = "No objects detected."
    return text


# Text-to-speech conversion
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def main():
    model_path = "models/yolov5m.pt"  # Provide the correct path to your YOLOv5 model
    yolo_model = load_yolo_model(model_path)
    cap = cv2.VideoCapture(0)  # Use your camera here

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        predictions = detect_objects(frame, yolo_model)
        object_names = get_object_names(predictions)
        speech_text = generate_speech_text(object_names)
        speak(speech_text)

        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    #cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
