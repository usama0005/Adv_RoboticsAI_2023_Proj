from ultralytics import YOLO
import pyttsx3


def detect_objects_and_announce(image_path):
    # Load YOLO model
    model = YOLO("models/yolov5mu.pt")

    # Detect objects in the image
    results = model(image_path)

    # Extract class names and counts
    class_counts = {}
    # for detection in results.xyxy[0]:
    #     class_id = int(detection[5])
    #     class_name = model.names[class_id]
    #     count = int(detection[4])
    #     class_counts[class_name] = count

    # Convert the information to text
    speech_text = generate_speech_text(class_counts)

    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    # Text-to-speech conversion
    speak(speech_text)

    # Close the text-to-speech engine
    engine.stop()

def generate_speech_text(class_counts):
    object_info = ", ".join([f"{count} {obj}" for obj, count in class_counts.items()])
    if class_counts:
        text = f"I see {object_info}"
    else:
        text = "No objects detected."
    return text

def speak(text):
    # Convert text to speech using pyttsx3
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    image_path = "test.jpg"  # Replace with the path to your image
    detect_objects_and_announce(image_path)
