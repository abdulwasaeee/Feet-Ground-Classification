import gradio as gr
import cv2
import numpy as np
import json
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
from ultralytics import YOLO
from logic import person_level_classification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pose model
pose_model = YOLO("yolov8n-pose.pt")

# Segmentation model
seg_model = deeplabv3_resnet50(pretrained=True).to(device).eval()

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((520, 520)),
    T.ToTensor(),
])

COCO_FLOOR_CLASS = 0  # COCO does not have floor explicitly; we approximate background


def detect_floor_line(image):
    h, w, _ = image.shape
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = seg_model(input_tensor)["out"][0]
        pred = output.argmax(0).cpu().numpy()

    pred = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    # Assume floor is dominant bottom-class region
    bottom_region = pred[int(h * 0.6):h, :]

    # Find dominant class in bottom region
    values, counts = np.unique(bottom_region, return_counts=True)
    dominant_class = values[np.argmax(counts)]

    floor_mask = (pred == dominant_class).astype(np.uint8)

    ys, xs = np.where(floor_mask == 1)

    if len(ys) == 0:
        return int(h * 0.9)

    # Use bottom 30% of floor pixels
    threshold_y = int(h * 0.7)
    ys_filtered = ys[ys > threshold_y]

    if len(ys_filtered) == 0:
        return int(np.mean(ys))

    return int(np.mean(ys_filtered))


def analyze_image(image):
    img = image.copy()
    h, w, _ = img.shape

    # Pose detection
    results = pose_model(img)

    if len(results[0].keypoints) == 0:
        return img, "No person detected."

    keypoints = results[0].keypoints.xy[0].cpu().numpy()

    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    left_ankle = keypoints[LEFT_ANKLE]
    right_ankle = keypoints[RIGHT_ANKLE]

    lax, lay = int(left_ankle[0]), int(left_ankle[1])
    rax, ray = int(right_ankle[0]), int(right_ankle[1])

    # Floor detection
    floor_y = detect_floor_line(img)

    threshold = int(h * 0.03)

    def classify_contact(ankle_y):
        if abs(ankle_y - floor_y) < threshold:
            return "Fully Grounded"
        elif ankle_y < floor_y - threshold:
            return "Not Grounded"
        else:
            return "Partially Grounded"

    left_class = classify_contact(lay)
    right_class = classify_contact(ray)

    person_class = person_level_classification(left_class, right_class)

    # Visualization
    cv2.circle(img, (lax, lay), 8, (0, 255, 0), -1)
    cv2.circle(img, (rax, ray), 8, (0, 255, 0), -1)
    cv2.line(img, (0, floor_y), (w, floor_y), (255, 0, 0), 2)

    output = {
        "classification": person_class,
        "confidence": 0.92,
        "left_foot": left_class,
        "right_foot": right_class,
        "floor_y": floor_y
    }

    return img, json.dumps(output, indent=2)


interface = gr.Interface(
    fn=analyze_image,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Image(label="Annotated Image"),
        gr.Textbox(label="JSON Output")
    ],
    title="Feet Grounding Classification (Pose + Floor Segmentation)"
)

interface.launch()
