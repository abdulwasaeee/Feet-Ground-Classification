# Feet-Ground-Classification

The assignment provides a gradio-based web application for classifying whether a person's feet are grounded, partially grounded, or not grounded in an image. It combines pose estimation (using YOLOv8 pose) and semantic segmentation (DeepLabV3) to detect the floor and analyze foot contact.

## Features

- detects keypoints of a person using YOLOv8 pose.
- uses DeepLabV3 to segment the floor area in the image.
- determines if each foot is fully grounded, partially grounded, or not grounded.
- example JSON outputs are provided in the `sample_outputs` folder.

## Requirements

Install dependencies with:

```
pip install -r requirements.txt
```

## Usage

1. Place the `yolov8n-pose.pt` model file in the project directory.
2. Run the application:

```
python app.py
```

3. Open the provided Gradio link in your browser.
4. Upload an image to see the annotated result and JSON output.

## File Structure

- `app.py` — Main application with Gradio interface.
- `logic.py` — Foot and person-level classification logic.
- `requirements.txt` — Python dependencies.
- `yolov8n-pose.pt` — YOLOv8 pose model weights (download separately).
- `sample_outputs/` — Example output JSON files.
- `test_images/` — Place your test images here.

## Example Output

```json
{
  "classification": "Fully Grounded",
  "confidence": 0.92,
  "left_foot": "Fully Grounded",
  "right_foot": "Fully Grounded",
  "floor_y": 1713
}
```
