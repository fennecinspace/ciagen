import numpy as np


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate Intersection over Union between two YOLO-format bounding boxes.

    Boxes are in (x_center, y_center, width, height) format.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_left = max(x1 - w1 / 2, x2 - w2 / 2)
    y_top = max(y1 - h1 / 2, y2 - h2 / 2)
    x_right = min(x1 + w1 / 2, x2 + w2 / 2)
    y_bottom = min(y1 + h1 / 2, y2 + h2 / 2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = w1 * h1
    box2_area = w2 * h2

    return intersection_area / (box1_area + box2_area - intersection_area)


def bbox_min_max_to_center_dims(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Convert (x_min, x_max, y_min, y_max) to YOLO (x_center, y_center, width, height)."""
    x_center = (x_min + x_max) / 2.0 / image_width
    y_center = (y_min + y_max) / 2.0 / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    return x_center, y_center, width, height
