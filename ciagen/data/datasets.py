import csv
import random
from pathlib import Path
from typing import Dict, List


def select_equal_classes(
    total_labels: List[Path],
    synth_images: List[Path],
    nb_synth_images: int,
) -> List[Path]:
    """Select synthetic images such that classes are balanced, based on their labels."""
    class_to_images: Dict[str, List[Path]] = {}

    for label_path in total_labels:
        with open(label_path, "r") as file:
            class_name = file.readline().strip()

        base_name = label_path.stem
        corresponding_image = next(
            (img for img in synth_images if Path(img).stem.startswith(f"{base_name}_")),
            None,
        )

        if corresponding_image:
            class_to_images.setdefault(class_name, []).append(corresponding_image)

    num_classes = len(class_to_images)
    images_per_class = max(1, nb_synth_images // num_classes)

    selected_images = []
    for class_name, images in class_to_images.items():
        random.shuffle(images)
        selected_images += images[:images_per_class]

    remaining_images = nb_synth_images - len(selected_images)
    if remaining_images > 0:
        available_classes = [cls for cls in class_to_images if len(class_to_images[cls]) > images_per_class]
        for _ in range(remaining_images):
            class_name = random.choice(available_classes)
            selected_images.append(class_to_images[class_name].pop())

    return selected_images


def create_csv_file(
    train_images: List[Path],
    val_images: List[Path],
    test_images: List[Path],
    real_train_captions: List[Path],
    val_captions: List[Path],
    test_captions: List[Path],
    output_csv: Path,
) -> None:
    """Create a CSV file with columns: Filename, Emotion, Dataset."""

    def extract_class_from_caption(caption_path: Path) -> str:
        with open(caption_path, "r") as file:
            return file.readline().strip()

    def map_captions_to_images(captions: List[Path]) -> Dict[str, str]:
        return {caption.stem: extract_class_from_caption(caption) for caption in captions}

    train_name_to_caption = map_captions_to_images(real_train_captions)
    val_name_to_caption = map_captions_to_images(val_captions)
    test_name_to_caption = map_captions_to_images(test_captions)

    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Emotion", "Dataset"])

        for image in train_images:
            base_name = Path(image).stem.split("_")[0]
            emotion = train_name_to_caption.get(base_name, "Unknown")
            writer.writerow([Path(image).name, emotion, "train"])

        for image in val_images:
            image_name = Path(image).stem
            emotion = val_name_to_caption.get(image_name, "Unknown")
            writer.writerow([Path(image).name, emotion, "val"])

        for image in test_images:
            image_name = Path(image).stem
            emotion = test_name_to_caption.get(image_name, "Unknown")
            writer.writerow([Path(image).name, emotion, "test"])
