import csv
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from omegaconf import DictConfig

from ciagen.data.datasets import select_equal_classes
from ciagen.utils.io import (
    list_files,
)
from ciagen.utils.io import (
    logger as ciagen_logger,
)


def sort_based_on_score(
    image_paths: List[str], scores: List[float], direction: str = "smaller"
) -> Tuple[List[str], List[int]]:

    combined_data = list(zip(scores, image_paths))
    sorted_data = sorted(
        combined_data,
        key=lambda x: x[0],
        reverse=False if direction == "smaller" else True,
    )
    sorted_scores, sorted_image_paths = zip(*sorted_data)
    return sorted_image_paths, sorted_scores


def create_csv_file_path(
    train_images: List[str],
    val_images: List[str],
    test_images: List[str],
    real_train_labels: List[Path],
    val_labels: List[Path],
    test_labels: List[Path],
    output_csv: str,
) -> None:

    def extract_class_from_label(label_path: Path) -> str:
        with open(label_path, "r") as file:
            cap = file.readline().strip()
        return cap

    def map_labels_to_images(labels: List[Path]) -> Dict[str, str]:
        return {label.stem: extract_class_from_label(label) for label in labels}

    train_name_to_label = map_labels_to_images(real_train_labels)
    val_name_to_label = map_labels_to_images(val_labels)
    test_name_to_label = map_labels_to_images(test_labels)

    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["Filename", "Emotion", "Dataset"])

        for image in train_images:
            base_name = os.path.splitext(os.path.basename(image))[0].split("_")[0]
            emotion = train_name_to_label.get(base_name, "Unknown")
            writer.writerow([image, emotion, "train"])

        for image in val_images:
            base_name = os.path.splitext(os.path.basename(image))[0]
            emotion = val_name_to_label.get(base_name, "Unknown")
            writer.writerow([image, emotion, "val"])

        for image in test_images:
            base_name = os.path.splitext(os.path.basename(image))[0]
            emotion = test_name_to_label.get(base_name, "Unknown")
            writer.writerow([image, emotion, "test"])


def mix_fer(cfg: DictConfig, paths: Dict[str, str | Path]) -> None:

    augmentation_percent = cfg["ml"]["augmentation_percent"]
    train_nb = cfg["ml"]["train_nb"]
    val_nb = cfg["ml"]["val_nb"]
    test_nb = cfg["ml"]["test_nb"]
    _sample = cfg["ml"]["sampling"]
    seed = 42
    formats = cfg["data"]["image_formats"]

    if not os.path.isdir(paths["mixed_yamls_folder_path"]):
        os.makedirs(paths["mixed_yamls_folder_path"])

    data_csv_path = Path(paths["mixed_yamls_folder_path"]) / "train_dataset.csv"

    real_images_path = Path(paths["real_images"])
    val_images_path = Path(paths["val_images"])
    test_images_path = Path(paths["test_images"])

    real_train_labels = list(Path(paths["real_labels"]).glob("*.txt"))
    real_test_labels = list(Path(paths["test_labels"]).glob("*.txt"))
    real_val_labels = list(Path(paths["val_labels"]).glob("*.txt"))

    synth_images_dir = Path(paths["generated"])

    total_captions = real_train_labels + real_test_labels + real_val_labels

    real_images = list_files(real_images_path, formats, train_nb)
    val_images = list_files(val_images_path, formats, val_nb)
    test_images = list_files(test_images_path, formats, test_nb)

    synth_images_full = list_files(synth_images_dir, formats)

    real_img_dict = {}
    for img in real_images:
        real_img_id = img.split(os.sep)[-1].split("_")[0]
        for format in formats:
            real_img_id = real_img_id.replace(f".{format}", "")

        real_img_dict[real_img_id] = True

    synth_images = []
    for img in synth_images_full:
        synth_img_id = img.split(os.sep)[-1].split("_")[0]
        for format in formats:
            synth_img_id = synth_img_id.replace(f".{format}", "")

        if synth_img_id in real_img_dict:
            synth_images.append(img)

    random.Random(seed).shuffle(synth_images)
    random.Random(seed).shuffle(real_images)

    if cfg["ml"]["keep_training_size"]:
        _nb_real_images = int(len(real_images) * (1 - augmentation_percent))
    else:
        _nb_real_images = train_nb
    nb_synth_images = int(len(real_images) * augmentation_percent)

    ciagen_logger.info(f"Total captions: {len(total_captions)}")
    ciagen_logger.info(f"Synthetic images: {len(synth_images)}")

    if cfg["ml"]["with_filtering"]:
        dataset_name = cfg["data"]["base"]
        cn_name = cfg["model"]["cn_use"]

        metadata_path = os.path.join("data", "generated", dataset_name, cn_name)
        metadata_file = os.path.join(metadata_path, "metadata.yaml")
        filtering_metric = cfg["ml"]["filtering_metric"]

        preferred_fe = cfg["ml"]["preferred_fe"]
        with open(metadata_file, "r") as f:
            metadata_dict = yaml.safe_load(f)

        filtered_images = metadata_dict["results"]["filtering"][filtering_metric][preferred_fe]

        synth_images = list(filtered_images.keys())

    synth_images = select_equal_classes(total_captions, synth_images, nb_synth_images)

    train_images = real_images + synth_images

    create_csv_file_path(
        train_images=train_images,
        val_images=val_images,
        test_images=test_images,
        real_train_labels=real_train_labels,
        val_labels=real_val_labels,
        test_labels=real_test_labels,
        output_csv=data_csv_path,
    )

    ciagen_logger.info(f"Training csv files created in : {paths['mixed_yamls_folder_path']}")
    ciagen_logger.info(f"Using {train_nb} Real Images from : {real_images_path}")
    ciagen_logger.info(f"Using Synthetic Images from : {synth_images_dir}, a total of {len(synth_images)} images")
    ciagen_logger.info(f"Using in total {len(train_images)} images for training")
    ciagen_logger.info(f"Using {val_nb} Validation Images from : {val_images_path}")
    ciagen_logger.info(f"Using {test_nb} Test Images from : {test_images_path}")

    return data_csv_path


if __name__ == "__main__":
    from omegaconf import OmegaConf

    from ciagen.data.paths import generate_all_paths

    cfg = OmegaConf.load("ciagen/conf/config.yaml")
    paths = generate_all_paths(cfg)
    mix_fer(cfg, paths)
