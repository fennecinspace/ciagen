import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from omegaconf import DictConfig


def generate_all_paths(cfg: DictConfig) -> Dict[str, str | Path]:
    """Generate all data paths from a Hydra configuration.

    Creates directories if they don't exist and returns a dictionary of all
    relevant paths for real, generated, and mixed data.
    """
    real_root = os.path.join("data", "real")
    real_dataset = os.path.join("data", "real", cfg["data"]["base"])
    generated_dataset = os.path.join(
        "data", "generated", cfg["data"]["base"], cfg["model"]["cn_use"]
    )

    mixed_yamls_folder_path = os.path.join(
        "data",
        "mixed",
        cfg["data"]["base"],
        str(cfg["ml"]["train_nb"]),
        cfg["model"]["cn_use"] + "-" + str(cfg["ml"]["augmentation_percent"]),
    )

    train_path = os.path.join(real_dataset, "train")
    test_path = os.path.join(real_dataset, "test")
    val_path = os.path.join(real_dataset, "val")

    real_train_images_path = os.path.join(train_path, "images")
    real_train_labels_path = os.path.join(train_path, "labels")
    real_train_captions_path = os.path.join(train_path, "captions")

    real_test_images_path = os.path.join(test_path, "images")
    real_test_labels_path = os.path.join(test_path, "labels")
    real_test_captions_path = os.path.join(test_path, "captions")

    real_val_images_path = os.path.join(val_path, "images")
    real_val_labels_path = os.path.join(val_path, "labels")
    real_val_captions_path = os.path.join(val_path, "captions")

    vocabulary_config_path = os.path.join(*cfg["prompt"]["template"])

    for d in (
        real_train_images_path,
        real_train_labels_path,
        real_train_captions_path,
        real_test_images_path,
        real_test_labels_path,
        real_test_captions_path,
        real_val_images_path,
        real_val_labels_path,
        real_val_captions_path,
    ):
        os.makedirs(d, exist_ok=True)

    if cfg["task"] not in ["coco", "flickr30k"]:
        if not os.path.exists(real_train_images_path):
            raise ValueError(
                f"One of the real dataset paths does not exist: {real_train_images_path}"
            )
        if not os.path.exists(real_test_images_path):
            raise ValueError(
                f"One of the real dataset paths does not exist: {real_test_images_path}"
            )
        if not os.path.exists(real_val_images_path):
            raise ValueError(
                f"One of the real dataset paths does not exist: {real_val_images_path}"
            )

    os.makedirs(generated_dataset, exist_ok=True)

    return {
        "root": real_root,
        "real": real_dataset,
        "generated": generated_dataset,
        "mixed_yamls_folder_path": mixed_yamls_folder_path,
        "real_images": real_train_images_path,
        "real_captions": real_train_captions_path,
        "real_labels": real_train_labels_path,
        "test_images": real_test_images_path,
        "test_captions": real_test_captions_path,
        "test_labels": real_test_labels_path,
        "val_images": real_val_images_path,
        "val_captions": real_val_captions_path,
        "val_labels": real_val_labels_path,
        "vocabulary_config": vocabulary_config_path,
    }


def get_model_config(name: str, models: List[Dict]) -> Optional[Dict]:
    """Look up a model configuration by name from the config list."""
    for model_dict in models:
        if name in model_dict:
            return model_dict[name]
    return None


def create_yaml_file(save_path: Path, train: Path, val: Path, test: Path) -> None:
    """Create a YOLO-format data.yaml file."""
    yaml_file = {
        "train": str(Path(train).absolute()),
        "val": str(Path(val).absolute()),
        "test": str(Path(test).absolute()),
        "names": {0: "person"},
    }
    with open(save_path, "w") as file:
        yaml.dump(yaml_file, file)
