import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict

import wget
from omegaconf import DictConfig
from tqdm import tqdm

from ciagen.utils.io import logger as ciagen_logger


def download_mocs(
    data_path: Path,
    size: str = "small",
):
    dataset_links = {
        "extra_small": "https://nextcloud.ig.umons.ac.be/s/tfoeSBoDDE3mzHp/download/MOCS_extra_small.zip",
        "small": "https://nextcloud.ig.umons.ac.be/s/SWfyj4wAqCtRGYy/download/MOCS_small.zip",
        "medium": "",
        "large": "",
        "full": "",
    }

    data_url = dataset_links[size]

    all_images_path = Path(data_path) / "Images"
    all_bbox_path = Path(data_path) / "Detection"
    all_segmentation_path = Path(data_path) / "Segmentation"
    all_captions_path = Path(data_path) / "Captions"

    path_to_data_zip = Path(data_path) / f"MOCS_{size}.zip"

    if not os.path.exists(path_to_data_zip):
        ciagen_logger.info(f"Downloading zip images from {data_url} to {path_to_data_zip}")
        wget.download(data_url, out=str(path_to_data_zip))

        with zipfile.ZipFile(path_to_data_zip, "r") as zip_ref:
            zip_ref.extractall(str(data_path))

    return all_images_path, all_bbox_path, all_segmentation_path, all_captions_path


def prepare_mocs(cfg: DictConfig, paths: Dict[str, str | Path]) -> None:
    real_path = paths["root"]
    real_path_dataset = os.path.join(real_path, "mocs")

    os.makedirs(real_path_dataset, exist_ok=True)

    image_path, bbox_labels_dir, segmentation_labels_dir, all_captions_path = download_mocs(real_path_dataset)

    origin_labels_dir = bbox_labels_dir

    test_nb = cfg["ml"]["test_nb"]
    val_nb = cfg["ml"]["val_nb"]
    train_nb = cfg["ml"]["train_nb"]

    real_train_images_path = Path(paths["real_images"])
    real_test_images_path = Path(paths["test_images"])
    real_val_images_path = Path(paths["val_images"])

    real_train_labels_path = Path(paths["real_labels"])
    real_test_labels_path = Path(paths["test_labels"])
    real_val_labels_path = Path(paths["val_labels"])

    real_train_captions_path = Path(paths["real_captions"])
    real_test_captions_path = Path(paths["test_captions"])
    real_val_captions_path = Path(paths["val_captions"])

    ciagen_logger.info(f"Moving TRAIN to {str(real_train_images_path)}")
    ciagen_logger.info(f"Moving TEST to {str(real_test_images_path)}")
    ciagen_logger.info(f"Moving VAL to {str(real_val_images_path)}")
    ciagen_logger.info(f"Using values test: {test_nb} and validation: {val_nb}")

    all_images = os.listdir(image_path)

    all_captions = os.listdir(all_captions_path.absolute())

    all_captions = list(map(lambda x: x.split(".")[0], all_captions))

    all_images = list(filter(lambda x: x.split(".")[0] in all_captions, all_images))

    length = val_nb + test_nb + train_nb if (val_nb + test_nb + train_nb) < len(all_images) else len(all_images)

    all_images = all_images[:length]

    counter = 0
    for file_name in tqdm(all_images, unit="img"):
        if counter > val_nb + test_nb + train_nb:
            break

        name = file_name.split(".")[0]
        img_file = name + ".jpg"
        txt_file = name + ".txt"

        image = image_path / img_file
        label = origin_labels_dir / txt_file
        caption = all_captions_path / txt_file

        if os.path.isfile(image) and os.path.isfile(label):
            if counter < val_nb:
                images_dir = real_val_images_path
                labels_dir = real_val_labels_path
                captions_dir = real_val_captions_path
            elif counter < val_nb + test_nb:
                images_dir = real_test_images_path
                labels_dir = real_test_labels_path
                captions_dir = real_test_captions_path
            else:
                images_dir = real_train_images_path
                labels_dir = real_train_labels_path
                captions_dir = real_train_captions_path

            shutil.copy(image, os.path.join(images_dir, img_file))
            shutil.copy(label, os.path.join(labels_dir, txt_file))
            shutil.copy(caption, os.path.join(captions_dir, txt_file))

            counter += 1


if __name__ == "__main__":
    from omegaconf import OmegaConf

    from ciagen.data.paths import generate_all_paths

    cfg = OmegaConf.load("ciagen/conf/config.yaml")
    paths = generate_all_paths(cfg)
    prepare_mocs(cfg, paths)
