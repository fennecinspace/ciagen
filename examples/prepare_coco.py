import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict

import cv2
import wget
from omegaconf import DictConfig
from pycocotools.coco import COCO
from tqdm import tqdm

from ciagen.utils.io import logger as ciagen_logger


def cocobox2yolo(img_path, coco_box):
    img = cv2.imread(img_path)
    image_height, image_width = img.shape[0:2]

    [left, top, box_width, box_height] = coco_box
    x_center = (left + box_width / 2) / image_width
    y_center = (top + box_height / 2) / image_height

    box_width /= image_width
    box_height /= image_height
    yolo_box = [x_center, y_center, box_width, box_height]

    return yolo_box


def download_coco(
    data_path: Path,
    coco_bbx: str = "Coco_1FullPerson_bbx",
    coco_caps: str = "Coco_1FullPerson_caps",
    image_path: str = "Coco_1FullPerson",
    annotations_path: str = "annotations",
    data_zip_name: str = "Coco_1FullPerson.zip",
    annotations_zip_name: str = "annotations_trainval2017.zip",
):

    data_path = Path(data_path)
    image_path: Path = data_path / Path(image_path)
    annotations_path: Path = data_path / Path(annotations_path)
    bbx_path: Path = data_path / Path(coco_bbx)
    caps_path: Path = data_path / Path(coco_caps)

    dirs = data_path, image_path, annotations_path, bbx_path, caps_path
    ciagen_logger.info(f"Attempting to create directories {[str(d) for d in dirs]}")
    for d in dirs:
        if isinstance(d, Path):
            d.mkdir(parents=True, exist_ok=True)
        else:
            os.makedirs(d, exist_ok=True)

    path_to_data_zip = data_path / data_zip_name
    path_to_annotations_zip = data_path / annotations_zip_name

    data_url = (
        "https://cloud.deepilia.com/s/BQcq8nQxFjizdFz/download/Coco_1FullPerson.zip"
    )
    annotations_url = (
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    )

    if not os.path.exists(path_to_data_zip):
        ciagen_logger.info(
            f"Downloading zip images from {data_url} to {path_to_data_zip}"
        )
        wget.download(data_url, out=str(path_to_data_zip))
    if not len(os.listdir(image_path)):
        ciagen_logger.info(f"Extracting zip images to {image_path}")
        with zipfile.ZipFile(path_to_data_zip, "r") as zip_ref:
            zip_ref.extractall(str(data_path))

    if not os.path.exists(path_to_annotations_zip):
        ciagen_logger.info(
            f"Downloading zip annotations from {annotations_url} to {path_to_annotations_zip}"
        )
        wget.download(annotations_url, out=str(path_to_annotations_zip))
    if not len(os.listdir(annotations_path)):
        ciagen_logger.info(f"Extracting zip annotations to {annotations_path}")
        with zipfile.ZipFile(path_to_annotations_zip, "r") as zip_ref:
            zip_ref.extractall(str(data_path))

    return image_path, annotations_path, bbx_path, caps_path


def prepare_coco(cfg: DictConfig, paths: Dict[str, str | Path]) -> None:
    real_path = paths["root"]
    real_path_dataset = os.path.join(real_path, "coco")

    os.makedirs(real_path_dataset, exist_ok=True)

    image_path, annotations_path, bbx_path, caps_path = download_coco(
        real_path_dataset
    )

    coco_version = "train2017"

    annFile = annotations_path / f"instances_{coco_version}.json"
    annFile_keypoints = annotations_path / f"person_keypoints_{coco_version}.json"
    annFile_captions = annotations_path / f"captions_{coco_version}.json"

    coco = COCO(annFile.absolute())
    coco_keypoints = COCO(annFile_keypoints.absolute())
    coco_captions = COCO(annFile_captions.absolute())

    catIds = coco.getCatIds(catNms=["person"])
    all_images = list(image_path.glob("*.jpg"))

    ciagen_logger.info("Writting captions and boxes info ...")
    for img_path in tqdm(all_images, unit="img"):
        img_path = str(img_path.absolute())
        img_id = int(img_path.split("/")[-1].split(".jpg")[0])

        Keypoints_annIds = coco_keypoints.getAnnIds(
            imgIds=img_id, catIds=catIds, iscrowd=None
        )
        Keypoints_anns = coco_keypoints.loadAnns(Keypoints_annIds)

        caps_annIds = coco_captions.getAnnIds(imgIds=img_id)
        caps_anns = coco_captions.loadAnns(caps_annIds)

        bbox_text_path = img_path.replace(".jpg", ".txt").replace(
            "Coco_1FullPerson", "Coco_1FullPerson_bbx"
        )
        captions_text_path = img_path.replace(".jpg", ".txt").replace(
            "Coco_1FullPerson", "Coco_1FullPerson_caps"
        )

        with open(bbox_text_path, "w") as file:
            coco_box = Keypoints_anns[0]["bbox"]
            yolo_box = cocobox2yolo(img_path, coco_box)
            KP_Yolo_format = "0 " + " ".join(list(map(str, yolo_box)))
            file.write(KP_Yolo_format)

        with open(captions_text_path, "w") as file:
            captions = [caps["caption"] for caps in caps_anns]
            file.write("\n".join(captions))

    test_nb = cfg["ml"]["test_nb"]
    val_nb = cfg["ml"]["val_nb"]
    train_nb = cfg["ml"]["train_nb"]

    real_train_images_path = paths["real_images"]
    real_test_images_path = paths["test_images"]
    real_val_images_path = paths["val_images"]

    real_train_labels_path = paths["real_labels"]
    real_test_labels_path = paths["test_labels"]
    real_val_labels_path = paths["val_labels"]

    real_train_captions_path = paths["real_captions"]
    real_test_captions_path = paths["test_captions"]
    real_val_captions_path = paths["val_captions"]

    ciagen_logger.info(f"Moving TRAIN to {str(real_train_images_path)}")
    ciagen_logger.info(f"Moving TEST to {str(real_test_images_path)}")
    ciagen_logger.info(f"Moving VAL to {str(real_val_images_path)}")
    ciagen_logger.info(f"Using values test: {test_nb} and validation: {val_nb}")

    all_images = os.listdir(image_path)

    length = (
        val_nb + test_nb + train_nb
        if (val_nb + test_nb + train_nb) < len(all_images)
        else all_images
    )

    all_images = all_images[:length]

    counter = 0
    for file_name in tqdm(all_images, unit="img"):

        if counter > val_nb + test_nb + train_nb:
            break

        name = file_name.split(".")[0]
        img_file = name + ".jpg"
        txt_file = name + ".txt"

        image = image_path / img_file
        label = bbx_path / txt_file
        caption = caps_path / txt_file

        if (
            os.path.isfile(image)
            and os.path.isfile(label)
            and os.path.isfile(caption)
        ):
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
    prepare_coco(cfg, paths)
