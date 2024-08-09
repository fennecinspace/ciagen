# © - 2024 Université de Mons, Multitel, Université Libre de Bruxelles, Université Catholique de Louvain

# CIA is free software. You can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License
# as published by the Free Software Foundation, either version 3
# of the License, or any later version. This program is distributed
# in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License
# for more details. You should have received a copy of the Lesser GNU
# General Public License along with this program.
# If not, see <http://www.gnu.org/licenses/>.

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

from ciagen.utils.common import logger


def cocobox2yolo(img_path, coco_box):
    I = cv2.imread(img_path)
    image_height, image_width = I.shape[0:2]

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
    logger.info(f"Attempting to create directories {[str(d) for d in dirs]}")
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

    # Only perform the work if necessary
    if not os.path.exists(path_to_data_zip):
        logger.info(f"Downloading zip images from {data_url} to {path_to_data_zip}")
        wget.download(data_url, out=str(path_to_data_zip))
    if not len(os.listdir(image_path)):
        logger.info(f"Extracting zip images to {image_path}")
        with zipfile.ZipFile(path_to_data_zip, "r") as zip_ref:
            zip_ref.extractall(str(data_path))

    if not os.path.exists(path_to_annotations_zip):
        logger.info(
            f"Downloading zip annotations from {annotations_url} to {path_to_annotations_zip}"
        )
        wget.download(annotations_url, out=str(path_to_annotations_zip))
    if not len(os.listdir(annotations_path)):
        logger.info(f"Extracting zip annotations to {annotations_path}")
        with zipfile.ZipFile(path_to_annotations_zip, "r") as zip_ref:
            zip_ref.extractall(str(data_path))

    return image_path, annotations_path, bbx_path, caps_path


class COCODataset:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    # @hydra.main(version_base=None, config_path=f"..{os.sep}conf", config_name="config")
    def __call__(self, paths: Dict[str, str | Path]) -> None:
        real_path = paths["root"]
        real_path_coco = os.path.join(real_path, "coco")

        os.makedirs(real_path_coco, exist_ok=True)

        # Download if necessary
        image_path, annotations_path, bbx_path, caps_path = download_coco(
            real_path_coco
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

        logger.info(f"Writting captions and boxes info ...")
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

        # Prepare the data for training and validation
        real_path_coco_images = os.path.join(real_path_coco, "images")
        real_path_coco_labels = os.path.join(real_path_coco, "labels")
        real_path_coco_captions = os.path.join(real_path_coco, "captions")

        os.makedirs(real_path_coco_images, exist_ok=True)
        os.makedirs(real_path_coco_captions, exist_ok=True)
        os.makedirs(real_path_coco_labels, exist_ok=True)

        test_nb = self.cfg["ml"]["test_nb"]
        val_nb = self.cfg["ml"]["val_nb"]
        train_nb = self.cfg["ml"]["train_nb"]

        logger.info(f"Moving images to {str(real_path_coco_images)}")
        logger.info(f"Moving captions to {str(real_path_coco_labels)}")
        logger.info(f"Moving boxes to {str(real_path_coco_captions)}")
        logger.info(f"Using values test: {test_nb} and validation: {val_nb}")

        # move all files
        coco_images = os.listdir(image_path)
        length = (
            val_nb + test_nb + train_nb
            if (val_nb + test_nb + train_nb) < len(coco_images)
            else coco_images
        )
        coco_images = coco_images[:length]

        counter = 0
        for file_name in tqdm(coco_images, unit="img"):

            if counter > val_nb + test_nb + train_nb:
                break

            name = file_name.split(".")[0]
            img_file = name + ".jpg"
            txt_file = name + ".txt"

            image = image_path / img_file
            label = bbx_path / txt_file
            caption = caps_path / txt_file

            # print(image, label, caption)

            if (
                os.path.isfile(image)
                and os.path.isfile(label)
                and os.path.isfile(caption)
            ):
                if counter < val_nb:
                    images_dir = Path(
                        str(real_path_coco_images).replace(
                            f"{os.sep}real{os.sep}", f"{os.sep}val{os.sep}"
                        )
                    )
                    labels_dir = Path(
                        str(real_path_coco_labels).replace(
                            f"{os.sep}real{os.sep}", f"{os.sep}val{os.sep}"
                        )
                    )
                    test_dir = Path(
                        str(real_path_coco_captions).replace(
                            f"{os.sep}real{os.sep}", f"{os.sep}val{os.sep}"
                        )
                    )
                elif counter < val_nb + test_nb:
                    images_dir = Path(
                        str(real_path_coco_images).replace(
                            f"{os.sep}real{os.sep}", f"{os.sep}test{os.sep}"
                        )
                    )
                    labels_dir = Path(
                        str(real_path_coco_labels).replace(
                            f"{os.sep}real{os.sep}", f"{os.sep}test{os.sep}"
                        )
                    )
                    test_dir = Path(
                        str(real_path_coco_captions).replace(
                            f"{os.sep}real{os.sep}", f"{os.sep}test{os.sep}"
                        )
                    )
                else:
                    images_dir = real_path_coco_images
                    labels_dir = real_path_coco_labels
                    test_dir = real_path_coco_captions

                os.makedirs(images_dir, exist_ok=True)
                os.makedirs(labels_dir, exist_ok=True)
                os.makedirs(test_dir, exist_ok=True)

                shutil.copy(image, os.path.join(images_dir, img_file))
                shutil.copy(label, os.path.join(labels_dir, txt_file))
                shutil.copy(caption, os.path.join(test_dir, txt_file))

                counter += 1
