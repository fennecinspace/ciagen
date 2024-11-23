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
import json
from omegaconf import DictConfig
from tqdm import tqdm

from ciagen.utils.common import ciagen_logger


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


def download_mocs(
    data_path: Path
):
    # Only using train images and labels as whole dataset because they are already 19 000 samples
    images_zips_links = [
        "https://nextcloud.ig.umons.ac.be/s/AoyiR5CHPMEPFxZ/download/instances_train.zip",
        # "https://nextcloud.ig.umons.ac.be/s/kyKmyi7PEwxHrqN/download/instances_val.zip",
        # "https://nextcloud.ig.umons.ac.be/s/pgBBCy7x3HXAtFk/download/instances_test.zip", # Don't use test images they are bad
    ]

    labels_links = [
        "https://nextcloud.ig.umons.ac.be/s/Rrz7DaobKzJBXcm/download/instances_train.json",
        # "https://nextcloud.ig.umons.ac.be/s/wmwbgjsE2egM6zk/download/instances_val.json",
        # "https://nextcloud.ig.umons.ac.be/s/RYRCjKFKc3HzMqr/download/image_info_test.json", # Don't use test images they are bad
    ]

    all_images_path = Path(data_path) / "Images"
    all_bbox_path = Path(data_path) / "Bbox"
    all_segmentation_path = Path(data_path) / "Segmentation"

    all_images_path.mkdir(parents=True, exist_ok=True)
    all_bbox_path.mkdir(parents=True, exist_ok=True)
    all_segmentation_path.mkdir(parents=True, exist_ok=True)

    image_folders = []
    # Downloading Images
    for data_url in images_zips_links:
        zip_name = data_url.split('/')[-1]
        path_to_data_zip = Path(data_path) / zip_name

        folder_name = zip_name.rstrip('.zip')
        path_to_folder = Path(data_path) / folder_name
        image_folders += [path_to_folder]

        # Only perform the work if necessary
        if not os.path.exists(path_to_data_zip):
            ciagen_logger.info(
                f"Downloading zip images from {data_url} to {path_to_data_zip}"
            )
            wget.download(data_url, out=str(path_to_data_zip))

            # unzip
            with zipfile.ZipFile(path_to_data_zip, "r") as zip_ref:
                zip_ref.extractall(str(data_path))

    labels_jsons = []
    # Downloading Labels
    for data_url in labels_links:
        json_file_name = data_url.split('/')[-1]
        path_to_json = Path(data_path) / json_file_name
        labels_jsons += [path_to_json]

        # Only perform the work if necessary
        if not os.path.exists(path_to_json):
            ciagen_logger.info(
                f"Downloading zip images from {data_url} to {path_to_json}"
            )
            wget.download(data_url, out=str(path_to_json))


    # Moving all images to a single folder
    for folder in image_folders:
        ciagen_logger.info(
            f"Moving images from {folder} to {all_images_path}"
        )
        for file_name in tqdm(os.listdir(folder)):
            if file_name.endswith(".jpg"):
                # Define full file paths
                source_file = os.path.join(folder, file_name)
                destination_file = os.path.join(all_images_path, file_name)

                # Move the file
                shutil.move(source_file, destination_file)


    nb_images = len(os.listdir(all_images_path))

    if len(os.listdir(all_bbox_path)) < nb_images or len(os.listdir(all_segmentation_path)) < nb_images:
        # Converting all json annotations to YOLO labels
        for labels_json_path in labels_jsons:
            ciagen_logger.info(
                f"Converting JSON labels from {labels_json_path} to {all_bbox_path} and {all_segmentation_path}"
            )
            with open(labels_json_path, 'r') as file:
                data = json.load(file)

                for i in tqdm(range(len(data['images']))):
                    annotation_file_content = ""
                    bbox_annotation_file_content = ""

                    file_name, ext = data['images'][i]['file_name'].split('.')
                    image_id = data['images'][i]['id']

                    for j in range(len(data['annotations'])):
                        if data['annotations'][j]['image_id'] == image_id:
                            segmentation = data['annotations'][j]['segmentation']
                            bbox = data['annotations'][j]['bbox']
                            category_id = data['annotations'][j]['category_id']

                            annotation_file_content = annotation_file_content + f'{category_id} {" ".join(map(str, segmentation[0]))}' + '\n'
                            bbox_annotation_file_content = bbox_annotation_file_content + f'{category_id} {" ".join(map(str, bbox))}'  + '\n'

                    with open( all_bbox_path / f'{file_name}.txt', 'w') as f:
                        f.write(bbox_annotation_file_content)

                    with open(all_segmentation_path / f'{file_name}.txt', 'w') as f:
                        f.write(annotation_file_content)

    return all_images_path, all_bbox_path, all_segmentation_path




class MOCSDataset:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        self.classes = [
            "Worker",
            "Static crane",
            "Hanging head",
            "Crane",
            "Roller",
            "Bulldozer",
            "Excavator",
            "Truck",
            "Loader",
            "Pump truck",
            "Concrete mixer",
            "Pile driving",
            "Other vehicle",
        ]

    # @hydra.main(version_base=None, config_path=f"..{os.sep}conf", config_name="config")
    def __call__(self, paths: Dict[str, str | Path]) -> None:
        real_path = paths["root"]
        real_path_dataset = os.path.join(real_path, "mocs")

        os.makedirs(real_path_dataset, exist_ok=True)

        # Download if necessary
        image_path, bbox_labels_dir, segmentation_labels_dir = download_mocs(
            real_path_dataset
        )

        origin_labels_dir = bbox_labels_dir

        test_nb = self.cfg["ml"]["test_nb"]
        val_nb = self.cfg["ml"]["val_nb"]
        train_nb = self.cfg["ml"]["train_nb"]

        real_train_images_path = Path(paths["real_images"])
        real_test_images_path = Path(paths["test_images"])
        real_val_images_path = Path(paths["val_images"])

        real_train_labels_path = Path(paths["real_labels"])
        real_test_labels_path = Path(paths["test_labels"])
        real_val_labels_path = Path(paths["val_labels"])

        # real_train_captions_path = paths["real_captions"]
        # real_test_captions_path = paths["test_captions"]
        # real_val_captions_path = paths["val_captions"]

        ciagen_logger.info(f"Moving TRAIN to {str(real_train_images_path)}")
        ciagen_logger.info(f"Moving TEST to {str(real_test_images_path)}")
        ciagen_logger.info(f"Moving VAL to {str(real_val_images_path)}")
        ciagen_logger.info(f"Using values test: {test_nb} and validation: {val_nb}")

        # move all files
        all_images = os.listdir(image_path)

        # print("=>", len(all_images))

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
            label = origin_labels_dir / txt_file

            if (
                os.path.isfile(image)
                and os.path.isfile(label)
            ):
                if counter < val_nb:
                    images_dir = real_val_images_path
                    labels_dir = real_val_labels_path
                elif counter < val_nb + test_nb:
                    images_dir = real_test_images_path
                    labels_dir = real_test_labels_path
                else:
                    images_dir = real_train_images_path
                    labels_dir = real_train_labels_path

                shutil.copy(image, os.path.join(images_dir, img_file))
                shutil.copy(label, os.path.join(labels_dir, txt_file))

                counter += 1
