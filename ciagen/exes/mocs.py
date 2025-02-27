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



def download_mocs(
    data_path: Path,
    size:str = 'extra_small',
):
    dataset_links = {
        'extra_small': 'https://nextcloud.ig.umons.ac.be/s/tfoeSBoDDE3mzHp/download/MOCS_extra_small.zip', # 1000
        'small': 'https://nextcloud.ig.umons.ac.be/s/SWfyj4wAqCtRGYy/download/MOCS_small.zip', # 3500
        'medium': '', # 7000
        'large': '', # 10 000
        'full': '', # +15 000
    }

    data_url = dataset_links[size]

    all_images_path = Path(data_path) / "Images"
    all_bbox_path = Path(data_path) / "Detection"
    all_segmentation_path = Path(data_path) / "Segmentation"
    all_captions_path = Path(data_path) / "Captions"

    # Download Dataset
    path_to_data_zip = Path(data_path) / f'MOCS_{size}.zip'

    # Only perform the work if necessary
    if not os.path.exists(path_to_data_zip):
        ciagen_logger.info(
            f"Downloading zip images from {data_url} to {path_to_data_zip}"
        )
        wget.download(data_url, out=str(path_to_data_zip))

        # unzip
        with zipfile.ZipFile(path_to_data_zip, "r") as zip_ref:
            zip_ref.extractall(str(data_path))

    return all_images_path, all_bbox_path, all_segmentation_path, all_captions_path




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
        image_path, bbox_labels_dir, segmentation_labels_dir, all_captions_path = download_mocs(
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

        real_train_captions_path = Path(paths["real_captions"])
        real_test_captions_path = Path(paths["test_captions"])
        real_val_captions_path = Path(paths["val_captions"])

        ciagen_logger.info(f"Moving TRAIN to {str(real_train_images_path)}")
        ciagen_logger.info(f"Moving TEST to {str(real_test_images_path)}")
        ciagen_logger.info(f"Moving VAL to {str(real_val_images_path)}")
        ciagen_logger.info(f"Using values test: {test_nb} and validation: {val_nb}")

        # move all files
        all_images = os.listdir(image_path)

        # [WARNING / NOT BEST PRACTICE] FILTERING : getting only images that have captions
        all_captions = os.listdir(all_captions_path.absolute())

        all_captions = list( map(lambda x: x.split('.')[0], all_captions))

        all_images = list(filter(lambda x: x.split('.')[0] in all_captions, all_images))


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
            caption = all_captions_path / txt_file

            if (
                os.path.isfile(image)
                and os.path.isfile(label)
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
