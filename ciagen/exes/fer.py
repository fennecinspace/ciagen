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
import subprocess
import random
import zipfile
from pathlib import Path
from typing import AnyStr, Dict

import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from ciagen.utils.common import logger

EMOTION_MAPPING = {
    "Anger": "an angry",
    "Happiness": "a happy",
    "Neutral": "a neutral",
    "Sadness": "a sad",
    "Fear": "a fearful",
    "Disgust": "a disgusted",
    "Surprise": "a surprised",
    "Contempt": "a contemptuous",
}

ETHNICITY_MAPPING = {
    "White": "a white",
    "Black": "a black",
    "East Asian": "an east asian",
    "Latino_Hispanic": "a latin hispanic",
    "Southeast Asian": "a southeast asian",
}


def create_label_file_from_label_and_path(
    labels_desination_path, image_name, list_of_name_label_pairs
):
    emotion = list(filter(lambda x: x[0] == image_name, list_of_name_label_pairs))

    if len(emotion) == 0:
        logger.warning(f"No label found for {image_name}")
        return
    elif len(emotion) == 1:
        emotion = emotion[0][1]
    else:
        logger.warning(
            f"More than one label found for {image_name}: {emotion}, using a random one."
        )
        emotion = emotion[0][random.randint(0, len(emotion) - 1)]
        # return

    # emotion = emotion[0][1]

    image_pure_name = image_name.split(".")[0]
    label_name = f"{image_pure_name}.txt"
    label_desination_path = os.path.join(labels_desination_path, label_name)

    with open(label_desination_path, "w+") as f:
        f.write(emotion)
    return


def download_fer_dataset(
    data_path: Path | str,
    dataset_name: AnyStr,  # TODO change this to Str
):

    data_zip_name: str = f"{dataset_name}.zip"

    data_path = Path(data_path)
    dirs = (data_path,)
    logger.info(f"Attempting to create directories {[str(d) for d in dirs]}")
    for d in dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise OSError(
                f"Could not create data folders ({d} throws an error). E: {e}"
            )

    path_to_data_zip = data_path / data_zip_name

    # TODO: fill the windows args here
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            f"jeremiebogaert/{dataset_name}",
            "-p",
            f"{data_path}",
        ]
    )

    # TODO add a way to not extract the zip file every time
    with zipfile.ZipFile(path_to_data_zip, "r") as zip_ref:
        zip_ref.extractall(data_path)

    return True


def prepare_fer_dataset(
    which_dataset: str,
    cfg: DictConfig,
    paths: Dict[str, str | Path],
) -> None:

    real_path = paths["root"]
    real_path_fer = os.path.join(real_path, "fer")
    real_path_fer_download = os.path.join(real_path, "fer_download")

    for p in (real_path_fer, real_path_fer_download):
        os.makedirs(p, exist_ok=True)

    if which_dataset not in ("fer_real", "fer_gen_1_5", "fer_gen_2_1"):
        raise ValueError(
            "which_dataset must be 'fer_real', 'fer_gen_1_5', or 'fer_gen_2_1'",
            f"got {which_dataset}",
        )

    labels_desination_path = None
    if which_dataset == "fer_real":
        dataset_name = "face-dataset-real"
        real_path_fer_download_which = os.path.join(real_path_fer_download, "real")
        images_download_path = os.path.join(
            real_path_fer_download_which, "Real", "Real"
        )
        split_file = os.path.join(real_path_fer_download_which, "combined_real.csv")
        images_desination_path: Path = Path(
            os.path.join(real_path_fer, "train", "images")
        )
        labels_desination_path: Path = Path(
            os.path.join(real_path_fer, "train", "labels")
        )
    elif which_dataset == "fer_gen_1_5":
        dataset_name = "face-dataset-gen1-5"
        real_path_fer_download_which = os.path.join(real_path_fer_download, "sd15")
        images_download_path = os.path.join(
            real_path_fer_download_which, "Generated_1.5", "Generated_1.5"
        )
        split_file = os.path.join(
            real_path_fer_download_which, "combined_generated.csv"
        )
        images_desination_path: Path = Path(
            os.path.join(
                os.path.dirname(os.path.dirname(paths["generated"])),
                "fer",
                "sd15_crucible_mediapipe_face",
            )
        )
        metatdata_cheat_file = os.path.join(
            os.getcwd(), "ciagen", "conf", "metadata-sd15.yaml"
        )
        metadata_dest_path = os.path.join(images_desination_path, "metadata.yaml")
    elif which_dataset == "fer_gen_2_1":
        dataset_name = "face-dataset-gen2-1"
        real_path_fer_download_which = os.path.join(real_path_fer_download, "sd21")
        images_download_path = os.path.join(
            real_path_fer_download_which, "Generated_2.1", "Generated_2.1"
        )
        split_file = os.path.join(
            real_path_fer_download_which, "combined_generated.csv"
        )
        images_desination_path: Path = Path(
            os.path.join(
                os.path.dirname(os.path.dirname(paths["generated"])),
                "fer",
                "sd21_crucible_mediapipe_face",
            )
        )
        metatdata_cheat_file = os.path.join(
            os.getcwd(), "ciagen", "conf", "metadata-sd21.yaml"
        )
        metadata_dest_path = os.path.join(images_desination_path, "metadata.yaml")

    # Download if necessary
    download_fer_dataset(real_path_fer_download_which, dataset_name)
    labels = load_csv_file(split_file)

    all_images = list(
        x
        for x in os.listdir(images_download_path)
        if ("jpg" in x or "png" in x or "jpeg" in x)
    )

    # for real images create the labels
    if "gen" in which_dataset:
        logger.info(
            f"Creating pre-generated directories: {str(images_download_path), str(images_desination_path)}"
        )
        for p in (images_download_path, images_desination_path):
            os.makedirs(p, exist_ok=True)

        logger.info(
            f"Moving pre-generated images from {str(images_download_path)} to {str(images_desination_path)}"
        )
        for img in tqdm(all_images, unit="img"):
            orig_img_path = Path(os.path.join(images_download_path, img))
            dest_img_path = os.path.join(images_desination_path.resolve(), img)

            if not os.path.exists(dest_img_path):
                shutil.move(orig_img_path, dest_img_path)

        logger.info(
            f"Copying cheat metadata file from {str(metatdata_cheat_file)} to {str(metadata_dest_path)}"
        )
        shutil.copy(metatdata_cheat_file, metadata_dest_path)

    else:
        # real images remain in this directory, they are copied each time

        real_train_images_path = paths["real_images"]
        real_test_images_path = paths["test_images"]
        real_val_images_path = paths["val_images"]

        real_train_labels_path = paths["real_labels"]
        real_test_labels_path = paths["test_labels"]
        real_val_labels_path = paths["val_labels"]

        labels = list((x[0], x[1]) for x in labels[["Filename", "Emotion"]].values)

        test_nb = cfg["ml"]["test_nb"]
        val_nb = cfg["ml"]["val_nb"]
        train_nb = cfg["ml"]["train_nb"]

        # clean the directories
        for p in (
            real_train_images_path,
            real_test_images_path,
            real_val_images_path,
            real_train_labels_path,
            real_test_labels_path,
            real_val_labels_path,
        ):
            for filename in os.listdir(p):
                filepath = os.path.join(p, filename)
                if os.path.isfile(filepath) or os.path.islink(filepath):
                    os.unlink(filepath)
                elif os.path.isdir(filepath):
                    shutil.rmtree(filepath)

        logger.info(f"Moving train to {str(real_train_images_path)}")
        logger.info(f"Moving test to {str(real_test_images_path)}")
        logger.info(f"Moving val to {str(real_val_images_path)}")
        logger.info(f"Using values test: {test_nb} and validation: {val_nb}")

        total_length = (
            (val_nb + test_nb + train_nb)
            if (val_nb + test_nb + train_nb) < len(all_images)
            else len(all_images)
        )

        all_images = all_images[:total_length]

        counter = 0
        for img in tqdm(all_images, unit="img"):
            orig_img_path = Path(os.path.join(images_download_path, img))

            if counter < val_nb:
                images_desination_path = real_val_images_path
                labels_desination_path = real_val_labels_path
            elif counter < val_nb + test_nb:
                images_desination_path = real_test_images_path
                labels_desination_path = real_test_labels_path
            else:
                images_desination_path = real_train_images_path
                labels_desination_path = real_train_labels_path

            images_desination_path = Path(images_desination_path)
            labels_desination_path = Path(labels_desination_path)

            dest_img_path = os.path.join(images_desination_path.resolve(), img)

            # create the label and copy
            create_label_file_from_label_and_path(labels_desination_path, img, labels)

            # copy the image
            if not os.path.exists(dest_img_path):
                shutil.copy(orig_img_path, dest_img_path)
            counter += 1

    return True


def load_csv_file(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


class FERDataset:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    # @hydra.main(version_base=None, config_path=f"..{os.sep}conf", config_name="config")
    def __call__(self, paths: Dict[str, str | Path]) -> None:

        for which_dataset in ("fer_real", "fer_gen_1_5", "fer_gen_2_1"):
            prepare_fer_dataset(which_dataset, self.cfg, paths)

        return


@hydra.main(version_base=None, config_path=f"..{os.sep}conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Get all paths
    data = cfg["data"]
    base_path = Path(data["base"])
    RAW_DATA_PATH = Path(base_path) / data["real"]

    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    FACE_DATA_PATH = RAW_DATA_PATH / "fer"
    FACE_DATA_PATH.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
