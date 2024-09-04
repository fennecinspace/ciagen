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

import hydra
import os
import shutil
from tqdm import tqdm
import zipfile

from omegaconf import DictConfig
from pathlib import Path

from typing import Dict, AnyStr

import os
import subprocess

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


def download_fer(
    data_path: Path | str,
    dataset_name: AnyStr,  # TODO change this to Str
    captions_path: str = "Captions",
    sentences_path: str = "Sentences",
    images_path: str = "Images",
    annotations_path: str = "Annotations",
    labels_path: str = "Labels",
):

    data_zip_name: str = f"{dataset_name}.zip"

    data_path = Path(data_path)
    image_path: Path = data_path / images_path
    annotations_path: Path = data_path / annotations_path
    caps_path: Path = data_path / captions_path
    sentences_path: Path = data_path / sentences_path
    labels_path: Path = data_path / labels_path

    dirs = data_path, image_path, annotations_path, caps_path, sentences_path
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

    return image_path, annotations_path, sentences_path, caps_path, labels_path


class FERDataset:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    # @hydra.main(version_base=None, config_path=f"..{os.sep}conf", config_name="config")
    def __call__(self, paths: Dict[str, str | Path]) -> None:

        real_path = paths["root"]
        real_path_fer = os.path.join(real_path, "fer")

        os.makedirs(real_path_fer, exist_ok=True)

        generated_path_fer = Path(
            os.path.join(
                os.path.dirname(os.path.dirname(paths["generated"])),
                "fer",
                self.cfg["model"]["cn_use"],
            )
        )

        real_path_fer_train_images = Path(
            os.path.join(real_path_fer, "train", "images")
        )
        generated_path_fer_15 = Path(
            os.path.join(
                os.path.dirname(os.path.dirname(paths["generated"])),
                "fer",
                "sd15_crucible_mediapipe_face",
            )
        )

        generated_path_fer_21 = Path(
            os.path.join(
                os.path.dirname(os.path.dirname(paths["generated"])),
                "fer",
                "sd21_crucible_mediapipe_face",
            )
        )

        possible_fer = self.cfg["data"]["base"]
        if possible_fer not in ("fer_real", "fer_gen_1_5", "fer_gen_2_1"):
            raise ValueError(f"Unknown FER dataset base: {possible_fer}")

        if possible_fer == "fer_real":
            dataset_name = "face-dataset-real"
        elif possible_fer == "fer_gen_1_5":
            dataset_name = "face-dataset-gen1-5"
        elif possible_fer == "fer_gen_2_1":
            dataset_name = "face-dataset-gen2-1"

        # Download if necessary
        images_path, _annotations_path, _sentences_path, _caps_path, labels_path = (
            download_fer(real_path_fer, dataset_name)
        )
        if "gen" in dataset_name:
            split_file = os.path.join(real_path_fer, "combined_generated.csv")
        else:
            split_file = os.path.join(real_path_fer, "combined_real.csv")

        test_nb = self.cfg["ml"]["test_nb"]
        val_nb = self.cfg["ml"]["val_nb"]
        train_nb = self.cfg["ml"]["train_nb"]

        max_sizes = {"train": train_nb, "val": val_nb, "test": test_nb}
        current_sizes = {"train": 0, "val": 0, "test": 0}

        caption_list = {"train": [], "val": [], "test": []}
        label_list = {"train": [], "val": [], "test": []}
        file_list = {"train": [], "val": [], "test": []}

        if possible_fer == "fer_real":
            gen_path = os.path.join(real_path_fer, "Real", "Real")
            os.makedirs(real_path_fer_train_images, exist_ok=True)
            for img in tqdm(os.listdir(gen_path)):
                orig_img_path = Path(os.path.join(gen_path, img))
                shutil.copy(
                    orig_img_path,
                    os.path.join(real_path_fer_train_images.resolve(), img),
                )
        elif possible_fer == "fer_gen_1_5":
            gen_path = os.path.join(real_path_fer, "Generated_1.5", "Generated_1.5")
            os.makedirs(generated_path_fer_15, exist_ok=True)
            for img in tqdm(os.listdir(gen_path)):
                orig_img_path = Path(os.path.join(gen_path, img))
                shutil.copy(
                    orig_img_path, os.path.join(generated_path_fer.resolve(), img)
                )

        elif possible_fer == "fer_gen_2_1":
            gen_path = os.path.join(real_path_fer, "Generated_2.1", "Generated_2.1")
            os.makedirs(generated_path_fer_21, exist_ok=True)

            for img in tqdm(os.listdir(gen_path)):
                orig_img_path = Path(os.path.join(gen_path, img))
                shutil.copy(
                    orig_img_path, os.path.join(generated_path_fer.resolve(), img)
                )

        with open(split_file, "r") as f:
            for line_nbr, line in enumerate(f):
                if line_nbr == 0:
                    continue
                if all(
                    [current_sizes[i] == max_sizes[i] for i in current_sizes.keys()]
                ):
                    break

                line = line.replace("\n", "")

                try:
                    [file_name, emotion, gender, ethnicity, set_type] = line.split(",")
                except Exception as e:
                    raise ValueError(
                        f"Error with line (too long or too short) {line_nbr}: {e}"
                    )

                if set_type == "test" and current_sizes["val"] < max_sizes["val"]:
                    set_type = "val"

                if current_sizes[set_type] < max_sizes[set_type]:
                    file_list[set_type] += [file_name]
                    caption = f"{ETHNICITY_MAPPING[ethnicity]} {gender} person with {EMOTION_MAPPING[emotion]} expression"
                    caption_list[set_type] += [
                        (file_name.replace(".jpg", ".txt"), caption)
                    ]

                    label = emotion
                    label_list[set_type] += [(file_name.replace(".jpg", ".txt"), label)]

                    current_sizes[set_type] += 1

        images_path = {
            "train": paths["real_images"],
            "val": paths["val_images"],
            "test": paths["test_images"],
        }
        labels_path = {
            "train": paths["real_labels"],
            "val": paths["val_labels"],
            "test": paths["test_labels"],
        }
        captions_path = {
            "train": paths["real_captions"],
            "val": paths["val_captions"],
            "test": paths["test_captions"],
        }

        for set_type in ["train", "val", "test"]:
            image_set_path = images_path[set_type]
            label_set_path = labels_path[set_type]
            caption_set_path = captions_path[set_type]

            # for img in file_list[set_type]:
            # if self.cfg["data"]["base"] == "fer_real":
            #     orig_img_path = Path(
            #         os.path.join(real_path_fer, "Real", "Real", img)
            #     )
            # elif self.cfg["data"]["base"] == "fer_gen_1_5":
            #     orig_img_path = Path(
            #         os.path.join(
            #             real_path_fer, "Generated_1.5", "Generated_1.5", img
            #         )
            #     )
            # else:
            #     orig_img_path = Path(
            #         os.path.join(
            #             real_path_fer, "Generated_2.1", "Generated_2.1", img
            #         )
            #     )
            for lab_file, lab in label_list[set_type]:
                with open(os.path.join(label_set_path, lab_file), "w+") as f:
                    f.write(lab)

            for cap_file, cap in caption_list[set_type]:
                with open(os.path.join(caption_set_path, cap_file), "w+") as f:
                    f.write(cap)

        if possible_fer == "fer_gen_1_5":
            shutil.copy(
                os.path.join(os.getcwd(), "ciagen", "conf", "metadata-sd15.yaml"),
                os.path.join(generated_path_fer_15, "metadata.yaml"),
            )

        elif possible_fer == "fer_gen_2_1":
            shutil.copy(
                os.path.join(os.getcwd(), "ciagen", "conf", "metadata-sd21.yaml"),
                os.path.join(generated_path_fer_21, "metadata.yaml"),
            )


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
