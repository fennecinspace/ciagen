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
import pandas as pd
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

    # Download if necessary
    download_fer_dataset(real_path_fer_download_which, dataset_name)
    labels = load_csv_file(split_file)

    print(f"{images_download_path=}")

    if "gen" in which_dataset:
        # only copy to the images stuff
        for p in (images_download_path, images_desination_path):
            os.makedirs(p, exist_ok=True)

        for img in tqdm(os.listdir(images_download_path), unit="img"):
            orig_img_path = Path(os.path.join(images_download_path, img))
            dest_img_path = os.path.join(images_desination_path.resolve(), img)

            if not os.path.exists(dest_img_path):
                shutil.copy(orig_img_path, dest_img_path)
    else:
        # generate the labels and copy to each needed directory
        labels = list((x[0], x[1]) for x in labels[["Filename", "Emotion"]].values)
        # print(labels)
        # print(paths)

        all_real_images = list(
            x
            for x in os.listdir(images_download_path)
            if ("jpg" in x or "png" in x or "jpeg" in x)
        )
        test_nb = cfg["ml"]["test_nb"]
        val_nb = cfg["ml"]["val_nb"]
        train_nb = cfg["ml"]["train_nb"]

        max_sizes = {"train": train_nb, "val": val_nb, "test": test_nb}
        current_sizes = {"train": 0, "val": 0, "test": 0}

        caption_list = {"train": [], "val": [], "test": []}
        label_list = {"train": [], "val": [], "test": []}
        file_list = {"train": [], "val": [], "test": []}

        print("tttttttttttttttttttttttttttttttttttttttttt")
        print(f"{val_nb=} {test_nb=} {train_nb=}")
        print(all_real_images)
        total_length = (
            (val_nb + test_nb + train_nb)
            if (val_nb + test_nb + train_nb) < len(all_real_images)
            else len(all_real_images)
        )

        all_real_images = all_real_images[:total_length]

        print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
        i = 0
        for img in tqdm(all_real_images, unit="img"):
            print("=====================")
            print("here", img)
            print("ee")
            if i > 100:
                return
            i += 1

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
                    orig_img_path, os.path.join(generated_path_fer_15.resolve(), img)
                )

        elif possible_fer == "fer_gen_2_1":
            gen_path = os.path.join(real_path_fer, "Generated_2.1", "Generated_2.1")
            os.makedirs(generated_path_fer_21, exist_ok=True)

            for img in tqdm(os.listdir(gen_path)):
                orig_img_path = Path(os.path.join(gen_path, img))
                shutil.copy(
                    orig_img_path, os.path.join(generated_path_fer_21.resolve(), img)
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

                # if current_sizes[set_type] < max_sizes[set_type]:
                file_list[set_type] += [file_name]
                caption = f"{ETHNICITY_MAPPING[ethnicity]} {gender} person with {EMOTION_MAPPING[emotion]} expression"
                caption_list[set_type] += [(file_name.replace(".jpg", ".txt"), caption)]

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
