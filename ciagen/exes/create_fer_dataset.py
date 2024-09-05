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
import random
import json

from omegaconf import DictConfig, open_dict
from pathlib import Path
from typing import List, Tuple, Dict
import csv
import yaml

from ciagen.utils.common import (
    list_images,
    create_files_list,
    create_csv_file,
    select_equal_classes,
)


def sort_based_on_score(
    image_paths: List[str], scores: List[float], direction: str = "smaller"
) -> Tuple[List[str], List[int]]:

    # Combine scores and image paths into a list of tuples
    combined_data = list(zip(scores, image_paths))
    # Sort the combined data based on scores (ascending order)
    sorted_data = sorted(
        combined_data,
        key=lambda x: x[0],
        reverse=False if direction == "smaller" else True,
    )
    # Extract sorted scores and image paths
    sorted_scores, sorted_image_paths = zip(*sorted_data)
    return sorted_image_paths, sorted_scores


def select_equal_classes_path(
    total_captions: List[Path], synth_images: List[str], nb_synth_images: int
) -> List[str]:
    """
    Selects synthetic images such that classes are balanced, based on their captions.
    """
    # Create a dictionary to store images by class
    try:
        assert len(synth_images) >= nb_synth_images
    except AssertionError as e:
        raise ValueError(
            f"The amount of images to select should be lower than the amount of available images, but {nb_synth_images} > {len(synth_images)}"
        )

    class_to_images: Dict[str, List[str]] = {}

    # Map captions to corresponding images
    for caption_path in total_captions:
        with open(caption_path, "r") as file:
            class_name = file.readline().strip()

        base_name = caption_path.stem  # e.g., "0" from "0.txt"
        corresponding_image = next(
            (img for img in synth_images if Path(img).stem == f"{base_name}_1"), None
        )

        if corresponding_image:
            if class_name not in class_to_images:
                class_to_images[class_name] = []
            class_to_images[class_name].append(corresponding_image)

    # Calculate number of images per class
    num_classes = len(class_to_images)
    if num_classes == 0:
        raise ZeroDivisionError(
            "No classes were found, cannot divide images among zero classes."
        )

    images_per_class = max(1, nb_synth_images // num_classes)

    selected_images = []

    for class_name, images in class_to_images.items():
        # Shuffle images for random selection
        random.shuffle(images)

        # Select the required number of images for this class
        selected_images += images[:images_per_class]

    # If there is a remainder, distribute remaining images to random classes
    remaining_images = nb_synth_images - len(selected_images)
    if remaining_images > 0:
        available_classes = [
            cls
            for cls in class_to_images
            if len(class_to_images[cls]) > images_per_class
        ]
        for _ in range(remaining_images):
            class_name = random.choice(available_classes)
            selected_images.append(class_to_images[class_name].pop())

    return selected_images


def create_csv_file_path(
    train_images: List[str],
    val_images: List[str],
    test_images: List[str],
    real_train_captions: List[Path],
    val_captions: List[Path],
    test_captions: List[Path],
    output_csv: str,
) -> None:
    """
    Creating a CSV file that contains filepath, class, and type of dataset (train, test, val).

    Args:
        train_images (List[str]): Path list of training images.
        val_images (List[str]): Path list of val images.
        test_images (List[str]): Path list of test images.
        real_train_captions (List[Path]): Path list of train captions.
        val_captions (List[Path]): Path list of val captions.
        test_captions (List[Path]): Path list of test captions.
        output_csv (str): Path where the CSV file is created.
    """

    def extract_class_from_caption(caption_path: Path) -> str:
        with open(caption_path, "r") as file:
            return file.readline().strip()

    def map_captions_to_images(captions: List[Path]) -> Dict[str, str]:
        return {
            caption.stem: extract_class_from_caption(caption) for caption in captions
        }

    train_name_to_caption = map_captions_to_images(real_train_captions)
    val_name_to_caption = map_captions_to_images(val_captions)
    test_name_to_caption = map_captions_to_images(test_captions)

    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["Filename", "Emotion", "Dataset"])

        for image in train_images:
            base_name = os.path.splitext(os.path.basename(image))[0].split("_")[
                0
            ]  # Extract base name from image filename
            emotion = train_name_to_caption.get(base_name, "Unknown")
            writer.writerow([image, emotion, "train"])

        for image in val_images:
            base_name = os.path.splitext(os.path.basename(image))[
                0
            ]  # Extract base name from image filename
            emotion = val_name_to_caption.get(base_name, "Unknown")
            writer.writerow([image, emotion, "val"])

        for image in test_images:
            base_name = os.path.splitext(os.path.basename(image))[
                0
            ]  # Extract base name from image filename
            emotion = test_name_to_caption.get(base_name, "Unknown")
            writer.writerow([image, emotion, "test"])


class CreateMixedFERDataset:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    # @hydra.main(version_base=None, config_path=f"..{os.sep}conf", config_name="config")
    def __call__(self, paths: Dict[str, str | Path]) -> None:
        """
        Construct the txt file containing real and synthetic data
        """

        augmentation_percent = self.cfg["ml"]["augmentation_percent"]
        train_nb = self.cfg["ml"]["train_nb"]
        val_nb = self.cfg["ml"]["val_nb"]
        test_nb = self.cfg["ml"]["test_nb"]
        sample = self.cfg["ml"]["sampling"]
        seed = 42
        formats = self.cfg["data"]["image_formats"]

        if not os.path.isdir(paths["mixed_yamls_folder_path"]):
            os.makedirs(paths["mixed_yamls_folder_path"])

        data_csv_path = Path(paths["mixed_yamls_folder_path"]) / "train_dataset.csv"

        real_images_path = Path(paths["real_images"].replace("/fer/", "/fer_real/"))
        val_images_path = Path(paths["val_images"].replace("/fer/", "/fer_real/"))
        test_images_path = Path(Path(paths["test_images"].replace("/fer/", "/fer_real/")))

        real_train_labels = list(Path(paths["real_labels"].replace("/fer/", "/fer_real/")).glob("*.txt"))
        real_test_labels = list(Path(paths["test_labels"].replace("/fer/", "/fer_real/")).glob("*.txt"))
        real_val_labels = list(Path(paths["val_labels"].replace("/fer/", "/fer_real/")).glob("*.txt"))

        print(paths["real_images"])
        print(paths["real_labels"])
        synth_images_dir = Path(paths["generated"])
        print(synth_images_dir)
        
        # TODO change total_captions to total_labels
        total_captions = real_train_labels + real_test_labels + real_val_labels

        real_images = list_images(real_images_path, formats, train_nb)
        val_images = list_images(val_images_path, formats, val_nb)
        test_images = list_images(test_images_path, formats, test_nb)

        synth_images = list_images(synth_images_dir, formats)
        print(real_images[0], synth_images[0])

        synth_images_full = list_images(synth_images_dir, formats)

        # TODO synth images should be train_nb + val_nb + test_nb total images corresponding to real ones

        real_img_dict = {}
        for img in real_images:

            # TODO make this windows compliant to please hamed
            real_img_id = img.split('/')[-1].split('_')[0].replace(".png", "").replace(".jpg", "")
            real_img_dict[real_img_id] = True

        synth_images = []
        for img in synth_images_full:
            # TODO make this windows compliant to please hamed
            synth_img_id = img.split('/')[-1].split('_')[0].replace(".png", "").replace(".jpg", "")
            if synth_img_id in real_img_dict:
                synth_images += [img]


        # if sample['enable']:
        #     txt_dir = txt_dir / (sample['metric'] + '_' + sample['sample'])

        # shuffle images
        random.Random(seed).shuffle(synth_images)
        random.Random(seed).shuffle(real_images)

        if self.cfg["ml"]["keep_training_size"]:
            nb_real_images = int(len(real_images) * (1 - augmentation_percent))
        else:
            nb_real_images = train_nb
        nb_synth_images = int(len(real_images) * augmentation_percent)

        print(f"Total captions: {len(total_captions)}")
        print(f"Synthetic images: {len(synth_images)}")
        print(total_captions[0], synth_images[0], nb_synth_images)

        if self.cfg["ml"]["with_filtering"]:
            # 1) Load metadata
            dataset_name = self.cfg["data"]["base"]
            cn_name = self.cfg["model"]["cn_use"]

            metadata_path = os.path.join("data", "generated", dataset_name, cn_name)
            metadata_file = os.path.join(metadata_path, "metadata.yaml")
            filtering_metric = self.cfg["ml"]["filtering_metric"]

            # TODO: make the script be able to use all feature extractors
            preferred_fe = self.cfg["ml"]["preferred_fe"]
            with open(metadata_file, "r") as f:
                metadata_dict = yaml.safe_load(f)

            # 2) read the filtered images
            filtered_images = metadata_dict["results"]["filtering"][filtering_metric][preferred_fe]

            # 3) use a map_reduce to filter captions and synth_images
            #synth_images = [i.replace('/fer/', '/fer_real/') for i in list(filtered_images.keys())]
            synth_images =  list(filtered_images.keys())
            #nb_synth_images = len(synth_images)

        print(total_captions[0], synth_images[0], nb_synth_images)
        synth_images = select_equal_classes(total_captions, synth_images, nb_synth_images)
        #synth_images = select_equal_classes_path(total_captions, synth_images, nb_synth_images)

        train_images = real_images + synth_images

        create_csv_file_path(
            train_images=train_images,
            val_images=val_images,
            test_images=test_images,
            real_train_captions=real_train_labels,
            val_captions=real_val_labels,
            test_captions=real_test_labels,
            output_csv=data_csv_path,
        )

        print(f"Training csv files created in : {paths['mixed_yamls_folder_path']}")
        print(f"Using {train_nb} Real Images from : ", real_images_path)
        print(f"Using Synthetic Images from : ", synth_images_dir)
        print(f"Using {val_nb} Validation Images from : ", val_images_path)
        print(f"Using {test_nb} Test Images from : ", test_images_path)

        return data_csv_path
