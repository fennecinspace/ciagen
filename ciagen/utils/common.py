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

import glob
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import yaml

import random
import csv

import torch
from torchvision import transforms
from diffusers.utils import load_image
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from omegaconf import DictConfig
from PIL import Image


FORMAT = "%(asctime)s %(clientip)-16s %(user)-8s %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger()


def generate_all_paths(cfg: DictConfig) -> Dict[str, str | Path]:
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

    # all data paths
    train_path = os.path.join(real_dataset, "train")
    test_path = os.path.join(real_dataset, "test")
    val_path = os.path.join(real_dataset, "val")

    # train
    real_train_images_path = os.path.join(train_path, "images")
    real_train_labels_path = os.path.join(train_path, "labels")
    real_train_captions_path = os.path.join(train_path, "captions")

    # test
    real_test_images_path = os.path.join(test_path, "images")
    real_test_labels_path = os.path.join(test_path, "labels")
    real_test_captions_path = os.path.join(test_path, "captions")

    # val
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


def get_model_config(name: str, l: List[Dict]) -> Optional[str]:
    for small_dict in l:
        if name in small_dict:
            return small_dict[name]
    return None


def read_caption(caption_path: str) -> List[str]:
    with open(caption_path, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines


def find_common_prefix(str_list: List[str]):
    return os.path.commonprefix(str_list)


def find_common_suffix(str_list: List[str]):
    str_list_inv = [x[::-1] for x in str_list]
    return find_common_prefix(str_list_inv)


def draw_landmarks_on_image(rgb_image, detection_result, mode: str = "default"):
    if mode not in ("default", "binary"):
        raise ValueError(f"Unkown mode: {mode}")

    face_landmarks_list = detection_result.face_landmarks
    if mode == "binary":
        rgb_image = Image.new(
            "RGB", (rgb_image.shape[0], rgb_image.shape[1]), (0, 0, 0)
        )
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for face_landmarks in face_landmarks_list:
        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in face_landmarks
            ]
        )

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

    return annotated_image


def plot_face_blendshapes_bar_graph(face_blendshapes):
    # Extract the face blendshapes category names and scores.
    face_blendshapes_names, face_blendshapes_scores = [], []
    for face_blendshapes_category in face_blendshapes:
        face_blendshapes_names.append(face_blendshapes_category.category_name)
        face_blendshapes_scores.append(face_blendshapes_category.category_score)

    # The blendshapes are ordered in decreasing score value.
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    _, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(
        face_blendshapes_ranks,
        face_blendshapes_scores,
        label=[str(x) for x in face_blendshapes_ranks],
    )
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(
            patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top"
        )

    ax.set_xlabel("Score")
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()


def normalizer(image: Image) -> Image:
    """Normalize an image pixel values between [0 - 255]"""

    img = np.array(image)
    return cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)


def contains_only_one_substring(input_string, substring_list):
    count = 0

    for substring in substring_list:
        if substring in input_string:
            count += 1

    return count == 1


def contains_word(string, words):
    for word in words:
        if word.lower() in string.lower():
            return True


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (np.array): Array representing the first bounding box in YOLOv5 format (x_center, y_center, width, height).
        box2 (np.array): Array representing the second bounding box in YOLOv5 format (x_center, y_center, width, height).

    Returns:
        float: IoU score between the two bounding boxes.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate coordinates of the intersection rectangle
    x_left = max(x1 - w1 / 2, x2 - w2 / 2)
    y_top = max(y1 - h1 / 2, y2 - h2 / 2)
    x_right = min(x1 + w1 / 2, x2 + w2 / 2)
    y_bottom = min(y1 + h1 / 2, y2 + h2 / 2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Calculate IoU
    iou = intersection_area / (box1_area + box2_area - intersection_area)
    return iou


def bbox_min_max_to_center_dims(x_min, x_max, y_min, y_max, image_width, image_height):
    x_center = (x_min + x_max) / 2.0 / image_width
    y_center = (y_min + y_max) / 2.0 / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    return x_center, y_center, width, height


def create_files_list(image_files, txt_file_path):
    with open(txt_file_path, "w") as f:
        f.write("\n".join(image_files))


def list_images(images_path: Path, formats: List[str], limit: int = None):
    images = []
    for format in formats:
        images += [*glob.glob(str(images_path.absolute()) + f"/*.{format}")]
    return images[:limit]


def create_yaml_file(save_path: Path, train: Path, val: Path, test: Path):
    """
    Construct the yaml file

    :param pathlib.Path txt_dir: path used to create the txt files
    :param pathlib.Path yaml_dir: path used to create the yaml file

    :return: None
    :rtype: NoneType
    """

    yaml_file = {
        "train": str(train.absolute()),
        "val": str(val.absolute()),
        "test": str(test.absolute()),
        "names": {0: "person"},
    }

    with open(save_path, "w") as file:
        yaml.dump(yaml_file, file)

def select_equal_classes(total_captions: List[Path], synth_images: List[Path], nb_synth_images: int) -> List[Path]:
            """
            Selects synthetic images such that classes are balanced, based on their captions.

            Args:
                total_captions (List[Path]): List of file paths for captions (txt files).
                synth_images (List[Path]): List of synthetic image paths (png files).
                nb_synth_images (int): Total number of synthetic images needed.

            Returns:
                List[Path]: A list of selected synthetic image paths, balanced by class.
            """
            # Create a dictionary to store images by class
            class_to_images: Dict[str, List[Path]] = {}
            
            # Map captions to corresponding images
            for caption_path in total_captions:
                # Read the class from the caption file
                with open(caption_path, 'r') as file:
                    class_name = file.readline().strip()
                
                # Corresponding image name (e.g., 0_1.png for 0.txt)
                base_name = caption_path.stem  # e.g., "0" from "0.txt"
                corresponding_image = next((img for img in synth_images if img.stem == f"{base_name}_1"), None)
                
                if corresponding_image:
                    if class_name not in class_to_images:
                        class_to_images[class_name] = []
                    class_to_images[class_name].append(corresponding_image)
            
            # Calculate number of images per class
            num_classes = len(class_to_images)
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
                available_classes = [cls for cls in class_to_images if len(class_to_images[cls]) > images_per_class]
                for _ in range(remaining_images):
                    class_name = random.choice(available_classes)
                    selected_images.append(class_to_images[class_name].pop())
            
            return selected_images

def create_csv_file(train_images: List[Path], val_images: List[Path], 
                            test_images: List[Path], real_train_captions: List[Path],
                            val_captions: List[Path], test_captions: List[Path], 
                            output_csv: Path) -> None:
            """
            creating csv file that contains filepath,class, and type od dataset(train,test,val).

            Args:
                train_images (List[Path]): path list of training images.
                val_images (List[Path]): path list of val images.
                test_images (List[Path]): path list of test images.
                real_train_captions (List[Path]): path list of train captions
                val_captions (List[Path]): path list of val captions
                test_captions (List[Path]): path list of test captions
                output_csv (Path): path where the csv file is created
            """
            
            def extract_class_from_caption(caption_path: Path) -> str:
                with open(caption_path, 'r') as file:
                    return file.readline().strip()

            def map_captions_to_images(captions: List[Path]) -> Dict[str, str]:
                return {caption.stem: extract_class_from_caption(caption) for caption in captions}

            train_name_to_caption = map_captions_to_images(real_train_captions)
            val_name_to_caption = map_captions_to_images(val_captions)
            test_name_to_caption = map_captions_to_images(test_captions)

            with open(output_csv, mode='w', newline='') as file:
                writer = csv.writer(file)
                
                writer.writerow(['Filename', 'Emotion', 'Dataset'])
                
            
                for image in train_images:
                    base_name = image.stem.split('_')[0]  
                    emotion = train_name_to_caption.get(base_name, 'Unknown')
                    writer.writerow([image.name, emotion, 'train'])
                
                for image in val_images:
                    image_name = image.stem
                    emotion = val_name_to_caption.get(image_name, 'Unknown')
                    writer.writerow([image.name, emotion, 'val'])
                
                for image in test_images:
                    image_name = image.stem
                    emotion = test_name_to_caption.get(image_name, 'Unknown')
                    writer.writerow([image.name, emotion, 'test'])


def load_images_from_directory(
    directory: Union[str, Path],
    formats: List[str] = ["png", "jpg", "jpeg"],
    to_tensors: bool = False,
    ptd: bool = False,
) -> List[str]:
    if type(directory) == str:
        directory = Path(directory)

    images_paths = list_images(directory, formats)
    images_paths.sort()

    if ptd:
        image_names = []
    images = []

    if to_tensors:
        transform = transforms.Compose(
            [
                transforms.ToTensor()  # Converts the image to a PyTorch tensor (scales pixel values to [0, 1])
            ]
        )
    else:
        transform = lambda image: image

    for image_path in images_paths:
        try:
            image = load_image(image_path)

            if to_tensors:
                image = transform(image)

            if ptd:
                image_names.extend([image_path.split("/")[-1]])
            images.append(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
    if ptd:
        return (images, image_names)
    else:
        return images
