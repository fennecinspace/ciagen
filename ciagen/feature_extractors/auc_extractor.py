import os
import re
import json
from PIL.Image import Image
import hydra

from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig
import numpy as np
import torch
import torchvision
from typing import List

from ciagen.utils.common import logger
from ciagen.feature_extractors.abc_feature_extractor import (
    FeatureExtractor,
    SampleT,
)

from feat import Detector

detector = Detector()


class AUExtractor(FeatureExtractor):
    def _extract(
        self,
        samples: List[torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray,
        **kwargs,
    ) -> List[SampleT] | SampleT:

        # The py-feat detector does not expose a API to extract the Action Unit
        # from tensor or arrays directly.
        # We are simply reusing their inner code here.

        face_model_kwargs = kwargs.pop("face_model_kwargs", dict())
        landmark_model_kwargs = kwargs.pop("landmark_model_kwargs", dict())
        au_model_kwargs = kwargs.pop("au_model_kwargs", dict())
        face_detection_threshold = kwargs.pop("face_detection_threshold", 0.5)

        aus_all = []

        def analyse_once(sample):
            faces = detector.detect_faces(
                sample, face_detection_threshold, **face_model_kwargs
            )
            landmarks = detector.detect_landmarks(
                sample, detected_faces=faces, **landmark_model_kwargs
            )
            aus = detector.detect_aus(sample, landmarks, **au_model_kwargs)

            # AD-HOC solution for the moment
            # the action unit is capable of detecting several faces, we are
            # runing with only one for the moment
            if not len(aus):
                aus = [np.zeros((1, 20))]
            else:
                if not len(aus[0]):
                    aus = [np.zeros((1, 20))]
                else:
                    aus = [aus[0]]
            aus = [torch.from_numpy(aus[0][0])]
            return aus

        if not isinstance(samples, list):
            aus_all.extend(analyse_once(samples))
        else:
            for sample in tqdm(samples):
                aus_all.extend(analyse_once(sample))

        return aus_all

    def transform_from_array(self, array: np.ndarray) -> SampleT:
        return torch.from_numpy(array)

    def transform_from_image(self, image: Image) -> SampleT:
        return torchvision.transforms.functional.pil_to_tensor(image)

    def transform_from_tensor(self, tensor: torch.Tensor) -> SampleT:
        return tensor


# Function to extract AU differences between real and generated images
def extract_au_difference_from_paths(
    real_image_path: str, generated_image_path: str
) -> np.ndarray:
    # Extract AUs from both images
    aus_real = detector.detect_image(real_image_path).aus
    aus_generated = detector.detect_image(generated_image_path).aus

    # Compute the difference between AUs
    difference_vector = aus_real.iloc[0].values - aus_generated.iloc[0].values
    return difference_vector


# Function to check if the image is a generated one
def is_generated_image(image_path: str, image_formats: list) -> bool:
    # The regex is updated to match the image formats specified in the config
    regex = r"^[0-9]+_[0-9]+\.({})$".format("|".join(image_formats))
    image_wo_path = os.path.basename(image_path)
    return re.match(regex, image_wo_path)


# Function to measure AU differences for several images
def measure_au_differences(image_pairs: list) -> list:
    differences = []
    for real_image, generated_image in tqdm(image_pairs, unit="image"):
        difference_vector = extract_au_difference_from_paths(
            real_image, generated_image
        )
        differences.append(
            difference_vector.tolist()
        )  # Convert to list for JSON serialization
    return differences


@hydra.main(version_base=None, config_path=f"..{os.sep}conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Base paths from the configuration
    data_path = cfg["data"]
    base_path = (
        os.path.join(*data_path["base"])
        if isinstance(data_path["base"], list)
        else data_path["base"]
    )
    REAL_DATA_PATH = Path(base_path) / data_path["real"]
    GEN_DATA_PATH = Path(base_path) / data_path["generated"] / cfg["model"]["cn_use"]

    # Create directories for output
    AU_RESULTS_PATH = Path(base_path) / "au_differences"
    AU_RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    file_json_au = AU_RESULTS_PATH / f"{cfg['model']['cn_use']}_au_differences.json"

    logger.info(f"Reading images from {REAL_DATA_PATH} and {GEN_DATA_PATH}")

    # Get real image paths
    real_image_paths = [
        str(REAL_DATA_PATH / image_path)
        for image_path in os.listdir(str(REAL_DATA_PATH))
        if image_path.split(".")[-1] in data_path["image_formats"]
    ]
    real_image_paths.sort()

    # Get corresponding generated image paths (based on naming convention)
    image_pairs = []
    for real_image in real_image_paths:
        real_image_name = os.path.basename(real_image)
        generated_image_name = real_image_name.replace(
            ".jpg", "_1.png"
        )  # You can change this based on your naming convention
        generated_image_path = GEN_DATA_PATH / generated_image_name

        if generated_image_path.exists():
            image_pairs.append((real_image, str(generated_image_path)))
        else:
            logger.warning(
                f"Generated image {generated_image_name} not found, skipping..."
            )

    # Extract AU differences between real and generated images
    au_differences = measure_au_differences(image_pairs)

    # Prepare the results dictionary
    dict_of_au_differences = {
        "image_pairs": [
            (os.path.basename(real), os.path.basename(gen)) for real, gen in image_pairs
        ],
        "au_differences": au_differences,
    }

    # Save the results to a JSON file
    logger.info(f"Writing AU differences to {file_json_au}")
    with open(file_json_au, "w") as outfile:
        json.dump(dict_of_au_differences, outfile, indent=4)


def test_au_extractor():
    image_test = "/gen_data/data/real/demo/trump.jpeg"
    from PIL import Image
    import torchvision

    image = Image.open(image_test)
    print(f"{image_test} loaded")
    image = torchvision.transforms.functional.pil_to_tensor(image)
    print("image converted to tensor")

    au_extractor = AUExtractor()

    a = au_extractor._extract([image])
    print(a)
