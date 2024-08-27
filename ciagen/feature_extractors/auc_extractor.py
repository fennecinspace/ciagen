import os
import re
import json
import hydra

from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig
from feat import Detector
import numpy as np
import pandas as pd


import sys

sys.path.append("/gen_data/ciagen")
sys.path.append("/gen_data")


from ciagen.utils.common import logger


# Initialize the AU detector globally to reuse it
detector = Detector()


def compute_au(
    samples,
    face_detection_threshold=0.5,
    face_identity_threshold=0.8,
    frame_counter=0,
    **kwargs,
):
    # TODO: maybe make this work with a dataloader that is better at handling batchs
    #       of data
    """
    The py-feat detector does not expose a API to extract the Action Unit
    from tensor or arrays directly.
    We are simply reusing their inner code here
    """
    batch_output = []
    face_model_kwargs = kwargs.pop("face_model_kwargs", dict())
    landmark_model_kwargs = kwargs.pop("landmark_model_kwargs", dict())
    au_model_kwargs = kwargs.pop("au_model_kwargs", dict())
    emotion_model_kwargs = kwargs.pop("emotion_model_kwargs", dict())
    facepose_model_kwargs = kwargs.pop("facepose_model_kwargs", dict())
    identity_model_kwargs = kwargs.pop("identity_model_kwargs", dict())

    for sample_index, sample in enumerate(tqdm(samples)):
        (
            faces,
            landmarks,
            poses,
            aus,
            emotions,
            identities,
        ) = detector._run_detection_waterfall(
            sample,
            face_detection_threshold,
            face_model_kwargs,
            landmark_model_kwargs,
            facepose_model_kwargs,
            emotion_model_kwargs,
            au_model_kwargs,
            identity_model_kwargs,
        )

        output = detector._create_fex(
            faces,
            landmarks,
            poses,
            aus,
            emotions,
            identities,
            [""],
            frame_counter,
        )
        batch_output.append(output)
        frame_counter += 1 * 1

    batch_output = pd.concat(batch_output)
    batch_output.reset_index(drop=True, inplace=True)
    batch_output.compute_identities(threshold=face_identity_threshold, inplace=True)
    return batch_output


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
        difference_vector = extract_au_difference(real_image, generated_image)
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


if __name__ == "__main__":
    image_test = "/gen_data/data/real/yoga1.jpeg"
    from PIL import Image

    image = Image(open(image_test))

    print(type(image))
