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
import os
import random
from pathlib import Path
from typing import Dict
from omegaconf import DictConfig

import torch

from ciagen.qm.metrics.frechet_inception_distance import FID
from ciagen.utils.common import logger, load_images_from_directory

# Do not let torch decide on best algorithm (we know better!)
torch.backends.cudnn.benchmark = False

class DTD:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def __call__(self, paths: Dict[str, str | Path]) -> None:
        data = self.cfg["data"]

        # Paths and data related work
        real_path = paths["real"]
        generated_path = paths["generated"]
        real_path_images = paths["real_images"]

        # Loading real images
        real_images = load_images_from_directory(
            directory = real_path_images,
            formats = data["image_formats"],
            # to_tensors = True
        )
        real_dataset_size = len(real_path_images)

        # Loading synthetic images
        synthetic_images = load_images_from_directory(
            directory = generated_path,
            formats = data["image_formats"],
            # to_tensors = True
        )
        synthetic_dataset_size = len(real_path_images)

        logger.info(f"Using {real_dataset_size} Real images from: {real_path_images}")
        logger.info(f"Using {synthetic_dataset_size} Synthetic images from: {generated_path}")


        FIDCalculator = FID()

        score = FIDCalculator.instant_score(
            real_samples = real_images,
            synthetic_samples = synthetic_images,
        )

        logger.info(f"FID is {score}")
