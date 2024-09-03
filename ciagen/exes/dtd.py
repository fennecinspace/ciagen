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

from pathlib import Path
from typing import Dict
from omegaconf import DictConfig, OmegaConf

import torch

from ciagen.qm.metrics.frechet_inception_distance import FID
from ciagen.qm.metrics.inception_score import IS
from ciagen.utils.common import logger, load_images_from_directory

# Do not let torch decide on best algorithm (we know better!)
torch.backends.cudnn.benchmark = False


class DTD:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.available_metrics = {
            "fid": FID,
            "inception_score": IS,
        }

    def __call__(self, paths: Dict[str, str | Path]) -> None:
        data = self.cfg["data"]

        # Paths and data related work
        _real_path = paths["real"]
        generated_path = paths["generated"]
        real_path_images = paths["real_images"]

        meta_data_file = Path(generated_path) / "metadata.yaml"

        # Loading real images
        real_images = load_images_from_directory(
            directory=real_path_images,
            formats=data["image_formats"],
        )
        real_dataset_size = len(real_path_images)

        # Loading synthetic images
        synthetic_images = load_images_from_directory(
            directory=generated_path,
            formats=data["image_formats"],
        )
        synthetic_dataset_size = len(real_path_images)

        logger.info(f"Using {real_dataset_size} Real images from: {real_path_images}")
        logger.info(
            f"Using {synthetic_dataset_size} Synthetic images from: {generated_path}"
        )
        logger.info(f"Will save to {meta_data_file}")

        metrics_values = {}
        for metric in self.cfg["metrics"]["dtd"]:
            if metric not in self.available_metrics:
                logger.exception(
                    f"There is no {metric} metric available, metrics are {list(self.available_metrics.keys())}"
                )
                continue

            metric_calculator = self.available_metrics[metric]()

            score = metric_calculator.run_score(
                real_samples=real_images,
                synthetic_samples=synthetic_images,
            )

            metrics_values[metric] = score

            logger.info(f"{metric} is {score}")

        metadata = OmegaConf.load(meta_data_file)

        if "results" not in metadata:
            metadata["results"] = {}

        if "metrics" not in metadata["results"]:
            metadata["results"]["metrics"] = {}

        if "dtd" not in metadata["results"]["metrics"]:
            metadata["results"]["metrics"]["dtd"] = {}

        # if metric not in metadata["results"]["metrics"]["dtd"]:
        #     metadata["results"]["metrics"]["dtd"] = metrics_values
        metadata["results"]["metrics"]["dtd"] = metrics_values

        with open(meta_data_file, "w") as f:
            OmegaConf.save(metadata, f)

        return metrics_values
