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
from typing import Any, Callable, Dict
from omegaconf import DictConfig, OmegaConf

import torch

from ciagen.qm.metrics.frechet_inception_distance import FID
from ciagen.qm.metrics.inception_score import IS
from ciagen.utils.common import logger, load_images_from_directory
from ciagen.feature_extractors import (
    AVAILABLE_FEATURE_EXTRACTORS,
    instance_feature_extractor,
    instance_transform,
)
from ciagen.utils.data_loader import create_local_dataloader

# Do not let torch decide on best algorithm (we know better!)
torch.backends.cudnn.benchmark = False


class DTD:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.available_metrics = {
            "fid": FID,
            "inception_score": IS,
        }

    def _instance_metric(
        self,
        metric_name: str,
        feature_extractor: None | Callable[[Any], torch.Tensor] = None,
        **kwargs,
    ) -> Any:
        return self.available_metrics[metric_name](
            feature_extractor=feature_extractor, **kwargs
        )
        pass

    def __call__(self, paths: Dict[str, str | Path]) -> None:
        data = self.cfg["data"]
        batch_size = self.cfg["data"]["batch_size"]

        transform_dict = {
            fe: instance_transform(fe) for fe in self.cfg["metrics"]["fe"]
        }

        def call_dataloader(fe, is_real):
            if is_real:
                samples_path = paths["real_images"]
                labels_path = paths["real_labels"]
                captions_path = paths["real_captions"]
                limit_size = self.cfg["data"]["limit_size_real"]
            else:
                samples_path = paths["generated"]
                labels_path = None
                captions_path = None
                limit_size = self.cfg["data"]["limit_size_syn"]
            return create_local_dataloader(
                samples_path=samples_path,
                labels_path=labels_path,
                captions_path=captions_path,
                limit_size=limit_size,
                datatype=self.cfg["data"]["datatype"],
                transform=transform_dict[fe],
                batch_size=batch_size,
                sample_formats=data["image_formats"],
            )

        # # Paths and data related work
        # _real_path = paths["real"]
        generated_path = paths["generated"]
        real_path_images = paths["real_images"]

        # real_labels_path = paths["real_labels"]
        # real_captions_path = paths["real_captions"]

        meta_data_file = Path(generated_path) / "metadata.yaml"

        # Loading real images
        # real_images = load_images_from_directory(
        #     directory=real_path_images,
        #     formats=data["image_formats"],
        #     limit_size=self.cfg["data"]["limit_size_real"],
        # )
        real_dataset_size = len(real_path_images)

        # # Loading synthetic images
        # synthetic_images = load_images_from_directory(
        #     directory=generated_path,
        #     formats=data["image_formats"],
        #     limit_size=self.cfg["data"]["limit_size_syn"],
        # )
        synthetic_dataset_size = len(real_path_images)

        logger.info(f"Using {real_dataset_size} Real images from: {real_path_images}")
        logger.info(
            f"Using {synthetic_dataset_size} Synthetic images from: {generated_path}"
        )
        logger.info(f"Will save to {meta_data_file}")

        metrics_values = {}
        current_metrics = self.cfg["metrics"]["dtd"]
        current_fe = self.cfg["metrics"]["fe"]

        for metric in current_metrics:
            if metric not in self.available_metrics:
                logger.exception(
                    f"There is no {metric} metric available, metrics are {list(self.available_metrics.keys())}"
                )
                continue

            current_metric_values = {}

            for fe in current_fe:

                if fe not in AVAILABLE_FEATURE_EXTRACTORS:
                    logger.exception(
                        f"There is no {fe} feature extractor available, feature extractors are {AVAILABLE_FEATURE_EXTRACTORS}"
                    )
                    continue

                real_dataloader = call_dataloader(fe, is_real=True)
                synthetic_dataloader = call_dataloader(fe, is_real=False)

                feature_extractor = instance_feature_extractor(fe)

                metric_calculator = self._instance_metric(
                    metric_name=metric,
                    feature_extractor=feature_extractor,
                )

                logger.info(f"Running {metric}x{fe}")

                score = metric_calculator.score(
                    real_samples=real_dataloader,
                    synthetic_samples=synthetic_dataloader,
                    batch_size=batch_size,
                )

                current_metric_values[fe] = score
                logger.info(f"{metric}x{fe} is {score}")

            metrics_values[metric] = current_metric_values

        metadata = OmegaConf.load(meta_data_file)

        if "results" not in metadata:
            metadata["results"] = {}

        if "metrics" not in metadata["results"]:
            metadata["results"]["metrics"] = {}

        if "dtd" not in metadata["results"]["metrics"]:
            metadata["results"]["metrics"]["dtd"] = {}

        metadata["results"]["metrics"]["dtd"] = metrics_values

        with open(meta_data_file, "w") as f:
            OmegaConf.save(metadata, f)

        return metrics_values
