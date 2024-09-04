from pathlib import Path
from typing import Dict
from omegaconf import DictConfig, OmegaConf

import torch

from ciagen.feature_extractors import instance_feature_extractor
from ciagen.qm.metrics.mahalanobis_distance import MLD
from ciagen.utils.common import logger, load_images_from_directory

# Do not let torch decide on best algorithm (we know better!)
torch.backends.cudnn.benchmark = False


class PTD:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.available_metrics = {
            "mld": MLD,
        }

    def __call__(self, paths: Dict[str, str | Path]) -> None:
        data = self.cfg["data"]

        # Paths and data related work
        _real_path = paths["real"]
        generated_path = paths["generated"]
        real_path_images = paths["real_images"]

        meta_data_file = Path(generated_path) / "metadata.yaml"

        def loading_images(directory):

            return load_images_from_directory(
                directory=directory,
                formats=data["image_formats"],
                ptd=True,
                # to_tensors = True
            )

        # Loading real images
        real_images, _real_image_names = loading_images(real_path_images)

        # Loading synthetic images
        synthetic_images, synthetic_image_names = loading_images(generated_path)

        logger.info(f"Using {len(real_images)} Real images from: {real_path_images}")
        logger.info(
            f"Using {len(synthetic_images)} Synthetic images from: {generated_path}"
        )
        logger.info(f"Will save to {meta_data_file}")

        metrics_values = {}
        current_fe = self.cfg["metrics"]["fe"]

        for metric in self.cfg["metrics"]["ptd"]:
            if metric not in self.available_metrics:
                logger.exception(
                    f"There is no {metric} metric available, metrics are {list(self.available_metrics.keys())}"
                )
                continue

            metric_calculator = self.available_metrics[metric]()
            current_metrics_values = {}

            for fe in current_fe:
                feature_extractor = instance_feature_extractor(fe)

                scores = metric_calculator.get_mahal_distance(
                    real_samples=real_images,
                    synthetic_samples=synthetic_images,
                    feature_extractor=feature_extractor,
                )

                for image_iter in range(len(synthetic_images)):
                    full_syn_image_path = str(
                        Path(generated_path) / synthetic_image_names[image_iter]
                    )
                    current_metrics_values[full_syn_image_path] = float(
                        scores[image_iter]
                    )
                metrics_values[fe] = current_metrics_values
        metadata = OmegaConf.load(meta_data_file)

        if "results" not in metadata:
            metadata["results"] = {}

        if "metrics" not in metadata["results"]:
            metadata["results"]["metrics"] = {}

        if "ptd" not in metadata["results"]["metrics"]:
            metadata["results"]["metrics"]["ptd"] = {}

        # Even if metric already in the metadata, re-running the file means a new computation
        # if metric not in metadata["results"]["metrics"]["ptd"]:
        #     metadata["results"]["metrics"]["ptd"][metric] = metrics_values
        metadata["results"]["metrics"]["ptd"][metric] = metrics_values

        with open(meta_data_file, "w") as f:
            OmegaConf.save(metadata, f)
