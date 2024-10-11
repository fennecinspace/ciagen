from pathlib import Path
from typing import Any, Callable, Dict
from omegaconf import DictConfig, OmegaConf

import torch

from ciagen.exes.setup_data import call_dataloader2
from ciagen.feature_extractors import (
    AVAILABLE_FEATURE_EXTRACTORS,
    instance_feature_extractor,
    instance_transform,
)
from ciagen.qm.metrics.mahalanobis_distance import MLD
from ciagen.utils.common import logger, load_images_from_directory
from ciagen.utils.data_loader import create_local_dataloader

# Do not let torch decide on best algorithm (we know better!)
torch.backends.cudnn.benchmark = False


class PTD:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.available_metrics = {
            "mld": MLD,
        }

    def _instance_metric(
        self,
        metric_name: str,
        feature_extractor: None | Callable[[Any], torch.Tensor] = None,
        device: str = "cpu",
        **kwargs,
    ):
        return self.available_metrics[metric_name](
            feature_extractor=feature_extractor, device=device, **kwargs
        )

    def __call__(self, paths: Dict[str, str | Path]) -> None:
        data = self.cfg["data"]
        batch_size = self.cfg["data"]["batch_size"]

        transform_dict = {
            fe: instance_transform(fe) for fe in self.cfg["metrics"]["fe"]
        }

        # Paths and data related work
        generated_path = paths["generated"]
        real_path_images = paths["real_images"]

        # real_labels_path = paths["real_labels"]
        # real_captions_path = paths["real_captions"]

        meta_data_file = Path(generated_path) / "metadata.yaml"

        def loading_images(directory, limit_size):

            return load_images_from_directory(
                directory=directory,
                formats=data["image_formats"],
                ptd=True,
                # to_tensors = True
                limit_size=limit_size,
            )

        # Loading real images
        first_fe = list(transform_dict.keys())[0]
        real_dummy_dataloader = call_dataloader2(
            paths, self.cfg, first_fe, transform_dict, is_real=True
        )
        syn_dummy_dataloader = call_dataloader2(
            paths, self.cfg, first_fe, transform_dict, is_real=False
        )

        # Loading synthetic images
        synthetic_images, synthetic_image_names = loading_images(
            generated_path, limit_size=self.cfg["data"]["limit_size_syn"]
        )

        logger.info(
            f"Using {len(real_dummy_dataloader.dataset)} Real images from: {paths['real_images']}"
        )
        logger.info(
            f"Using {len(syn_dummy_dataloader.dataset)} Synthetic images from: {paths['generated']}"
        )
        logger.info(f"Will save to {meta_data_file}")

        metrics_values = {}
        current_metrics = self.cfg["metrics"]["ptd"]
        current_fe = transform_dict.keys()

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        for metric in current_metrics:
            if metric not in self.available_metrics:
                logger.exception(
                    f"There is no {metric} metric available, metrics are {list(self.available_metrics.keys())}"
                )
                continue

            current_metrics_values = {}

            for fe in current_fe:
                if fe not in AVAILABLE_FEATURE_EXTRACTORS:
                    logger.exception(
                        f"There is no {fe} feature extractor available, feature extractors are {list(AVAILABLE_FEATURE_EXTRACTORS.keys())}"
                    )
                    continue

                real_dataloader = call_dataloader2(
                    paths, self.cfg, fe, transform_dict, is_real=True
                )
                synthetic_dataloader = call_dataloader2(
                    paths, self.cfg, fe, transform_dict, is_real=False
                )

                feature_extractor = instance_feature_extractor(fe, device=device)

                metric_calculator = self._instance_metric(
                    metric_name=metric,
                    feature_extractor=feature_extractor,
                    device=device,
                )

                logger.info(f"Running {metric} metric with {fe} as feature extractor")

                scores = metric_calculator.score(
                    real_samples=real_dataloader,
                    synthetic_samples=synthetic_dataloader,
                    batch_size=batch_size,
                )

                logger.info(
                    f"Done running {metric} metric with {fe} as feature extractor"
                )

                specific_dict = {}
                for image_iter in range(len(synthetic_images)):
                    full_syn_image_path = str(
                        Path(generated_path).absolute()
                        / synthetic_image_names[image_iter]
                    )
                    specific_dict[full_syn_image_path] = float(scores[image_iter])

                current_metrics_values[fe] = specific_dict

                # del everything before next iteration
                del real_dataloader
                del synthetic_dataloader
                del feature_extractor
                del metric_calculator
                torch.cuda.empty_cache()

            # TODO: this, the metrics for ptd should offer a buffer to read from and then write to file:
            # we need to write to file each time, otherwise too much in memory => process killed => and all will be lost

            metrics_values[metric] = current_metrics_values
        metadata = OmegaConf.load(meta_data_file)

        if "results" not in metadata:
            metadata["results"] = {}

        if "metrics" not in metadata["results"]:
            metadata["results"]["metrics"] = {}

        if "ptd" not in metadata["results"]["metrics"]:
            metadata["results"]["metrics"]["ptd"] = {}

        metadata["results"]["metrics"]["ptd"] = metrics_values

        with open(meta_data_file, "w") as f:
            OmegaConf.save(metadata, f)
