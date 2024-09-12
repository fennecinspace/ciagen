from omegaconf import OmegaConf
import yaml
from typing import Dict
from pathlib import Path
import os


class Filtering:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, paths: Dict[str, str | Path]) -> None:

        dataset_name = self.cfg["data"]["base"]
        cn_name = self.cfg["model"]["cn_use"]
        metadata_path = os.path.join("data", "generated", dataset_name, cn_name)
        metadata_file = os.path.join(metadata_path, "metadata.yaml")

        with open(metadata_file, "r") as f:
            metadata_dict = yaml.safe_load(f)
            ptds = metadata_dict["results"]["metrics"]["ptd"]

        # ptd filtering
        kept_images = {}
        for metric_name in ptds:
            ptd_by_fe = ptds[metric_name]
            kept_images_by_fe = {}
            for fe in ptd_by_fe:
                ptd = list(ptd_by_fe[fe].items())

                if self.cfg["filtering"]["type"] == "threshold":
                    t = self.cfg["filtering"]["value"]
                    kept = [i for i in ptd if float(i[1]) >= t]
                elif self.cfg["filtering"]["type"] == "top-p":
                    p = self.cfg["filtering"]["value"]
                    try:
                        assert 0 <= p <= 1
                    except AssertionError as e:
                        raise ValueError(
                            "When using top-p filtering, the value (i.e. The proportion of kept datapoint) should be between 0 and 1"
                        )

                    ptd = sorted(ptd, key=lambda a: float(a[1]), reverse=True)
                    kept = ptd[: int(len(ptd) * p)]
                elif self.cfg["filtering"]["type"] == "top-k":
                    k = self.cfg["filtering"]["value"]
                    try:
                        assert 0 <= k <= len(ptd)
                        assert isinstance(k, int)
                    except AssertionError as e:
                        raise ValueError(
                            "When using top-k filtering, the value (i.e. The amount of kept datapoint) should be between 0 and the total amount of datapoints"
                        )

                    # ptd = sorted(ptd, key=lambda a: float(a[1]), reverse=True)
                    ptd = sorted(ptd, key=lambda a: float(a[1]), reverse=False)
                    kept = ptd[:k]
                else:
                    raise ValueError(
                        "Wrong filtering type specified, please select 'threshold', 'top-p' or 'top-k'"
                    )

                kept = sorted(kept, reverse=True, key=lambda a: a[1])
                kept_images_by_fe[fe] = {i[0]: i[1] for i in kept}

            kept_images[metric_name] = kept_images_by_fe

        metadata = OmegaConf.load(metadata_file)

        if "results" not in metadata:
            metadata["results"] = {}

        metadata["results"]["filtering"] = kept_images
        with open(metadata_file, "w") as file:
            OmegaConf.save(metadata, file)
