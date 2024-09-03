import os
import yaml


from typing import Dict
from pathlib import Path

class Filtering:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, paths: Dict[str, str | Path]) -> None:

        dataset_name = self.cfg["data"]["base"]

        #TODO make this windows compliant using path.join
        cn_name = self.cfg["model"]["cn_use"]
        metadata_path = f"data/generated/{dataset_name}/{cn_name}"
        metadata_file = f"{metadata_path}/metadata.yaml"
        with open(metadata_file, 'r') as f:
            metadata_dict = yaml.safe_load(f)
            for metric_name in metadata_dict["results"]["metrics"]["ptd"]:
                ptds = metadata_dict["results"]["metrics"]["ptd"]
        metadata_dict["filtering"] = {i: {} for i in metadata_dict["results"]["metrics"]["ptd"]}
        #ptd filtering
        kept_images = {}
        for metric_name in ptds.keys():
            ptd =  list(ptds[metric_name].items())

            if self.cfg["filtering"]["type"] == "threshold":
                t = self.cfg["filtering"]["value"]
                kept = [i for i in ptd if float(i[1]) >= t]
            elif self.cfg["filtering"]["type"] == "top-p":
                p = self.cfg["filtering"]["value"]
                try:
                    assert 0 <= p <= 1
                except AssertionError as e:
                    raise ValueError("When using top-p filtering, the value (i.e. The proportion of kept datapoint) should be between 0 and 1")

                ptd = sorted(ptd, key=lambda a: float(a[1]), reverse=True)
                kept = ptd[:int(len(ptd)*p)]
            elif self.cfg["filtering"]["type"] == "top-k":
                k = self.cfg["filtering"]["value"]
                try:
                    assert 0 <= k <= len(ptd)
                    assert isinstance(k, int)
                except AssertionError as e:
                    raise ValueError("When using top-k filtering, the value (i.e. The amount of kept datapoint) should be between 0 and the total amount of datapoints")

                ptd = sorted(ptd, key=lambda a: float(a[1]), reverse=True)
                kept = ptd[:k]
            else:
                raise ValueError("Wrong filtering type specified, please select threshold, top-p or top-k")

            kept = sorted(kept, reverse=True, key=lambda a: a[1])
            kept_images[metric_name] = {i[0]: i[1] for i in kept}

        metadata_dict["filtering"][metric_name] = kept_images

        with open(metadata_file, 'w') as file:
            yaml.dump(metadata_dict, file)

