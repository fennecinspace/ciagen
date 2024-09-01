import os
from typing import Dict
import uuid
import sys
import yaml

from pathlib import Path
from omegaconf import DictConfig

from ciagen.utils.common import logger

# TODO: fix this, I do not like this at all
sys.path.append(os.path.join(os.getcwd(), "..", "ultralytics"))
from ultralytics import YOLO


class YOLOTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def __call__(self, paths: Dict[str, str | Path]) -> None:
        mixed_yaml_path = paths["mixed_yamls_folder_path"]
        data_yaml_path = Path(mixed_yaml_path) / "data.yaml"

        logger.info(f"Using dataset in: {data_yaml_path.resolve()}")

        model = YOLO("yolov8n.yaml")
        cn_use = self.cfg['model']['cn_use']
        aug_percent = self.cfg['ml']['augmentation_percent']
        name = f"{uuid.uuid4().hex.upper()[0:6]}_{cn_use}_{aug_percent}"
        sampling_code_name = (self.cfg['ml']['sampling']['metric'] + '_' + self.cfg['ml']['sampling']['sample'])

        model.train(
            data = data_yaml_path.resolve(),
            epochs = self.cfg['ml']['epochs'],
            entity = self.cfg['ml']['wandb']['entity'],
            project = self.cfg['ml']['wandb']['project'],
            name = name,
            control_net = 'Starting_point' if self.cfg['ml']['augmentation_percent'] == 0 else cn_use,
            sampling = sampling_code_name if self.cfg['ml']['sampling']['enable'] else 'disabled'
        )
