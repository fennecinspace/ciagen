import uuid
from pathlib import Path
from typing import Dict

from omegaconf import DictConfig
from ultralytics import YOLO

from ciagen.utils.io import logger as ciagen_logger


def train_yolo(cfg: DictConfig, paths: Dict[str, str | Path]) -> None:
    mixed_yaml_path = paths["mixed_yamls_folder_path"]
    data_yaml_path = Path(mixed_yaml_path) / "data.yaml"

    ciagen_logger.info(f"Using dataset in: {data_yaml_path.resolve()}")

    model = YOLO("yolov8n.yaml")
    cn_use = cfg["model"]["cn_use"]
    aug_percent = cfg["ml"]["augmentation_percent"]
    name = f"{uuid.uuid4().hex.upper()[0:6]}_{cn_use}_{aug_percent}"
    sampling_code_name = cfg["ml"]["sampling"]["metric"] + "_" + cfg["ml"]["sampling"]["sample"]

    model.train(
        data=data_yaml_path.resolve(),
        epochs=cfg["ml"]["epochs"],
        entity=cfg["ml"]["wandb"]["entity"],
        project=cfg["ml"]["wandb"]["project"],
        name=name,
        control_net=("Starting_point" if cfg["ml"]["augmentation_percent"] == 0 else cn_use),
        sampling=(sampling_code_name if cfg["ml"]["sampling"]["enable"] else "disabled"),
    )


if __name__ == "__main__":
    from omegaconf import OmegaConf

    from ciagen.data.paths import generate_all_paths

    cfg = OmegaConf.load("ciagen/conf/config.yaml")
    paths = generate_all_paths(cfg)
    train_yolo(cfg, paths)
