import os

import hydra
from omegaconf import DictConfig, OmegaConf

from ciagen.utils.common import generate_all_paths


def help_task(cgf: DictConfig) -> None:
    title = f'Modulable generative data generator. Using {", ".join(architectures)}.'
    sep = "=" * len(title)
    help_message = f"""
| {sep}
| {title}
|
| Possible tasks:
| [{", ".join(list(allowed_tasks.keys())[1:])}]
|
| For basic usage do `python run.py task=<task>`.
| You can also modify the `config.yaml` file.
| For more information, please go see the README.md file.
| {sep}
"""
    print(help_message)
    return


def gen_task(cfg: DictConfig) -> None:
    from ciagen.exes import Generator

    generator = Generator(cfg)
    paths = generate_all_paths(cfg)

    return generator(paths)


def coco_task(cfg: DictConfig) -> None:
    from ciagen.exes import COCODataset

    coco_downloader = COCODataset(cfg)
    paths = generate_all_paths(cfg)

    return coco_downloader(paths)


def flickr30k_task(cfg: DictConfig) -> None:
    from ciagen.exes import Flickr30kDataset

    flickr_downloader = Flickr30kDataset(cfg)
    paths = generate_all_paths(cfg)

    return flickr_downloader(paths)


def fer_task(cfg: DictConfig) -> None:
    from ciagen.exes import FERDataset

    fer_downloader = FERDataset(cfg)
    paths = generate_all_paths(cfg)

    return fer_downloader(paths)


def prepare_data_task(cfg: DictConfig) -> None:
    from ciagen.exes import Flickr30kDataset, COCODataset, FERDataset

    if cfg["data"]["base"] == "coco":
        downloader = COCODataset(cfg)
    elif cfg["data"]["base"] == "flickr30k":
        downloader = Flickr30kDataset(cfg)
    elif cfg["data"]["base"] == "fer":
        downloader = FERDataset(cfg)
    else:
        downloader: lambda paths: print(
            f'[ERROR]: Dataset {cfg["data"]["base"]} not predefined, please use "coco", "flickr30k", "fer" in the config file'
        )

    paths = generate_all_paths(cfg)
    return downloader(paths)


def create_mixed_yolo_dataset_task(cfg: DictConfig) -> None:
    from ciagen.exes import CreateMixedYoloDataset

    mixed_yolo_dataset_creator = CreateMixedYoloDataset(cfg)
    paths = generate_all_paths(cfg)

    return mixed_yolo_dataset_creator(paths)


def create_mixed_fer_dataset_task(cfg: DictConfig) -> None:
    from ciagen.exes import CreateMixedFERDataset

    mixed_fer_dataset_creator = CreateMixedFERDataset(cfg)
    paths = generate_all_paths(cfg)

    return mixed_fer_dataset_creator(paths)


def yolo_trainer_task(cfg: DictConfig) -> None:
    from ciagen.exes import YOLOTrainer

    yolo_trainer = YOLOTrainer(cfg)
    paths = generate_all_paths(cfg)

    return yolo_trainer(paths)


def dtd_task(cfg: DictConfig) -> None:
    from ciagen.exes import DTD

    dtd_calculator = DTD(cfg)
    paths = generate_all_paths(cfg)

    return dtd_calculator(paths)


def ptd_task(cfg: DictConfig) -> None:
    from ciagen.exes import PTD

    ptd = PTD(cfg)
    paths = generate_all_paths(cfg)

    return ptd(paths)


def csv_classifier_trainer_task(cfg: DictConfig) -> None:
    from ciagen.exes import CSVClassificationTrainer

    csv_classifier_trainer = CSVClassificationTrainer(cfg)
    paths = generate_all_paths(cfg)

    return csv_classifier_trainer(paths)


def filtering_task(cfg: DictConfig) -> None:
    from ciagen.exes import Filtering

    filtering = Filtering(cfg)
    paths = generate_all_paths(cfg)

    return filtering(paths)


def mix_dataset(cfg: DictConfig) -> None:
    dataset = cfg["data"]["base"]

    if dataset == "coco":
        create_mixed_yolo_dataset_task(cfg)
    elif dataset == "flickr30k":
        create_mixed_fer_dataset_task(cfg)
    elif dataset == "fer":
        create_mixed_fer_dataset_task(cfg)
    else:
        raise ValueError(
            f"Dataset {dataset} not defined, please use 'coco', 'flickr30k', 'fer' in the config file"
        )


def train(cfg: DictConfig) -> None:
    dataset = cfg["data"]["base"]

    if dataset == "coco":
        yolo_trainer_task(cfg)
    elif dataset == "flickr30k":
        yolo_trainer_task(cfg)
    elif dataset == "fer":
        csv_classifier_trainer_task(cfg)
    else:
        raise ValueError(
            f"Dataset {dataset} not defined, please use 'coco', 'flickr30k', 'fer' in the config file"
        )


architectures = ("StableDiffusion", "ControlNet")
allowed_tasks = {
    "help": help_task,
    "prepare_data": prepare_data_task,
    "gen": gen_task,
    "dtd": dtd_task,
    "ptd": ptd_task,
    "filtering": filtering_task,
    "mix": mix_dataset,
    "train": train,
}


path_to_config = os.path.join(os.getcwd(), "ciagen", "conf")


@hydra.main(version_base=None, config_path=path_to_config, config_name="config")
def my_app(cfg: DictConfig) -> None:
    config_as_pretty_yaml = OmegaConf.to_yaml(cfg)

    # print(config_as_pretty_yaml)

    task = cfg["task"]
    if task not in allowed_tasks:
        raise ValueError(f"task {task} not in {allowed_tasks}")

    function = allowed_tasks[task]
    function(cfg)


if __name__ == "__main__":
    my_app()
