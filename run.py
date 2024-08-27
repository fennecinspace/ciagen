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


def prepare_data_task(cfg: DictConfig) -> None:
    from ciagen.exes import Flickr30kDataset, COCODataset

    if cfg['data']['base'] == 'coco':
        downloader = COCODataset(cfg)
    elif cfg['data']['base'] == 'flickr30k':
        downloader = Flickr30kDataset(cfg)
    else:
        downloader: lambda paths: print(f'[ERROR]: Dataset {cfg["data"]["base"]} not predefined, please use "coco" or "flickr30k" in the config file')
    
    paths = generate_all_paths(cfg)
    return downloader(paths)


def create_mixed_yolo_dataset_task(cfg: DictConfig) -> None:
    from ciagen.exes import CreateMixedYoloDataset

    mixed_yolo_dataset_creator = CreateMixedYoloDataset(cfg)
    paths = generate_all_paths(cfg)

    return mixed_yolo_dataset_creator(paths)

architectures = ("StableDiffusion", "ControlNet")
allowed_tasks = {
    "help": help_task,
    "gen": gen_task,
    # "test": ciagen.test,
    # "train_studies": ciagen.train_studies,
    # "train": ciagen.train,
    # "coco": coco_task,
    # "flickr30k": flickr30k_task,
    "prepare_data": prepare_data_task,
    'create_mixed_yolo_dataset': create_mixed_yolo_dataset_task,
    # "iqa": ciagen.iqa,
    # "iqa_paper": ciagen.iqa_paper,
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
