#!/bin/python3

import hydra
from omegaconf import DictConfig, OmegaConf


import ciagen
from ciagen.exes import gen, iqa, iqa_paper, test, train_studies, train
from ciagen import extractors
from ciagen import generators
from ciagen import qmetrics


def help_task(cgf: DictConfig) -> None:
    print(dir(ciagen))
    help_message = f"""
Modulable generative data generator. Using {", ".join(architectures)}.
Possible tasks:
[{", ".join(list(allowed_tasks.keys())[1:])}]
For basic usage do `python run.py -- task=<task>`
For more information, please go see the README.md file."""
    print(help_message)
    return


architectures = ("StableDiffusion", "ControlNet")
allowed_tasks = {
    "help": help_task,
    # "gen": gen.main,
    # "test": ciagen.test,
    # "train_studies": ciagen.train_studies,
    # "train": ciagen.train,
    # "coco": ciagen.coco,
    # "flickr30k": ciagen.flickr30k,
    # "iqa": ciagen.iqa,
    # "iqa_paper": ciagen.iqa_paper,
}


@hydra.main(version_base=None, config_path=".", config_name="config")
def my_app(cfg: DictConfig) -> None:
    config_as_pretty_yaml = OmegaConf.to_yaml(cfg)

    print(config_as_pretty_yaml)

    task = cfg["task"]
    if task not in allowed_tasks:
        raise ValueError(f"task {task} not in {allowed_tasks}")

    function = allowed_tasks[task]
    function(cfg)


if __name__ == "__main__":
    my_app()
