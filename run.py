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
| For basic usage do `python run.py -- task=<task>`.
| You can also modify the `config.yaml` file.
| For more information, please go see the README.md file.
| {sep}
"""
    print(help_message)
    return


def gen(cfg: DictConfig) -> None:
    from ciagen.exes import Generator

    generator = Generator(cfg)
    paths = generate_all_paths(cfg)
    return generator(paths)


architectures = ("StableDiffusion", "ControlNet")
allowed_tasks = {
    "help": help_task,
    "gen": gen,
    # "test": ciagen.test,
    # "train_studies": ciagen.train_studies,
    # "train": ciagen.train,
    # "coco": ciagen.coco,
    # "flickr30k": ciagen.flickr30k,
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
