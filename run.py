import os

import hydra
from omegaconf import DictConfig


def help_task(cfg: DictConfig) -> None:
    title = "Controllable Image Augmentation. Using StableDiffusion + ControlNet."
    sep = "=" * len(title)

    lines = [
        sep,
        title,
        "",
        "Possible tasks:",
        str([", ".join(list(allowed_tasks.keys())[1:])]),
        "",
        "For basic usage do `python run.py task=<task>`.",
        "You can also modify the `config.yaml` file.",
        "For more information, see the README.md file.",
        sep,
    ]
    maximum_length = max(len(line) for line in lines)
    new_lines = [f"| {line} {' ' * (maximum_length - len(line))}|\n" for line in lines]
    print("".join(new_lines))


def gen_task(cfg: DictConfig) -> None:
    from ciagen.hydra_compat import run_gen

    run_gen(cfg)


def dtd_task(cfg: DictConfig) -> None:
    from ciagen.hydra_compat import run_dtd

    run_dtd(cfg)


def ptd_task(cfg: DictConfig) -> None:
    from ciagen.hydra_compat import run_ptd

    run_ptd(cfg)


def filtering_task(cfg: DictConfig) -> None:
    from ciagen.hydra_compat import run_filtering

    run_filtering(cfg)


def auto_captioner_task(cfg: DictConfig) -> None:
    from ciagen.api.caption import caption
    from ciagen.data.paths import generate_all_paths

    paths = generate_all_paths(cfg)
    caption(
        images=paths["real_images"],
        captions_dir=paths["real_captions"],
        engine=cfg["auto_captioner"]["service"]["engine"],
        model=cfg["auto_captioner"]["service"]["model"],
        api_key=cfg["auto_captioner"]["service"].get("api_key"),
        image_formats=list(cfg["data"]["image_formats"]),
    )


def prepare_data_task(cfg: DictConfig) -> None:
    from ciagen.data.paths import generate_all_paths

    dataset = cfg["data"]["base"]
    paths = generate_all_paths(cfg)

    if dataset == "coco":
        from examples.prepare_coco import prepare_coco

        prepare_coco(cfg, paths)
    elif dataset == "flickr30k":
        from examples.prepare_flickr30k import prepare_flickr30k

        prepare_flickr30k(cfg, paths)
    elif dataset in ("fer_real", "fer_gen_1_5", "fer_gen_2_1"):
        from examples.prepare_fer import prepare_fer

        prepare_fer(cfg, paths)
    elif dataset == "mocs":
        from examples.prepare_mocs import prepare_mocs

        prepare_mocs(cfg, paths)
    else:
        print(f"[ERROR] Unknown dataset: {dataset}")


def mix_dataset(cfg: DictConfig) -> None:
    from ciagen.data.paths import generate_all_paths

    dataset = cfg["data"]["base"]
    paths = generate_all_paths(cfg)
    if dataset in ("coco", "flickr30k"):
        from examples.mix_yolo_dataset import mix_yolo

        mix_yolo(cfg, paths)
    elif dataset == "fer":
        from examples.mix_fer_dataset import mix_fer

        mix_fer(cfg, paths)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def train(cfg: DictConfig) -> None:
    from ciagen.data.paths import generate_all_paths

    dataset = cfg["data"]["base"]
    paths = generate_all_paths(cfg)
    if dataset in ("coco", "flickr30k"):
        from examples.train_yolo import train_yolo

        train_yolo(cfg, paths)
    elif dataset == "fer":
        from examples.train_classifier import train_classifier

        train_classifier(cfg, paths)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


allowed_tasks = {
    "help": help_task,
    "prepare_data": prepare_data_task,
    "auto_caption": auto_captioner_task,
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
    task = cfg["task"]
    if task not in allowed_tasks:
        raise ValueError(f"Unknown task: {task}. Available: {list(allowed_tasks.keys())}")

    allowed_tasks[task](cfg)


if __name__ == "__main__":
    my_app()
