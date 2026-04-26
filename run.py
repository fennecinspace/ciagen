import os

import hydra
from omegaconf import DictConfig, OmegaConf

from ciagen.data.paths import generate_all_paths


def help_task(cfg: DictConfig) -> None:
    title = 'Controllable Image Augmentation. Using StableDiffusion + ControlNet.'
    sep = "=" * len(title)

    lines = [
        sep,
        title,
        "",
        "Possible tasks:",
        str([{", ".join(list(allowed_tasks.keys())[1:])}]),
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
    from ciagen.extractors import instantiate_extractor
    from ciagen.generators import SDCN, NaivePromptGenerator
    from ciagen.utils.io import logger, read_caption

    paths = generate_all_paths(cfg)

    import glob
    import random
    from pathlib import Path

    import torch
    from diffusers.utils import load_image
    from omegaconf import OmegaConf

    torch.backends.cudnn.benchmark = False

    data = cfg["data"]
    generated_path = paths["generated"]
    real_path_images = paths["real_images"]
    real_path_captions = paths["real_captions"]

    formats = data["image_formats"]
    real_images = []
    for fmt in formats:
        real_images += [*glob.glob(str(Path(real_path_images).absolute()) + f"/*.{fmt}")]
    real_images.sort()
    real_dataset_size = len(real_images)

    prompt = cfg["prompt"]
    modify_captions = bool(prompt["modify_captions"])
    prompt_generation_size = prompt["generation_size"]
    promp_generator = NaivePromptGenerator(prompt["template"]) if modify_captions else None

    if isinstance(prompt["base"], str):
        positive_prompt = [prompt["base"] + " " + prompt["modifier"] + " " + prompt["quality"]]
    else:
        positive_prompt = [pb + " " + prompt["modifier"] + " " + prompt["quality"] for pb in prompt["base"]]
    negative_prompt = ["".join(prompt["negative_simple"])]

    model_data = cfg["model"]
    model_to_use = model_data["cn_use"]

    from ciagen.data.paths import get_model_config
    model_to_use_conf = get_model_config(model_to_use, model_data["cn"])
    logger.info(f"Using config: {model_to_use_conf}")

    sd_model = model_to_use_conf["sd"]
    cn_model = model_to_use_conf["cn"]
    extractor_name = model_to_use_conf["extractor"]
    cn_extra_settings = model_to_use_conf.get("cn_extra_settings", {})

    use_captions = bool(prompt["use_captions"])
    prompt_per_line = bool(prompt["caption_per_line"])
    extra_empty_caption = bool(prompt["extra_empty_caption"])

    if use_captions:
        if not os.path.exists(real_path_captions):
            raise ValueError(f"Cannot find captions in {real_path_captions}")
        real_path_captions = sorted(glob.glob(str(Path(real_path_captions).absolute()) + "/*"))
        if len(real_path_captions) != real_dataset_size:
            raise ValueError("Caption dataset size doesn't match image dataset size")
        logger.info("Using captions")
    else:
        real_path_captions = []

    seed = model_data["seed"]
    device = model_data["device"]

    generator = SDCN(sd_model, cn_model, seed, device=device, cn_extra_settings=cn_extra_settings)
    ext = instantiate_extractor(extractor_name)
    logger.info(f"Using extractor: {ext}")
    logger.info(f"Using generator: {generator}")
    logger.info(f"Results will be saved to {generated_path}")

    metadata_dict = {"gen_config": cfg}
    with open(os.path.join(generated_path, "metadata.yaml"), "w") as f:
        OmegaConf.save(metadata_dict, f)

    for idx in range(real_dataset_size):
        image_path = real_images[idx]
        caption_path = real_path_captions[idx] if use_captions else None

        try:
            image = load_image(image_path)
            image_name = image_path.split(f"{os.sep}")[-1].split(".")[0]

            feature = ext.extract(image)

            if use_captions:
                positive_prompt = [
                    p.lower() for p in read_caption(
                        caption_path,
                        prompt_per_line=prompt_per_line,
                        extra_empty_caption=extra_empty_caption,
                    )
                ]
                negative_prompt = ["".join(prompt["negative_simple"])]

                if modify_captions:
                    def modify_prompt(p):
                        modified_list = promp_generator.prompts(prompt_generation_size, p)
                        cleaned_list = list(filter(lambda new_p: new_p != p, modified_list))
                        if not cleaned_list:
                            logger.info(f'No new prompt for "{p}", using original.')
                            return p
                        return random.choice(cleaned_list)
                    positive_prompt = [modify_prompt(p) for p in positive_prompt]

            output_images = generator.gen(feature, positive_prompt, negative_prompt)
            logger.info(f"Generated {len(output_images)} images from real sample.")

            for j, img in enumerate(output_images):
                img.save(os.path.join(generated_path, f"{image_name}_{j + 1}.png"))

        except Exception as e:
            logger.warning(f"Image {image_path}: Exception during Extraction/SDCN", e)

        if (idx + 1) % 50 == 0:
            logger.info(f"Processed {idx + 1} images ({idx / real_dataset_size * 100}%).")


def dtd_task(cfg: DictConfig) -> None:
    import torch

    from ciagen.data.loader import create_dataloader, create_transform_dict
    from ciagen.feature_extractors import (
        AVAILABLE_FEATURE_EXTRACTORS,
        available_feature_extractors,
        instance_feature_extractor,
    )
    from ciagen.metrics.fid import FID
    from ciagen.metrics.inception_score import IS
    from ciagen.utils.io import logger

    torch.backends.cudnn.benchmark = False

    paths = generate_all_paths(cfg)
    batch_size = cfg["data"]["batch_size"]
    transform_dict = create_transform_dict(cfg)
    current_feature_extractors = available_feature_extractors()

    available_metrics = {"fid": FID, "inception_score": IS}

    generated_path = paths["generated"]
    meta_data_file = os.path.join(generated_path, "metadata.yaml")

    first_fe = list(transform_dict.keys())[0]
    real_dummy = create_dataloader(paths, cfg, first_fe, transform_dict, is_real=True)
    syn_dummy = create_dataloader(paths, cfg, first_fe, transform_dict, is_real=False)

    logger.info(f"Using {len(real_dummy.dataset)} Real images")
    logger.info(f"Using {len(syn_dummy.dataset)} Synthetic images")

    config_device = "cuda" if (cfg["model"]["device"] == "cuda" and torch.cuda.is_available()) else "cpu"

    metrics_values = {}
    current_metrics = cfg["metrics"]["dtd"]
    current_fe = list(transform_dict.keys())

    for metric_name in current_metrics:
        if metric_name not in available_metrics:
            logger.error(f"Unknown metric: {metric_name}")
            continue

        metric_device = config_device if available_metrics[metric_name].allows_for_gpu() else "cpu"
        current_metric_values = {}

        for fe in current_fe:
            if fe not in AVAILABLE_FEATURE_EXTRACTORS:
                logger.error(f"Unknown feature extractor: {fe}")
                continue

            fe_device = metric_device if current_feature_extractors[fe].allows_for_gpu() else "cpu"
            real_dl = create_dataloader(paths, cfg, fe, transform_dict, is_real=True)
            syn_dl = create_dataloader(paths, cfg, fe, transform_dict, is_real=False)
            feature_extractor = instance_feature_extractor(fe, device=fe_device)
            metric_calc = available_metrics[metric_name](feature_extractor=feature_extractor, device=fe_device)

            logger.info(f"Running {metric_name} with {fe}")
            score = metric_calc.score(real_samples=real_dl, synthetic_samples=syn_dl, batch_size=batch_size)
            current_metric_values[fe] = score
            logger.info(f"{metric_name}[{fe}] = {score}")

        metrics_values[metric_name] = current_metric_values

    metadata = OmegaConf.load(meta_data_file)
    if "results" not in metadata:
        metadata["results"] = {}
    if "metrics" not in metadata["results"]:
        metadata["results"]["metrics"] = {}
    metadata["results"]["metrics"]["dtd"] = metrics_values

    with open(meta_data_file, "w") as f:
        OmegaConf.save(metadata, f)


def ptd_task(cfg: DictConfig) -> None:
    import torch

    from ciagen.data.loader import create_dataloader, create_transform_dict, load_images_from_directory
    from ciagen.feature_extractors import (
        AVAILABLE_FEATURE_EXTRACTORS,
        available_feature_extractors,
        instance_feature_extractor,
    )
    from ciagen.metrics.mahalanobis import MLD
    from ciagen.utils.io import logger

    torch.backends.cudnn.benchmark = False

    paths = generate_all_paths(cfg)
    batch_size = cfg["data"]["batch_size"]
    transform_dict = create_transform_dict(cfg)
    current_feature_extractors = available_feature_extractors()

    available_metrics = {"mld": MLD}
    generated_path = paths["generated"]
    meta_data_file = os.path.join(generated_path, "metadata.yaml")

    first_fe = list(transform_dict.keys())[0]
    create_dataloader(paths, cfg, first_fe, transform_dict, is_real=True)
    create_dataloader(paths, cfg, first_fe, transform_dict, is_real=False)

    synthetic_images, synthetic_image_names = load_images_from_directory(
        generated_path, formats=cfg["data"]["image_formats"], ptd=True,
        limit_size=cfg["data"]["limit_size_syn"]
    )

    config_device = "cuda" if (cfg["model"]["device"] == "cuda" and torch.cuda.is_available()) else "cpu"

    metrics_values = {}
    current_metrics = cfg["metrics"]["ptd"]
    current_fe = list(transform_dict.keys())

    for metric_name in current_metrics:
        if metric_name not in available_metrics:
            logger.error(f"Unknown metric: {metric_name}")
            continue

        metric_device = config_device if available_metrics[metric_name].allows_for_gpu() else "cpu"
        current_metric_values = {}

        for fe in current_fe:
            if fe not in AVAILABLE_FEATURE_EXTRACTORS:
                logger.error(f"Unknown feature extractor: {fe}")
                continue

            fe_device = metric_device if current_feature_extractors[fe].allows_for_gpu() else "cpu"
            real_dl = create_dataloader(paths, cfg, fe, transform_dict, is_real=True)
            syn_dl = create_dataloader(paths, cfg, fe, transform_dict, is_real=False)
            feature_extractor = instance_feature_extractor(fe, device=fe_device)
            metric_calc = available_metrics[metric_name](feature_extractor=feature_extractor, device=fe_device)

            logger.info(f"Running {metric_name} with {fe}")
            scores = metric_calc.score(real_samples=real_dl, synthetic_samples=syn_dl, batch_size=batch_size)

            specific_dict = {}
            for i in range(len(synthetic_images)):
                full_path = str(os.path.join(generated_path, synthetic_image_names[i]))
                specific_dict[full_path] = float(scores[i])

            current_metric_values[fe] = specific_dict

            del real_dl, syn_dl, feature_extractor, metric_calc
            torch.cuda.empty_cache()

        metrics_values[metric_name] = current_metric_values

    metadata = OmegaConf.load(meta_data_file)
    if "results" not in metadata:
        metadata["results"] = {}
    if "metrics" not in metadata["results"]:
        metadata["results"]["metrics"] = {}
    metadata["results"]["metrics"]["ptd"] = metrics_values

    with open(meta_data_file, "w") as f:
        OmegaConf.save(metadata, f)


def filtering_task(cfg: DictConfig) -> None:
    import os

    import yaml
    from omegaconf import OmegaConf


    dataset_name = cfg["data"]["base"]
    cn_name = cfg["model"]["cn_use"]
    metadata_file = os.path.join("data", "generated", dataset_name, cn_name, "metadata.yaml")

    with open(metadata_file, "r") as f:
        metadata_dict = yaml.safe_load(f)
        ptds = metadata_dict["results"]["metrics"]["ptd"]

    kept_images = {}
    for metric_name in ptds:
        ptd_by_fe = ptds[metric_name]
        kept_images_by_fe = {}
        for fe in ptd_by_fe:
            ptd = list(ptd_by_fe[fe].items())
            ptd = [(x[0], abs(float(x[1]))) for x in ptd]

            if cfg["filtering"]["type"] == "threshold":
                t = cfg["filtering"]["value"]
                kept = [i for i in ptd if i[1] <= t]
            elif cfg["filtering"]["type"] == "top-p":
                p = cfg["filtering"]["value"]
                if not 0 <= p <= 1:
                    raise ValueError("top-p value must be between 0 and 1")
                ptd_sorted = sorted(ptd, key=lambda a: a[1])
                kept = ptd_sorted[:int(len(ptd) * p)]
            elif cfg["filtering"]["type"] == "top-k":
                k = cfg["filtering"]["value"]
                if not 0 <= k <= len(ptd):
                    raise ValueError(f"top-k must be between 0 and {len(ptd)}")
                ptd_sorted = sorted(ptd, key=lambda a: a[1])
                kept = ptd_sorted[:k]
            else:
                raise ValueError("Use 'threshold', 'top-p', or 'top-k'")

            kept = sorted(kept, reverse=True, key=lambda a: a[1])
            kept_images_by_fe[fe] = {i[0]: i[1] for i in kept}

        kept_images[metric_name] = kept_images_by_fe

    metadata = OmegaConf.load(metadata_file)
    if "results" not in metadata:
        metadata["results"] = {}
    metadata["results"]["filtering"] = kept_images
    with open(metadata_file, "w") as f:
        OmegaConf.save(metadata, f)


def auto_captioner_task(cfg: DictConfig) -> None:
    from ciagen.captioning import AutoCaptioner

    paths = generate_all_paths(cfg)
    captioner = AutoCaptioner(
        engine=cfg["auto_captioner"]["service"]["engine"],
        model=cfg["auto_captioner"]["service"]["model"],
        api_key=cfg["auto_captioner"]["service"].get("api_key"),
        image_formats=list(cfg["data"]["image_formats"]),
    )
    captioner(paths)


def prepare_data_task(cfg: DictConfig) -> None:
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
    dataset = cfg["data"]["base"]
    if dataset in ("coco", "flickr30k"):
        from examples.mix_yolo_dataset import mix_yolo
        mix_yolo(cfg, generate_all_paths(cfg))
    elif dataset == "fer":
        from examples.mix_fer_dataset import mix_fer
        mix_fer(cfg, generate_all_paths(cfg))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def train(cfg: DictConfig) -> None:
    dataset = cfg["data"]["base"]
    if dataset in ("coco", "flickr30k"):
        from examples.train_yolo import train_yolo
        train_yolo(cfg, generate_all_paths(cfg))
    elif dataset == "fer":
        from examples.train_classifier import train_classifier
        train_classifier(cfg, generate_all_paths(cfg))
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
