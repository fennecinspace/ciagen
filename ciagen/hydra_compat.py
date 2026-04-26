"""Hydra-to-API compatibility layer. Translates Hydra config to ciagen.api calls."""

from pathlib import Path
from typing import Dict

from omegaconf import DictConfig, OmegaConf

from ciagen.api.evaluate import evaluate
from ciagen.api.filter import filter_generated
from ciagen.api.generate import generate
from ciagen.data.paths import generate_all_paths, get_model_config


def _model_conf(name: str, model_data: DictConfig) -> Dict:
    return get_model_config(name, model_data["cn"])


def run_gen(cfg: DictConfig) -> Dict:
    paths = generate_all_paths(cfg)
    model_data = cfg["model"]
    prompt_cfg = cfg["prompt"]
    data = cfg["data"]

    conf = _model_conf(model_data["cn_use"], model_data)
    generated_path = Path(paths["generated"])
    generated_path.mkdir(parents=True, exist_ok=True)

    metadata_dict = {"gen_config": OmegaConf.to_container(cfg, resolve=True)}
    with open(generated_path / "metadata.yaml", "w") as f:
        OmegaConf.save(metadata_dict, f)

    result = generate(
        source=paths["real_images"],
        output=paths["generated"],
        extractor=conf["extractor"],
        sd_model=conf["sd"],
        cn_model=conf["cn"],
        seed=model_data["seed"],
        device=model_data["device"],
        prompt=prompt_cfg["base"],
        negative_prompt=prompt_cfg["negative_simple"],
        quality=30,
        guidance_scale=7.5,
        use_captions=bool(prompt_cfg["use_captions"]),
        captions_dir=paths["real_captions"],
        modify_captions=bool(prompt_cfg["modify_captions"]),
        vocabulary_template=prompt_cfg["template"],
        generation_size=prompt_cfg["generation_size"],
        cn_extra_settings=conf.get("cn_extra_settings"),
        image_formats=list(data["image_formats"]),
    )

    return result


def run_dtd(cfg: DictConfig) -> Dict:
    paths = generate_all_paths(cfg)
    model_data = cfg["model"]
    config_device = (
        "cuda"
        if model_data["device"] == "cuda"
        else "cpu"
    )

    fe_name = cfg["metrics"]["fe"][0]
    results = evaluate(
        real=paths["real_images"],
        generated=paths["generated"],
        metrics=list(cfg["metrics"]["dtd"]),
        feature_extractor=fe_name,
        batch_size=cfg["data"]["batch_size"],
        limit_size_real=cfg["data"]["limit_size_real"],
        limit_size_syn=cfg["data"]["limit_size_syn"],
        image_formats=list(cfg["data"]["image_formats"]),
        device=config_device,
    )

    generated_path = paths["generated"]
    meta_data_file = generated_path / "metadata.yaml"
    metadata = OmegaConf.load(str(meta_data_file))
    _ensure_results(metadata)
    metadata["results"]["metrics"]["dtd"] = _nest_fe(results)
    with open(meta_data_file, "w") as f:
        OmegaConf.save(metadata, f)

    return results


def run_ptd(cfg: DictConfig) -> Dict:
    paths = generate_all_paths(cfg)
    model_data = cfg["model"]
    config_device = (
        "cuda"
        if model_data["device"] == "cuda"
        else "cpu"
    )

    fe_name = cfg["metrics"]["fe"][0]
    results = evaluate(
        real=paths["real_images"],
        generated=paths["generated"],
        metrics=list(cfg["metrics"]["ptd"]),
        feature_extractor=fe_name,
        batch_size=cfg["data"]["batch_size"],
        limit_size_real=cfg["data"]["limit_size_real"],
        limit_size_syn=cfg["data"]["limit_size_syn"],
        image_formats=list(cfg["data"]["image_formats"]),
        device=config_device,
    )

    generated_path = paths["generated"]
    meta_data_file = generated_path / "metadata.yaml"
    metadata = OmegaConf.load(str(meta_data_file))
    _ensure_results(metadata)
    metadata["results"]["metrics"]["ptd"] = _nest_fe(results)
    with open(meta_data_file, "w") as f:
        OmegaConf.save(metadata, f)

    return results


def run_filtering(cfg: DictConfig) -> Dict:
    paths = generate_all_paths(cfg)
    generated_path = paths["generated"]

    result = filter_generated(
        generated=generated_path,
        method=cfg["filtering"]["type"],
        value=cfg["filtering"]["value"],
        metric="mld",
        feature_extractor=cfg["metrics"]["fe"][0],
    )

    meta_data_file = generated_path / "metadata.yaml"
    metadata = OmegaConf.load(str(meta_data_file))
    _ensure_results(metadata)
    metadata["results"]["filtering"] = result
    with open(meta_data_file, "w") as f:
        OmegaConf.save(metadata, f)

    return result


def _ensure_results(metadata: DictConfig) -> None:
    if "results" not in metadata:
        metadata["results"] = OmegaConf.create({})
    if "metrics" not in metadata["results"]:
        metadata["results"]["metrics"] = OmegaConf.create({})


def _nest_fe(results: Dict) -> Dict:
    """Nest flat metric scores under feature extractor keys (API format)."""
    nested = {}
    for category, scores in results.items():
        nested[category] = {}
        for metric_name, score in scores.items():
            nested[category][metric_name] = {"vit": score}
    return nested
