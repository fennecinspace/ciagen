"""CLI entry point for ciagen."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="ciagen",
        description="Controllable Image Augmentation framework",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # generate
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic images")
    gen_parser.add_argument("--source", required=True, help="Source images directory")
    gen_parser.add_argument("--output", required=True, help="Output directory")
    gen_parser.add_argument(
        "--extractor",
        required=True,
        choices=["canny", "openpose", "segmentation", "mediapipe_face"],
    )
    gen_parser.add_argument("--sd-model", required=True, help="Stable Diffusion model ID")
    gen_parser.add_argument("--cn-model", required=True, help="ControlNet model ID")
    gen_parser.add_argument("--num", type=int, default=1, help="Images per source image")
    gen_parser.add_argument("--seed", type=int, default=34567)
    gen_parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    gen_parser.add_argument("--prompt", default=None, help="Positive prompt")
    gen_parser.add_argument("--negative-prompt", default=None, help="Negative prompt")
    gen_parser.add_argument("--quality", type=int, default=30, help="Inference steps")
    gen_parser.add_argument("--guidance-scale", type=float, default=7.0)

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate generated images")
    eval_parser.add_argument("--real", required=True, help="Real images directory")
    eval_parser.add_argument("--generated", required=True, help="Generated images directory")
    eval_parser.add_argument("--metrics", nargs="+", default=["fid", "mld"], help="Metrics to compute")
    eval_parser.add_argument("--feature-extractor", default="vit", choices=["vit", "inception"])
    eval_parser.add_argument("--batch-size", type=int, default=32)
    eval_parser.add_argument("--device", default=None, help="Device (auto-detected if omitted)")

    # filter
    filter_parser = subparsers.add_parser("filter", help="Filter generated images by quality")
    filter_parser.add_argument("--generated", required=True, help="Generated images directory")
    filter_parser.add_argument("--method", required=True, choices=["threshold", "top-k", "top-p"])
    filter_parser.add_argument("--value", type=float, required=True, help="Filtering threshold value")
    filter_parser.add_argument("--metric", default="mld")
    filter_parser.add_argument("--feature-extractor", default="vit")

    # caption
    cap_parser = subparsers.add_parser("caption", help="Auto-caption images")
    cap_parser.add_argument("--images", required=True, help="Images directory")
    cap_parser.add_argument("--output", required=True, help="Output captions directory")
    cap_parser.add_argument("--engine", default="openrouter", choices=["openrouter", "openai", "ollama"])
    cap_parser.add_argument("--model", default="google/gemini-2.0-flash-001", help="Vision model name")
    cap_parser.add_argument("--api-key", default=None)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "generate":
        from ciagen.api.generate import generate

        result = generate(
            source=args.source,
            output=args.output,
            extractor=args.extractor,
            sd_model=args.sd_model,
            cn_model=args.cn_model,
            num_per_image=args.num,
            seed=args.seed,
            device=args.device,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            quality=args.quality,
            guidance_scale=args.guidance_scale,
        )
        print(f"Generated {result['total_generated']} images -> {result['output_path']}")

    elif args.command == "evaluate":
        from ciagen.api.evaluate import evaluate

        scores = evaluate(
            real=args.real,
            generated=args.generated,
            metrics=args.metrics,
            feature_extractor=args.feature_extractor,
            batch_size=args.batch_size,
            device=args.device,
        )
        for category, values in scores.items():
            for metric_name, score in values.items():
                print(f"{category}/{metric_name}: {score}")

    elif args.command == "filter":
        from ciagen.api.filter import filter_generated

        kept = filter_generated(
            generated=args.generated,
            method=args.method,
            value=args.value,
            metric=args.metric,
            feature_extractor=args.feature_extractor,
        )
        for metric_name, fe_data in kept.items():
            for fe, images in fe_data.items():
                print(f"{metric_name}/{fe}: kept {len(images)} images")

    elif args.command == "caption":
        from ciagen.api.caption import caption

        caption(
            images=args.images,
            captions_dir=args.output,
            engine=args.engine,
            model=args.model,
            api_key=args.api_key,
        )
        print(f"Captions saved to {args.output}")


if __name__ == "__main__":
    main()
