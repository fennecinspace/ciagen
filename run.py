#!/bin/python3

import argparse
from enum import Enum

architectures = ("StableDiffusion", "ControlNet")
allowed_tasks = (
    "gen",
    "test",
    "train_studies",
    "train",
    "coco",
    "flickr30k",
    "iqa",
    "iqa_paper",
)


class Task(str, Enum):
    gen = "gen"
    test = "test"
    train_studies = "train_studies"
    train = "train"
    coco = "coco"
    flickr30k = "flickr30k"
    iqa = "iqa"
    iqa_paper = "iqa_paper"

    def __str__(self) -> str:
        return self.name


parser = argparse.ArgumentParser(
    prog="CIA", description=f"Generative AI module using {'+'.join(architectures)}."
)


parser.add_argument("task", choices=allowed_tasks)


if __name__ == "__main__":
    args = parser.parse_args()
    print("args", args)

    task = Task(args.task)
    print(task)
