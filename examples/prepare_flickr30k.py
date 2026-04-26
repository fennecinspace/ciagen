import os
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import wget
from omegaconf import DictConfig
from tqdm import tqdm

from ciagen.utils.bbox import (
    bbox_min_max_to_center_dims,
    calculate_iou,
)
from ciagen.utils.io import contains_word
from ciagen.utils.io import logger as ciagen_logger

PERSON_WORDS = [
    "individual",
    "human",
    "human being",
    "mortal",
    "soul",
    "creature",
    "man",
    "woman",
    "girl",
    "boy",
    "child",
    "kid",
    "baby",
    "toddler",
    "adult",
    "person",
    "humanity",
    "personage",
    "being",
    "someone",
    "somebody",
    "folk",
    "mankind",
    "fellow",
    "chap",
    "dude",
    "gentleman",
    "lady",
    "gent",
    "lass",
    "character",
    "resident",
    "residentiary",
    "homo sapiens",
    "homosapien",
    "mother",
    "mom",
    "mum",
    "mama",
    "mommy",
    "father",
    "dad",
    "daddy",
    "papa",
    "parent",
    "sister",
    "brother",
    "grandparent",
    "cousin",
    "aunt",
    "uncle",
    "niece",
    "nephew",
    "friend",
    "acquaintance",
    "companion",
    "colleague",
    "associate",
    "ally",
    "neighbor",
    "stranger",
    "mate",
    "buddy",
    "pal",
    "partner",
    "confidant",
    "confidante",
    "bachelor",
    "bachelorette",
    "betrothed",
    "bride",
    "groom",
    "spouse",
    "husband",
    "wife",
    "fiance",
    "fiancee",
    "male",
    "female",
    "player",
    "referee",
    "catcher",
    "thrower",
]

GROUP_WORDS = [
    "spectators",
    "committee",
    "faction",
    "organization",
    "society",
    "regiment",
    "troop",
    "party",
    "corps",
    "company",
    "bunch",
    "clan",
    "division",
    "sect",
    "gang",
    "gathering",
    "congregation",
    "throng",
    "staff",
    "group",
    "posse",
    "battalion",
    "crowd",
    "tribe",
    "force",
    "crew",
    "community",
    "assembly",
    "unit",
    "squad",
    "multitude",
    "association",
    "children",
    "mob",
    "family",
    "band",
    "club",
    "team",
    "ensemble",
    "collective",
    "females",
    "audience",
    "kids",
    "other",
]

BIG_NUMBERS_WORDS = [
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
]


def filter_redundant_boxes(boxes, threshold=0.9):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    keep_indices = []

    for i in range(len(boxes)):
        keep = True
        for j in range(i + 1, len(boxes)):
            iou = calculate_iou(boxes[i], boxes[j])
            if iou > threshold:
                keep = False
                break

        if keep:
            keep_indices.append(i)

    filtered_boxes = boxes[keep_indices]
    return filtered_boxes.tolist()


def filter_redundant_yolo_annotations(annotations, threshold=0.9):
    if len(annotations) == 0:
        return []

    parsed_annotations = []
    for annotation in annotations:
        class_id, x_center, y_center, width, height = annotation.split()
        parsed_annotations += [[float(x_center), float(y_center), float(width), float(height)]]

    filtered_boxes = filter_redundant_boxes(parsed_annotations, threshold)

    filtered_annotations = [f"0 {box[0]} {box[1]} {box[2]} {box[3]}" for box in filtered_boxes]

    return filtered_annotations


def download_flickr(
    data_path: Path,
    captions_path: str = "Captions",
    sentences_path: str = "Sentences",
    images_path: str = "Images",
    annotations_path: str = "Annotations",
    labels_path: str = "Labels",
    data_zip_name: str = "flickr30k.zip",
):

    data_path = Path(data_path)
    image_path: Path = data_path / images_path
    annotations_path: Path = data_path / annotations_path
    caps_path: Path = data_path / captions_path
    sentences_path: Path = data_path / sentences_path
    labels_path: Path = data_path / labels_path

    dirs = data_path, image_path, annotations_path, caps_path, sentences_path
    ciagen_logger.info(f"Attempting to create directories {[str(d) for d in dirs]}")
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    path_to_data_zip = data_path / data_zip_name

    data_url = "http://cloud.deepilia.com/s/F83NCnFzRTJDmeY/download/flickr30k.zip"

    if not os.path.exists(path_to_data_zip):
        ciagen_logger.info(f"Downloading zip images from {data_url} to {path_to_data_zip}")
        wget.download(data_url, out=str(path_to_data_zip))
    if not len(os.listdir(image_path)):
        ciagen_logger.info(f"Extracting zip images to {image_path}")
        with zipfile.ZipFile(path_to_data_zip, "r") as zip_ref:
            zip_ref.extractall(str(data_path))

    caps_path.mkdir(parents=True, exist_ok=True)
    labels_path.mkdir(parents=True, exist_ok=True)

    return image_path, annotations_path, sentences_path, caps_path, labels_path


def get_sentence_data(fn):
    with open(fn, "r") as f:
        sentences = f.read().split("\n")

    annotations = []
    for sentence in sentences:
        if not sentence:
            continue

        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []
        current_phrase = []
        add_to_phrase = False
        for token in sentence.split():
            if add_to_phrase:
                if token[-1] == "]":
                    add_to_phrase = False
                    token = token[:-1]
                    current_phrase.append(token)
                    phrases.append(" ".join(current_phrase))
                    current_phrase = []
                else:
                    current_phrase.append(token)

                words.append(token)
            else:
                if token[0] == "[":
                    add_to_phrase = True
                    first_word.append(len(words))
                    parts = token.split("/")
                    phrase_id.append(parts[1][3:])
                    phrase_type.append(parts[2:])
                else:
                    words.append(token)

        sentence_data = {"sentence": " ".join(words), "phrases": []}
        for index, phrase, p_id, p_type in zip(first_word, phrases, phrase_id, phrase_type):
            sentence_data["phrases"].append(
                {
                    "first_word_index": index,
                    "phrase": phrase,
                    "phrase_id": p_id,
                    "phrase_type": p_type,
                }
            )

        annotations.append(sentence_data)

    return annotations


def get_annotations(fn):
    tree = ET.parse(fn)
    root = tree.getroot()
    size_container = root.findall("size")[0]
    anno_info = {"boxes": {}, "scene": [], "nobox": []}
    for size_element in size_container:
        anno_info[size_element.tag] = int(size_element.text)

    for object_container in root.findall("object"):
        for names in object_container.findall("name"):
            box_id = names.text
            box_container = object_container.findall("bndbox")
            if len(box_container) > 0:
                if box_id not in anno_info["boxes"]:
                    anno_info["boxes"][box_id] = []
                xmin = int(box_container[0].findall("xmin")[0].text) - 1
                ymin = int(box_container[0].findall("ymin")[0].text) - 1
                xmax = int(box_container[0].findall("xmax")[0].text) - 1
                ymax = int(box_container[0].findall("ymax")[0].text) - 1
                anno_info["boxes"][box_id].append([xmin, ymin, xmax, ymax])
            else:
                nobndbox = int(object_container.findall("nobndbox")[0].text)
                if nobndbox > 0:
                    anno_info["nobox"].append(box_id)

                scene = int(object_container.findall("scene")[0].text)
                if scene > 0:
                    anno_info["scene"].append(box_id)

    return anno_info


def region_info_to_yolov5(image_path, region_info, class_id=0):
    img = cv2.imread(image_path)
    image_height, image_width = img.shape[0:2]

    x_center, y_center, width, height = bbox_min_max_to_center_dims(
        **region_info, image_width=image_width, image_height=image_height
    )

    yolo_str = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    return yolo_str


def create_region_desc(sentence_file, annotation_file, image_id, image_file):
    sen_data = get_sentence_data(sentence_file)
    anno_data = get_annotations(annotation_file)

    boxes = []
    captions = []

    for description in sen_data:
        captions += [description["sentence"]]

    ann_ids = []
    ann_boxes = []
    for key, boxes_list in anno_data["boxes"].items():
        for b in boxes_list:
            ann_ids += [key]
            ann_boxes += [b]

    ann_phrases = {i: [] for i in ann_ids}
    for description in sen_data:
        for phrase in description["phrases"]:
            if "people" in phrase["phrase_type"] and phrase["phrase_id"] in ann_phrases:
                ann_phrases[phrase["phrase_id"]] += [phrase["phrase"]]

    for anno_id, anno_box in zip(ann_ids, ann_boxes):
        for phrase in ann_phrases[anno_id]:
            if contains_word(phrase, GROUP_WORDS + BIG_NUMBERS_WORDS):
                return [], []

            if contains_word(phrase, PERSON_WORDS):
                box = {
                    "x_min": anno_box[0],
                    "y_min": anno_box[1],
                    "x_max": anno_box[2],
                    "y_max": anno_box[3],
                }

                boxes += [box]

    yolo_annotations = []
    for box in boxes:
        yolo_annotations += [region_info_to_yolov5(image_file, box)]

    yolo_annotations = filter_redundant_yolo_annotations(yolo_annotations)

    return yolo_annotations, captions


def prepare_flickr30k(cfg: DictConfig, paths: Dict[str, str | Path]) -> None:

    real_path = paths["root"]
    real_path_flickr = os.path.join(real_path, "flickr30k")

    os.makedirs(real_path_flickr, exist_ok=True)

    images_path, annotations_path, sentences_path, caps_path, labels_path = download_flickr(real_path_flickr)

    all_images = list(images_path.glob("*.jpg"))

    ciagen_logger.info("Extracting captions and boxes info from Initial Dataset")

    for img_path in tqdm(all_images, unit="img"):
        img_path = str(img_path.absolute())
        name = img_path.split(os.sep)[-1].split(".jpg")[0]
        img_file = name + ".jpg"
        txt_file = name + ".txt"
        xml_file = name + ".xml"

        image_file = images_path / img_file
        sentence_file = sentences_path / txt_file
        annotation_file = annotations_path / xml_file
        label_file = labels_path / txt_file
        caption_file = caps_path / txt_file

        yolo_labels, captions = create_region_desc(sentence_file, annotation_file, name, str(image_file))

        if yolo_labels:
            with open(label_file, "w") as label_file:
                label_file.write("\n".join(yolo_labels))

            with open(caption_file, "w") as caption_file:
                caption_file.write("\n".join(captions))

    test_nb = cfg["ml"]["test_nb"]
    val_nb = cfg["ml"]["val_nb"]
    train_nb = cfg["ml"]["train_nb"]

    real_train_images_path = paths["real_images"]
    real_test_images_path = paths["test_images"]
    real_val_images_path = paths["val_images"]

    real_train_labels_path = paths["real_labels"]
    real_test_labels_path = paths["test_labels"]
    real_val_labels_path = paths["val_labels"]

    real_train_captions_path = paths["real_captions"]
    real_test_captions_path = paths["test_captions"]
    real_val_captions_path = paths["val_captions"]

    ciagen_logger.info(f"Moving TRAIN to {str(real_train_images_path)}")
    ciagen_logger.info(f"Moving TEST to {str(real_test_images_path)}")
    ciagen_logger.info(f"Moving VAL to {str(real_val_images_path)}")
    ciagen_logger.info(f"Using values test: {test_nb} and validation: {val_nb}")

    all_images = [label.replace(".txt", ".jpg") for label in os.listdir(labels_path)]

    length = val_nb + test_nb + train_nb if (val_nb + test_nb + train_nb) < len(all_images) else all_images

    all_images = all_images[:length]

    counter = 0
    for file_name in tqdm(all_images, unit="img"):
        if counter > val_nb + test_nb + train_nb:
            break

        name = file_name.split(".")[0]
        img_file = name + ".jpg"
        txt_file = name + ".txt"

        image = images_path / img_file
        label = labels_path / txt_file
        caption = caps_path / txt_file

        if os.path.isfile(image) and os.path.isfile(label) and os.path.isfile(caption):
            if counter < val_nb:
                images_dir = real_val_images_path
                labels_dir = real_val_labels_path
                captions_dir = real_val_captions_path
            elif counter < val_nb + test_nb:
                images_dir = real_test_images_path
                labels_dir = real_test_labels_path
                captions_dir = real_test_captions_path
            else:
                images_dir = real_train_images_path
                labels_dir = real_train_labels_path
                captions_dir = real_train_captions_path

            shutil.copy(image, os.path.join(images_dir, img_file))
            shutil.copy(label, os.path.join(labels_dir, txt_file))
            shutil.copy(caption, os.path.join(captions_dir, txt_file))

            counter += 1


if __name__ == "__main__":
    from omegaconf import OmegaConf

    from ciagen.data.paths import generate_all_paths

    cfg = OmegaConf.load("ciagen/conf/config.yaml")
    paths = generate_all_paths(cfg)
    prepare_flickr30k(cfg, paths)
