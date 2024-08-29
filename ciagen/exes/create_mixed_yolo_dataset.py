# © - 2024 Université de Mons, Multitel, Université Libre de Bruxelles, Université Catholique de Louvain

# CIA is free software. You can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License
# as published by the Free Software Foundation, either version 3
# of the License, or any later version. This program is distributed
# in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License
# for more details. You should have received a copy of the Lesser GNU
# General Public License along with this program.
# If not, see <http://www.gnu.org/licenses/>.

import hydra
import os
import random
import json

from omegaconf import DictConfig, open_dict
from pathlib import Path
from typing import List, Tuple, Dict

from ciagen.utils.common import create_yaml_file, list_images, create_files_list


def sort_based_on_score(image_paths: List[str], scores: List[float], direction: str = 'smaller') -> Tuple[List[str], List[int]]:
    """
    Sorts two arrays of the same size based on scores and returns sorted scores and image paths.

    Args:
        image_paths (List[str]): List of image paths.
        scores (List[int]): List of scores for each image.
        direction (str): either ascending or descending

    Returns:
        Tuple[List[str], List[int]]: A tuple containing the sorted scores and sorted image paths.
    """
    # Combine scores and image paths into a list of tuples
    combined_data = list(zip(scores, image_paths))
    # Sort the combined data based on scores (ascending order)
    sorted_data = sorted(combined_data, key = lambda x: x[0], reverse = False if direction == 'smaller' else True)
    # Extract sorted scores and image paths
    sorted_scores, sorted_image_paths = zip(*sorted_data)
    return sorted_image_paths, sorted_scores


class CreateMixedYoloDataset:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    # @hydra.main(version_base=None, config_path=f"..{os.sep}conf", config_name="config")
    def __call__(self, paths: Dict[str, str | Path]) -> None:

        """
        Construct the txt file containing real and synthetic data
        """

        augmentation_percent = self.cfg['ml']['augmentation_percent']
        train_nb = self.cfg['ml']['train_nb']
        val_nb = self.cfg['ml']['val_nb']
        test_nb = self.cfg['ml']['test_nb']
        sample = self.cfg['ml']['sampling']
        seed = 42
        formats = self.cfg['data']['image_formats']

        # if sample['enable']:
        #     txt_dir = txt_dir / (sample['metric'] + '_' + sample['sample'])

        if not os.path.isdir(paths['mixed_yamls_folder_path']):
            os.makedirs(paths['mixed_yamls_folder_path'])

        train_txt_path = Path(paths['mixed_yamls_folder_path']) / 'train.txt'
        val_txt_path = Path(paths['mixed_yamls_folder_path']) / 'val.txt'
        test_txt_path = Path(paths['mixed_yamls_folder_path']) / 'test.txt'
        data_yaml_path = Path(paths['mixed_yamls_folder_path']) / 'data.yaml'

        real_images_path = Path(paths['real_images'])
        val_images_path = Path(paths['val_images'])
        test_images_path = Path(Path(paths['test_images']))

        real_images = list_images(real_images_path, formats, train_nb)
        val_images = list_images(val_images_path, formats, val_nb)
        test_images = list_images(test_images_path, formats, test_nb)

        synth_images_dir = Path(paths['generated'])


        # if sample['enable']:
        #     with open(sample['score_file'], 'r') as f:
        #         score_data = json.load(f)

        #     # set sort direction of synthetic images to work with best or worst
        #     if sample['sample'] == 'best':
        #         order = score_data['best'][sample['metric']]
        #     else:
        #         if score_data['best'][sample['metric']] == 'smaller':
        #             order = 'bigger'
        #         else:
        #             order = 'smaller'

        #     synth_images, scores = sort_based_on_score(
        #         score_data['image_paths'],
        #         score_data[sample['metric']],
        #         order
        #     )

        #     synth_images = [str((Path(base_path).parent / img).absolute()) for img, score in zip(synth_images, scores)]

        # else:train_txt_path

        synth_images = list_images(synth_images_dir, formats)
        print(synth_images)
        # shuffle images
        random.Random(seed).shuffle(synth_images)

        # shuffle images
        random.Random(seed).shuffle(real_images)

        # nb_real_images = int(len(real_images) * (1 - augmentation_percent))
        nb_synth_images = int(len(real_images) * augmentation_percent)
        synth_images = synth_images[:nb_synth_images]

        train_images = real_images + synth_images

        create_files_list(train_images, train_txt_path)
        create_files_list(val_images, val_txt_path)
        create_files_list(test_images, test_txt_path)

        create_yaml_file(data_yaml_path, train_txt_path, val_txt_path, test_txt_path)

        print(f"Training yaml files created in : {paths['mixed_yamls_folder_path']}")
        print(f"Using {train_nb} Real Images from : ", real_images_path)
        print(f"Using Synthetic Images from : ", synth_images_dir)
        print(f"Using {val_nb} Validation Images from : ", val_images_path)
        print(f"Using {test_nb} Test Images from : ", test_images_path)

        return data_yaml_path
