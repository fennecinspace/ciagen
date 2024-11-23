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

import os
from pathlib import Path
from typing import Dict

from omegaconf import DictConfig
from tqdm import tqdm

from ciagen.utils.common import ciagen_logger

import openai
import ollama

import os
import base64
import glob


class AutoCaptioner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        openai.api_key = self.cfg["auto_captioner"]["service"]["api_key"]

    # @hydra.main(version_base=None, config_path=f"..{os.sep}conf", config_name="config")
    def __call__(self, paths: Dict[str, str | Path]) -> None:

        to_process = []

        formats = self.cfg["data"]["image_formats"]

        if self.cfg["auto_captioner"]["custom_images_path"] and self.cfg["auto_captioner"]["custom_captions_path"]:

            images_path = self.cfg["auto_captioner"]["custom_images_path"]
            captions_path = self.cfg["auto_captioner"]["custom_captions_path"]

            Path(captions_path).mkdir(parents=True, exist_ok=True)

            to_process += [(images_path, captions_path,)]

        else:
            real_images_path = Path(paths["real_images"])
            val_images_path = Path(paths["val_images"])
            test_images_path = Path(paths["test_images"])

            real_captions_path = Path(paths["real_captions"])
            val_captions_path = Path(paths["val_captions"])
            test_captions_path = Path(paths["test_captions"])

            to_process += [
                (real_images_path, real_captions_path,),
                (val_images_path, val_captions_path,),
                (test_images_path, test_captions_path,),
            ]


        print("paths are (image,captions) :", to_process)

        for images_path, captions_path  in to_process:
            images = []
            for format in formats:
                images += [
                    *glob.glob(str(Path(images_path).absolute()) + f"/*.{format}")
                ]

            for image in images:
                caption_path = os.path.join(captions_path, image.split('/')[-1].split('.')[0] + '.txt')

                if not os.path.exists(caption_path):
                    # Generate caption
                    if self.cfg["auto_captioner"]["service"]["engine"] == "ollama":
                        caption = self.llama_generate_image_caption(image)
                    elif self.cfg["auto_captioner"]["service"]["engine"] == "openai":
                        caption = self.openai_generate_image_caption(image)
                    else:
                        raise Exception("Invalid Engine : options are ollama or openai")
                    print(f"Caption for {image}: {caption}")
                    print(f"Saving to", caption_path)
                    with open(caption_path, "w") as f:
                        f.write(caption)


    def image_to_base64(self, image_path):
        """
        Convert an image to a Base64 string.

        :param image_path: Path to the image file.
        :return: Base64-encoded string.
        """
        try:
            # Open the image file in binary mode
            with open(image_path, "rb") as image_file:
                # Read the file and encode it to Base64
                base64_string = base64.b64encode(image_file.read()).decode('utf-8')
            return base64_string
        except Exception as e:
            ciagen_logger.exception(e)


    def llama_generate_image_caption(self, image_path):
        try:
            # Send prompt to llama model
            response = ollama.chat(
                model= self.cfg["auto_captioner"]["service"]["model"],
                messages=[
                    {
                        "role": "user",
                        "content": "What is in this picture?",
                        "images": [image_path],
                    },
                    {
                        "role": "system",
                        "content": "Don't say stuff like the This 'Image contains' or 'This image Depicts', just get directly into the description of the content and the objects. Don't repeat to me the prompt I gave you, and don't talk like you're an assistant, simply give me a description of what's in the image",
                    },
                ],
            )

            caption = response['message']['content']
            return caption
        except Exception as e:
            ciagen_logger.exception(e)


    def openai_generate_image_caption(self, image_path):
        """
        Generate a caption for an image using OpenAI's text completion model.

        :param image_path: Path to the image file.
        :return: Caption string.
        """
        try:
            # Describe the image in the prompt
            image_base64 = self.image_to_base64(image_path)
            # Send prompt to GPT model
            response = openai.chat.completions.create(
                model= self.cfg["auto_captioner"]["service"]["model"],
                max_tokens= 100,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What is in this picture?",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                }
                            },
                        ],
                    },
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "Don't say stuff like the This Image contains or This image Depicts, just get directly into the description of the content of the image. Don't repeat to me the prompt I gave you, and don't talk like you're an assistant, simply give me a description of what's in the image",
                            },
                        ],
                    },

                ],
            )

            caption = str(response.choices[0].message.content)
            return caption
        except Exception as e:
            return f"Error generating caption: {e}"
