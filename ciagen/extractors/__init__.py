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

from .false_segmentation import FalseSegmentation
from .segmentation import Segmentation
from .openpose import OpenPose
from .canny import Canny
from .mediapipe import MediaPipeFace

AVAILABLE_EXTRACTORS = ("openpose", "canny", "mediapipe_face", "segmentation")


def extract_model_from_name(raw_name: str) -> str:
    if "openpose" in raw_name:
        return "openpose"
    elif "canny" in raw_name:
        return "canny"
    elif "mediapipe" in raw_name:
        return "mediapipe_face"
    elif "false_segmentation" in raw_name:
        return "false_segmentation"
    elif "segmentation" in raw_name:
        return "segmentation"
    else:
        raise Exception(f"Unkown model: {raw_name}")


class Extractor:
    def __new__(cls, control_model: str, **kwargs):
        if control_model not in AVAILABLE_EXTRACTORS:
            raise Exception(f"Unknown control model: {control_model}")

        if "openpose" in control_model:
            return OpenPose(**kwargs)
        elif "canny" in control_model:
            return Canny(**kwargs)
        elif "mediapipe_face" in control_model:
            return MediaPipeFace(**kwargs)
        elif "false_segmentation" in control_model:  # for paper
            return FalseSegmentation(**kwargs)
        elif "segmentation" in control_model:
            return Segmentation(**kwargs)
