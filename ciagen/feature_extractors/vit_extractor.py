import torch
from transformers import ViTModel
from transformers import AutoImageProcessor
from tqdm import tqdm

from transformers import ViTModel, ViTForImageClassification
from transformers.models.vit.modeling_vit import ViTSelfAttention
import torch

from typing import Collection
from torchvision.transforms import Resize, Compose
from ciagen.feature_extractors.abc_feature_extractor import FeatureExtractor


def _default_collate(batch):
    data = [
        item[0] for item in batch
    ]  # item[0] contains the data, item[1] would contain the label
    return torch.stack(data)


def _instance_vit_extractor(
    model_name, batch_size, num_workers, features_output, collate_fn, device
):
    return (
        InnerFeatureExtractor(
            model_name=model_name,
            batch_size=batch_size,
            num_workers=num_workers,
            features_output=features_output,
            device=device,
            collate_fn=collate_fn,
        ),
    )


def vit_transform():
    return Compose(
        [
            Resize((224, 224)),
            # ToTensor(),
        ]
    )


class VitExtractor(FeatureExtractor):
    def __init__(
        self,
        model_name="google/vit-base-patch16-224-in21k",
        batch_size=128,
        num_workers=4,
        device="cuda",
        features_output="pooler",
        collate_fn=_default_collate,
    ):
        super().__init__()
        self._vit = InnerFeatureExtractor(
            model_name=model_name,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            features_output=features_output,
            collate_fn=collate_fn,
        )

    def forward(self, x):
        return self._vit(x)


class WrappedViTModel(torch.nn.Module):
    def __init__(self, model_name, objective="raw"):
        super().__init__()

        assert objective in ("raw", "classification")

        self.model_name = model_name

        if objective == "raw":
            self.vit_model = ViTModel.from_pretrained(model_name)
        if objective == "classification":
            self.vit_model = ViTForImageClassification.from_pretrained(model_name)

        self.vit_model.eval()

        # Initialize storage for attention maps and gradients
        self.attention_maps = []
        self.attn_gradients = []

        self.attention_layers = []
        # Register hooks for all MultiheadAttention modules
        for _name, module in self.vit_model.named_modules():
            # Attach hook to droput inside the layer, as the layer itself doesn't return attention_probs
            if isinstance(module, ViTSelfAttention):
                # module.dropout.register_forward_hook(self.attention_hook)
                module.register_forward_hook(self.attention_hook)

    # backwards and forward dont have same index .........
    def attention_hook(self, module, input, output):
        # Save the attention map
        self.attention_maps.append(output[1])

        # Register hook for gradients (this will be called during the backward pass)
        output[1].register_hook(self.save_attn_gradients)

    def save_attn_gradients(self, grad):
        self.attn_gradients = [grad] + self.attn_gradients

    def forward(self, pixel_values):
        # Reset attention maps and gradients at each forward call
        self.attention_maps = []
        self.attn_gradients = []

        outputs = self.vit_model.forward(
            pixel_values=pixel_values, output_attentions=True
        )

        # return outputs.pooler_output
        return outputs

    def get_attention_maps(self):
        return self.attention_maps

    def get_attn_gradients(self):
        return self.attn_gradients


class InnerFeatureExtractor:
    # assumes the elements of the dataset are images
    def __init__(
        self,
        model_name,
        batch_size=128,
        num_workers=1,
        device=None,
        features_output="pooler",
        collate_fn=None,
    ):
        assert features_output in [
            "pooler",
            "last_hidden",
        ], f"Incorrect features_output specified: {features_output}. Should be one of [pooler, last_hidden]"
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        model = ViTModel.from_pretrained(model_name).to(device)
        model.eval()
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.features_output = features_output
        self.collate_fn = collate_fn

        self.image_processor = AutoImageProcessor.from_pretrained(model_name)

    def __call__(self, x):
        # Define the feature extractor
        features = []
        with torch.no_grad():
            # for batch in tqdm(dataloader):
            images = x
            if not isinstance(images[0], torch.Tensor):
                raise ValueError(
                    "Dataset gettitem should return only one value (not data, labels). Use a collate funtion."
                )
            # Preprocess images using ViT feature extractor
            inputs = self.image_processor(
                images=images.permute(0, 2, 3, 1).numpy(), return_tensors="pt"
            )  # Adjust dimensions
            inputs = {
                key: val.to(self.device) for key, val in inputs.items()
            }  # Move inputs to GPU
            # Forward pass through the model
            outputs = self.model(**inputs)
            if self.features_output == "last_hidden":
                # Extract features (last hidden state)
                feature_batch = (
                    outputs.last_hidden_state.detach().cpu().to(dtype=torch.float32)
                )
            if self.features_output == "pooler":
                # Extract features (pooler output)
                # pooler output is the first token passed trough a FC layer + activation:
                # https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py#L610
                feature_batch = (
                    outputs.pooler_output.detach().cpu().to(dtype=torch.float32)
                )

            features.append(feature_batch)
        return torch.cat(features)
