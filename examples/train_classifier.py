import os
import uuid
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from ciagen.utils.io import logger as ciagen_logger

torch.backends.cudnn.benchmark = False


class CSVDataframeDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform=None,
        images_column_index: int = 0,
        labels_column_index: int = 1,
        labels_column_title: str = "labels",
    ):
        self.images_column_index = images_column_index
        self.labels_column_index = labels_column_index
        self.dataframe = dataframe
        self.transform = transform
        self.label_dict = {label: idx for idx, label in enumerate(self.dataframe[labels_column_title].unique())}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        img_path = self.dataframe.iloc[idx, self.images_column_index]
        image = Image.open(img_path).convert("RGB")
        label = self.label_dict[self.dataframe.iloc[idx, self.labels_column_index]]

        if self.transform:
            image = self.transform(image)

        return image, label


def train_classifier(
    cfg: DictConfig,
    paths: Dict[str, str | Path],
    train_dataset_csv_filename: str = "train_dataset.csv",
    labels_column_title: str = "Emotion",
) -> None:
    metadata_file = Path(paths["mixed_yamls_folder_path"]) / train_dataset_csv_filename

    epochs = cfg["ml"]["epochs"]
    df = pd.read_csv(metadata_file)

    ciagen_logger.info(f"Training Classifier to {epochs} epochs using Dataset {metadata_file}")

    train_df = df[df["Dataset"] == "train"]
    val_df = df[df["Dataset"] == "val"]
    test_df = df[df["Dataset"] == "test"]

    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = CSVDataframeDataset(train_df, transform=transform, labels_column_title=labels_column_title)
    val_dataset = CSVDataframeDataset(val_df, transform=transform, labels_column_title=labels_column_title)
    test_dataset = CSVDataframeDataset(test_df, transform=transform, labels_column_title=labels_column_title)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = models.inception_v3(weights="DEFAULT")
    num_classes = len(train_dataset.label_dict)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    entity = cfg["ml"]["wandb"]["entity"]
    project = cfg["ml"]["wandb"]["project"]

    ciagen_logger.info(f"Logging to wandb user/team {entity} project {project}")

    cn_use = cfg["model"]["cn_use"]
    aug_percent = cfg["ml"]["augmentation_percent"]
    train_nb = cfg["ml"]["train_nb"]
    filtering_strategy = cfg["filtering"]["type"]
    name = f"{uuid.uuid4().hex.upper()[0:6]}_{cn_use}_{aug_percent}_{train_nb}_{epochs}_{filtering_strategy}"

    wb = wandb.init(
        entity=entity,
        project=project,
        name=name,
        config=dict(cfg),
    )

    table_train = wandb.Table(columns=["Train_images"], data=train_df)
    table_val = wandb.Table(columns=["Val_images"], data=val_df)
    table_test = wandb.Table(columns=["Test_images"], data=test_df)

    wb.log({"Tables/Train": table_train}, commit=False)
    wb.log({"Tables/Val": table_val}, commit=False)
    wb.log({"Tables/Test": table_test}, commit=False)

    _train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        wb=wb,
        epochs=epochs,
    )


def _train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    device,
    wb,
    epochs: int = 10,
):
    best_val_acc = 0.0
    best_model_path = os.path.join(wb.dir, "best.pth")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Training]")
        for images, labels in train_loader_tqdm:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs, _ = model(images)
            loss = criterion(outputs, labels)

            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            train_loader_tqdm.set_postfix(loss=loss.item(), accuracy=correct / total)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total

        wb.log(
            {
                "Train Loss": epoch_loss,
                "Train Accuracy": epoch_acc,
                "Val Loss": val_loss,
                "Val Accuracy": val_acc,
            },
            step=epoch + 1,
            commit=True,
            sync=True,
        )

        ciagen_logger.info(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": val_loss,
                    "accuracy": val_acc,
                },
                best_model_path,
            )

            ciagen_logger.info(f"New best model saved with validation accuracy: {best_val_acc:.4f}")

    wb_artifact = wandb.Artifact(type="model", name=f"run_{wb.id}_model")
    wb_artifact.add_file(best_model_path)
    wb.log_artifact(wb_artifact, aliases=["best"])

    wb.finish()


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = test_correct / test_total
    ciagen_logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    from omegaconf import OmegaConf

    from ciagen.data.paths import generate_all_paths

    cfg = OmegaConf.load("ciagen/conf/config.yaml")
    paths = generate_all_paths(cfg)
    train_classifier(cfg, paths)
