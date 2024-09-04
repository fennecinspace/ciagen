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

from pathlib import Path
from typing import Dict
from omegaconf import DictConfig, OmegaConf
import uuid

import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm

from ciagen.utils.common import logger

# Do not let torch decide on best algorithm (we know better!)
torch.backends.cudnn.benchmark = False

class CSVDataframeDataset(Dataset):
    def __init__(self,
            dataframe: pd.DataFrame,
            transform = None, # In case we need to transform images later, I don't know what type atm.
            images_column_index: int = 0, # first column by default
            labels_column_index: int = 1, # second column by default
            labels_column_title: str = 'labels'
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
        image = Image.open(img_path).convert('RGB')
        label = self.label_dict[self.dataframe.iloc[idx, self.labels_column_index]]

        if self.transform:
            image = self.transform(image)

        return image, label


class CSVClassificationTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg


    def __call__(self, paths: Dict[str, str | Path]) -> None:
        # CONFIG
        meta_data_file = Path(paths['mixed_yamls_folder_path']) / "train_dataset.csv"

        labels_column_title = "Emotion" # This shouldn't be hard coded, we must find a better way.

        epochs = self.cfg['ml']['epochs']
        df = pd.read_csv(meta_data_file)

        logger.info(f"Training Classifier to {epochs} epochs using Dataset {meta_data_file}")

        # Split the data into train, validation, and test sets
        train_df = df[df['Dataset'] == 'train']
        val_df = df[df['Dataset'] == 'val']
        test_df = df[df['Dataset'] == 'test']

        # Define Image Transformations
        transform = transforms.Compose([
            transforms.Resize((299, 299)),  # Resize to 299x299 for InceptionV3
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalization for InceptionV3
        ])

        # Create Data Loaders
        self.train_dataset = CSVDataframeDataset(train_df, transform=transform, labels_column_title = labels_column_title)
        self.val_dataset = CSVDataframeDataset(val_df, transform=transform,  labels_column_title = labels_column_title)
        self.test_dataset = CSVDataframeDataset(test_df, transform=transform,  labels_column_title = labels_column_title)

        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)

        # Load Pretrained InceptionV3 Model
        model = models.inception_v3(weights="DEFAULT")
        num_classes = len(self.train_dataset.label_dict)
        model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify the final layer

        # USE GPU !!!!
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        # Define Loss Function and Optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        entity = self.cfg['ml']['wandb']['entity']
        project = self.cfg['ml']['wandb']['project']

        logger.info(f"Logging to wandb user/team {entity} project {project}")

        cn_use = self.cfg['model']['cn_use']
        aug_percent = self.cfg['ml']['augmentation_percent']
        name = f"{uuid.uuid4().hex.upper()[0:6]}_{cn_use}_{aug_percent}"

        self.wb = wandb.init(
            entity = entity,
            project = project,
            name = name,
            config = dict(self.cfg),
            # settings = wb.Settings(start_method="thread")
        )

        table_train = wandb.Table(columns = ["Train_images"], data = train_df)
        table_val = wandb.Table(columns = ["Val_images"], data = val_df)
        table_test = wandb.Table(columns = ["Test_images"], data = test_df)

        self.wb.log({"Tables/Train": table_train}, commit=False)
        self.wb.log({"Tables/Val": table_val}, commit=False)
        self.wb.log({"Tables/Test": table_test}, commit=False)

        self.train_model(epochs = epochs)


    def train_model(self, epochs: int = 10):
        best_val_acc = 0.0
        best_model_path = os.path.join(self.wb.dir, "best.pth")

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            train_loader_tqdm = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Training]")
            for images, labels in train_loader_tqdm:
                images, labels = images.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs, _ = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimize
                loss.requires_grad_(True)
                loss.backward()
                self.optimizer.step()

                # Calculate running loss and accuracy
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                train_loader_tqdm.set_postfix(loss=loss.item(), accuracy=correct / total)

            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_acc = correct / total

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_loss /= len(self.val_loader.dataset)
            val_acc = val_correct / val_total

            self.wb.log({
                "Train Loss": epoch_loss,
                "Train Accuracy": epoch_acc,
                "Val Loss": val_loss,
                "Val Accuracy": val_acc,
            }, step=epoch + 1, commit=True, sync=True)

            logger.info(f'Epoch {epoch+1}/{epochs}, '
                f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                    'accuracy': val_acc,
                }, best_model_path)

                logger.info(f"New best model saved with validation accuracy: {best_val_acc:.4f}")

        wb_artifact = wandb.Artifact(type='model', name=f'run_{self.wb.id}_model')
        wb_artifact.add_file(best_model_path)
        self.wb.log_artifact(wb_artifact, aliases=['best'])

        self.wb.finish()



    def evaluate_model(self):
        self.model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_loss /= len(self.test_loader.dataset)
        test_acc = test_correct / test_total
        logger.info(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')


# if __name__ == '__main__':
#     paths = {
#         'mixed_yamls_folder_path': "/home/mohamed/Desktop",
#     }

#     classifier = CSVClassificationTrainer({
#         'ml': {'epochs': 300}
#     })

#     classifier(paths)
