"""
Training script for Bladder Segmentation
U-Net++ with Boundary Attention
"""

import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import create_dataloaders
from src.models.unet_plus import create_model
from src.models.losses import get_loss_function


class Trainer:
    def __init__(self, config):
        self.device = torch.device(config["device"])
        self.model = create_model(config).to(self.device)
        self.criterion = get_loss_function(config)
        self.config = config

        self.train_loader, self.val_loader = create_dataloaders(config)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.get("lr", 1e-4)
        )

        os.makedirs(config["output_dir"], exist_ok=True)

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for batch in tqdm(self.train_loader, desc="Training"):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            self.optimizer.zero_grad()
            outputs, _ = self.model(images)

            loss, _ = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                outputs, _ = self.model(images)
                loss, _ = self.criterion(outputs, masks)

                running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(self.val_loader.dataset)
        return epoch_loss

    def train(self):
        best_val_loss = float("inf")
        for epoch in range(self.config["epochs"]):
            print(f"\n=== Epoch {epoch+1}/{self.config['epochs']} ===")

            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val   Loss: {val_loss:.4f}")

            # Save checkpoint
            checkpoint_path = os.path.join(
                self.config["output_dir"], f"checkpoint_epoch_{epoch+1}.pth"
            )
            torch.save(self.model.state_dict(), checkpoint_path)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(self.config["output_dir"], "best_model.pth")
                torch.save(self.model.state_dict(), best_path)
                print(f"Saved best model to {best_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="models/checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--loss_type", type=str, default="combined")
    parser.add_argument("--lr", type=float, default=1e-4)

    # Optional model params
    parser.add_argument("--encoder_name", type=str, default="efficientnet-b3")
    parser.add_argument("--encoder_weights", type=str, default="imagenet")
    parser.add_argument("--classes", type=int, default=1)

    args = parser.parse_args()
    config = vars(args)

    print("Using device:", config["device"])
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
