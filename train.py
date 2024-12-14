import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import argparse

from dataset import DIV2KDataset
from model import FSRCNN

def train(config_fn: str):
    # Load the configuration file
    with open(config_fn, "r") as stream:
        config = yaml.safe_load(stream)

    # Dataset and DataLoader
    train_dataset = DIV2KDataset(
        hr_image_folder=config["data_path"],
        set_type="train",
        hr_img_size=config["hr_img_size"],
        lr_img_size=config["lr_img_size"],
        color_channels=config["color_channels"],
        downsample_mode=config["downsample_mode"]
    )
    val_dataset = DIV2KDataset(
        hr_image_folder=config["data_path"],
        set_type="val",
        hr_img_size=config["hr_img_size"],
        lr_img_size=config["lr_img_size"],
        color_channels=config["color_channels"],
        downsample_mode=config["downsample_mode"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["val_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )

    # Model
    model = FSRCNN(d=config["model_d"], s=config["model_s"], m=config["model_m"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=config["lr_init"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-6, verbose=True
    )

    # Training Loop
    best_val_loss = float("inf")
    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        model.train()
        train_loss = 0.0

        for lr_images, hr_images in tqdm(train_loader, desc="Training"):
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)

            # Forward pass
            outputs = model(lr_images)
            loss = criterion(outputs, hr_images)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Training Loss: {train_loss:.6f}")

        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr_images, hr_images in tqdm(val_loader, desc="Validation"):
                lr_images, hr_images = lr_images.to(device), hr_images.to(device)

                # Forward pass
                outputs = model(lr_images)
                loss = criterion(outputs, hr_images)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.6f}")

        # Checkpoint and Learning Rate Scheduler
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config["weights_fn"])
            print(f"Saved Best Model to {config['weights_fn']}")

        print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file"
    )
    args = parser.parse_args()
    train(config_fn=args.config)
