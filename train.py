import argparse
import os

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm, trange

import wandb
from Networks import ViT
from utils import CustomMNIST, RotateTransform, eval


def main(
    n_patches,
    n_blocks,
    hidden_d,
    n_heads,
    out_d,
    ssl_preprocess,
    n_epochs,
    lr,
    batch_size,
    save_model_freq,
    eval_freq,
    save_dir,
    checkpoint,
    use_wandb,
):
    # Loading data

    if ssl_preprocess == "rotation":
        rotation_angles = [0, 90, 180, 270]

        # Define the transform including rotation
        transform = transforms.Compose(
            [transforms.ToTensor(), RotateTransform(rotation_angles)]
        )
        train_set = CustomMNIST(
            root="./../datasets",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
            angles=rotation_angles,
        )
        test_set = CustomMNIST(
            root="./../datasets",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
            angles=rotation_angles,
        )
    elif ssl_preprocess == "none":
        transform = ToTensor()
        train_set = MNIST(
            root="./../datasets", train=True, download=True, transform=transform
        )
        test_set = MNIST(
            root="./../datasets", train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown SSL preprocessing: {ssl_preprocess}")

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)

    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        "Using device: ",
        device,
        f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "",
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    out_d = 4 if ssl_preprocess == "rotation" else out_d
    model = ViT(
        (1, 28, 28),
        n_patches=n_patches,
        n_blocks=n_blocks,
        hidden_d=hidden_d,
        n_heads=n_heads,
        out_d=out_d,
    ).to(device)

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))
        print(f"Loaded model from {checkpoint}")

    # print model summary
    print(model)

    if use_wandb:
        wandb.init(project="vit-mnist", config=vars(args))
    # Training loop
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()
    for epoch in trange(n_epochs, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1} in training", leave=False
        ):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)

            loss = criterion(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().cpu().item() / len(train_loader)

        print(f"Epoch {epoch + 1}/{n_epochs} loss: {train_loss:.2f}")
        if use_wandb:
            wandb.log(
                {
                    "train_loss": train_loss,
                }
            )
        if (epoch + 1) % save_model_freq == 0:
            print(f"Saving model at epoch {epoch} to {save_dir}...")
            torch.save(
                model.state_dict(), os.path.join(save_dir, f"model_{epoch + 1}.pth")
            )
        if (epoch + 1) % eval_freq == 0:
            test_loss, test_acc = eval(model, test_loader, device, criterion=criterion)
            print(f"Test loss: {test_loss:.2f}, Test accuracy: {test_acc:.2f}%")
            if use_wandb:
                wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})

    print("Training finished!")
    # final eval
    final_test_loss, final_test_acc = eval(
        model, test_loader, device, criterion=criterion
    )
    print(
        f"Final test loss: {final_test_loss:.2f}, Final test accuracy: {final_test_acc:.2f}%"
    )
    print(f"Saving model to {save_dir}...")
    torch.save(model.state_dict(), os.path.join(save_dir, "model_latest.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_patches", type=int, default=7, help="Number of patches")
    parser.add_argument("--n_blocks", type=int, default=2, help="Number of blocks")
    parser.add_argument("--hidden_d", type=int, default=8, help="Hidden dimension")
    parser.add_argument(
        "--n_heads", type=int, default=2, help="Number of attention heads"
    )
    parser.add_argument("--out_d", type=int, default=10, help="Output dimension")
    parser.add_argument(
        "--ssl_preprocess",
        type=str,
        default="rotation",
        choices=["rotation", "none"],
        help="SSL data prepprocessing",
    )
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--save_model_freq",
        type=int,
        default=1,
        help="Frequency of epochs to save model",
    )
    parser.add_argument(
        "--save_dir", type=str, default="./models", help="Directory to save models"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint to load model from"
    )
    parser.add_argument(
        "--eval_freq", type=int, default=5, help="Frequency of epochs for evaluation"
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use wandb for logging"
    )
    args = parser.parse_args()
    # print full command line with all arguments including default values
    print(
        "Full command line:\n"
        + "python3 train.py "
        + " ".join([f"--{k} {v}" for k, v in vars(args).items()])
        + "\n"
    )
    main(**vars(args))
