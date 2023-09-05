import torch
from torchvision.transforms import ToTensor
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader

from tqdm import tqdm

from model import DDPM

import wandb
from config import model_params, hyperparams

def train():

    hyperparams.update(model_params)

    logger = wandb.init(project="fashion-mnist", name="baseline")
    logger.config.update(hyperparams)

    model = DDPM(**model_params)
    dataset = FashionMNIST(root="data", download=True, train=True, transform=ToTensor())
    dataloader = DataLoader(dataset, batch_size=hyperparams["batch_size"], shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    model.train()

    for epoch in range(hyperparams["epochs"]):

        optimizer.param_groups[0]["lr"] = hyperparams["learning_rate"] * (1 - epoch / hyperparams["epochs"])

        pbar = tqdm(dataloader, mininterval=2)

        mean_loss = 0

        for images, labels in pbar:
            
            optimizer.zero_grad()
            images = images.to(hyperparams["device"])
            labels = labels.to(hyperparams["device"]).to(torch.float32)

            noise = torch.randn_like(images)

            t = torch.randint(1, hyperparams["timesteps"]+1, (images.shape[0],)).to(hyperparams["device"])

            x_perturbed = model.perturb_input(images, t, noise)

            pred_noise = model.predict_noise(x_perturbed, t/hyperparams["timesteps"], c=labels)

            loss = torch.mean((pred_noise - noise)**2)
            loss.backward()
            optimizer.step()

            mean_loss += loss.item()

        wandb.log({"loss": loss / len(dataloader)}, step=epoch)

        if epoch % 10 == 0:
            samples, intermediate = test_model(model)
            # Save a wandb table
            table = wandb.Table(columns=["Sample", "Context"])
            for i in range(50):
                table.add_data(wandb.Image(samples[i]), labels[i])

            wandb.log({"loss": loss, "epoch": epoch, "samples": table})


def test_model(model):

    model.eval()

    contexts = torch.arange(10).to(model.device).repeat(3).to(torch.float32)
    samples, intermediate = model.sample_ddpm(30, contexts=contexts, save_rate=1)

    return samples, intermediate





if __name__ == "__main__":
    #train()
    model = DDPM(**model_params)
    test_model(model)

