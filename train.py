import torch
from torchvision.transforms import ToTensor
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader

from tqdm import tqdm

from model import DDPM

import wandb
from config import model_params, hyperparams

def target_transform(x):
    return torch.nn.functional.one_hot(torch.tensor(x), num_classes=10).to(torch.float32)


def train():

    hyperparams.update(model_params)

    logger = wandb.init(project="fashion-mnist", name="baseline")
    logger.config.update(hyperparams)

    model = DDPM(**model_params)
    dataset = FashionMNIST(root="data", download=True, train=True, transform=ToTensor(), 
                           target_transform=target_transform)
    dataloader = DataLoader(dataset, batch_size=hyperparams["batch_size"], shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    model.train()

    for epoch in range(hyperparams["epochs"]):

        optimizer.param_groups[0]["lr"] = hyperparams["learning_rate"] * (1 - epoch / hyperparams["epochs"])

        pbar = tqdm(dataloader, mininterval=2)

        mean_loss = 0

        for batch_idx, (images, labels) in enumerate(pbar):

            # Log Examples
            if epoch==0 and batch_idx==0:
                wandb.log({"Train Images": wandb.Image(images[:32])}, step=epoch)
            
            optimizer.zero_grad()
            images = images.to(hyperparams["device"])
            labels = labels.to(hyperparams["device"])

            noise = torch.randn_like(images)

            t = torch.randint(1, hyperparams["timesteps"]+1, (images.shape[0],)).to(hyperparams["device"])

            x_perturbed = model.perturb_input(images, t, noise)

            pred_noise = model.predict_noise(x_perturbed, t/hyperparams["timesteps"], c=labels)

            loss = torch.mean((pred_noise - noise)**2)
            loss.backward()
            optimizer.step()

            mean_loss += loss.item()

        wandb.log({"loss": loss / len(dataloader)}, step=epoch)

        if epoch % 10 == 0 or epoch == hyperparams["epochs"] - 1:
            samples, intermediate = test_model(model)
            wandb.log({"samples": wandb.Image(samples)}, step=epoch)

@torch.no_grad()
def test_model(model):

    model.eval()

    contexts = torch.arange(10)
    contexts = torch.nn.functional.one_hot(contexts, num_classes=10)
    contexts = contexts.repeat(3, 1).to(model.device).to(torch.float32)
    samples, intermediate = model.sample_ddpm(30, contexts=contexts, save_rate=1)

    model.train()

    return samples, intermediate


if __name__ == "__main__":
    train()