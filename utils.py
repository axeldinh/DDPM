import numpy as np


def make_grid(images: np.ndarray, ncol: int=8) -> np.ndarray:
    """
    Takes a batch of images and makes a grid of them.
    """

    n_images = images.shape[0]
    nrow = int(np.ceil(n_images / ncol))
    channels = images.shape[1]
    width = images.shape[2]
    height = images.shape[3]

    grid = np.zeros((channels, nrow * width, ncol * height))

    for i in range(n_images):

        row = i // ncol
        col = i % ncol

        img = images[i]
        grid[:, row * width:(row + 1) * width, col * height:(col + 1) * height] = img

    return grid.transpose(1, 2, 0)