import numpy as np
import torch

def draw_edge(img : np.array, device : str) -> np.array:

    nit = 100 # the number of EM-iterations

    # first, transform to PyTorch tensor to easy everything
    x = (torch.from_numpy(img)/128 - 1).to(device) # [h,w,3], range: +-1
    height = x.shape[0]
    width = x.shape[1]

    # add x/y to have rgbxy-points
    x_channel = torch.linspace(-1, 1, width).reshape([1, width, 1]).expand([height, width, 1]).to(device)
    y_channel = torch.linspace(-1, 1, height).reshape([height, 1, 1]).expand([height, width, 1]).to(device)

    x = torch.cat((x, x_channel, y_channel), dim=2)

    # initial mask (random)
    mask0 = (torch.rand([height, width, 1]) > 0.5).float().to(device)
    mask1 = 1 - mask0

    for it in range(nit):
        # M-step: update parameter
        sum0 = mask0.sum()
        sum1 = mask1.sum()
        #mixing coefficients
        p0 = sum0 / (height * width)
        p1 = sum1 / (height * width)
        #means
        mu0 = (x*mask0).sum([0,1], keepdim=True)/sum0
        mu1 = (x*mask1).sum([0,1], keepdim=True)/sum1
        diff = ((x - mu0) ** 2) * mask0 + ((x - mu1) ** 2) * mask1
        std = diff.mean([0,1], keepdim=True)

        # E-step: reposibility calculation
        a = (mu1 - mu0) / std
        b = torch.log(p1) - torch.log(p0) + ((mu0**2 - mu1**2) / (2*std)).sum()

        mask1 = ((x * a).sum(2, keepdim=True) + b).sigmoid() # weak mask -> posterior probabilities
        mask0 = 1 - mask1

    # linear classifier (keep only x,y)
    mu0 = mu0[:,:,3:5]
    mu1 = mu1[:,:,3:5]
    std = std[:,:,3:5]
    x = x[:,:,3:5]
    a = (mu1 - mu0) / std
    b = torch.log(p1) - torch.log(p0) + ((mu0**2 - mu1**2) / (2*std)).sum()

    mask1 = (((x * a).sum(2, keepdim=True) + b)>0).float() # hard 0/1
    np2 = (mask1*255).to(torch.uint8).expand([height, width, 3]).cpu().numpy()

    grayscale_np2 = np2[:,:,0]  # Extract the first channel (all channels are the same)

    # Compute the horizontal and vertical gradients
    gradient_x = np.abs(np.diff(grayscale_np2, axis=1))  # Gradient along the x-axis
    gradient_y = np.abs(np.diff(grayscale_np2, axis=0))  # Gradient along the y-axis

    # Add padding to gradients to match the original image size
    gradient_x = np.pad(gradient_x, ((0, 0), (0, 1)), mode='constant', constant_values=0)
    gradient_y = np.pad(gradient_y, ((0, 1), (0, 0)), mode='constant', constant_values=0)

    # Combine the gradients to form the boundary mask
    boundary = np.clip(gradient_x + gradient_y, 0, 1) * 255

    return boundary
