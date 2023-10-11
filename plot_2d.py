import numpy as np
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import os

def get_all_energies( energy, min_x=-10, max_x=-10):
    min_y = min_x
    max_y = max_x
    
    grid_coarseness = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    x = np.linspace(min_x, max_x, grid_coarseness)
    y = np.linspace(min_y, max_y, grid_coarseness)
    xx, yy = np.meshgrid(x, y)
    xy = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
    xy = torch.from_numpy(xy).float().to(device)
    base_dist = torch.distributions.Normal(0,1)
    energy_base_dist = - base_dist.log_prob(xy).flatten(1).sum(1).reshape(grid_coarseness,grid_coarseness,)
    just_energy = energy(xy).reshape(grid_coarseness, grid_coarseness)
    energy_prior = just_energy + energy_base_dist

    energy_list = [energy_base_dist, energy_prior, just_energy]
    energy_list_names = ["Base Distribution", "EBM Prior", "Just EBM"]

    return energy_list, energy_list_names, x, y


def cut_samples(samples, min_x=-10, max_x =-10):
    min_y = min_x
    max_y = max_x
    tensor_min = torch.cat([torch.full_like(samples[:,0,None], min_x),torch.full_like(samples[:,1, None], min_y)], dim=1)
    tensor_max = torch.cat([torch.full_like(samples[:,0,None], max_x),torch.full_like(samples[:,1, None], max_y)], dim=1)
    samples = torch.where(samples < tensor_min, tensor_min, samples)
    samples = torch.where(samples > tensor_max, tensor_max, samples)
    return samples


def plot_contour(sample, energy_list, energy_list_names, x, y, title, logdir, epoch, step):
    if sample is not None :
        sample = sample.detach().cpu().numpy()
    fig, axs = plt.subplots(nrows=1, ncols= len(energy_list), figsize=(len(energy_list)*10, 10))
    for k,energy in enumerate(energy_list):
        energy = energy.detach().cpu().numpy()
        fig.colorbar(axs[k].contourf(x,y, energy,), ax=axs[k])
        if sample is not None :
            axs[k].scatter(sample[:,0], sample[:,1], c="red", s=1, alpha=0.5)
        axs[k].set_title(energy_list_names[k])
    fig.suptitle(title)
    if not os.path.exists(logdir +"/contour/"):
        os.makedirs(logdir +"/contour/")
    plt.savefig(logdir +"/contour/"+ f"/{title}_{str(epoch)}_{str(step)}.png")
    plt.close(fig=fig)