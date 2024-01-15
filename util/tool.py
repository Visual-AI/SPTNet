import os
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from matplotlib.lines import Line2D


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []

    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n.strip("0.").strip("1."))
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())
    
    plt.plot(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c")
    plt.plot(np.arange(len(ave_grads)), ave_grads, alpha=0.5, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=0, top=max(max_grads)) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    # plt.legend([Line2D([0], [0], color="c", lw=4),
    #             Line2D([0], [0], color="b", lw=4)], ['max-gradient', 'mean-gradient'])
    
    fig = plt.gcf()
    fig.savefig(os.path.join("/disk/work/hjwang/gcd/methods/SimGCD/util", '1.pdf'), format='pdf', bbox_inches='tight', dpi=300)
    plt.close('all')