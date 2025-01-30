import numpy as np 
import networkx as nx
from tqdm import tqdm, trange
import argparse
import matplotlib.pyplot as plt


## Fix centers
## Change amplitude of Gaussians 
## Save several datasets 
## Run training on the <downsampled version> of the datasets

def gaussian_2d(xv, yv, amplitude=1, center_x=0, center_y=0, sigma_x=1, sigma_y=1):
    z1 = (xv - center_x)**2/(2*(sigma_x**2))
    z2 = (yv - center_y)**2/(2*(sigma_y**2))
    z = amplitude * np.exp(-(z1 + z2))
    return z

def create_artificial_dem(xv, yv, centers, amp=2.0):
    z_out = np.zeros((xv.shape[0], xv.shape[1]))
    for i in range(len(centers)):
        x_c = centers[i][0]
        y_c = centers[i][1]
        
        z = gaussian_2d(xv, yv, amplitude=amp, center_x = x_c, center_y = y_c, sigma_x = 1.0, sigma_y = 1.0)
        z_out += z
    return z_out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amplitudes", type=float, nargs='+')
    parser.add_argument("--size", type=int)
    args = parser.parse_args()
    n = args.size 
    # create x and y range
    x = np.linspace(0, 10, n)
    y = np.linspace(0, 10, n)
    xv, yv = np.meshgrid(x, y)

    fig =plt.figure(figsize=(40, 5))
    ax_ct = 1
    centers = np.random.choice(x, size=(15, 2))
    for amp in args.amplitudes:
        name=f'/data/sam/terrain/data/artificial/change-heights/amp-{amp}.npy'
        
        img = create_artificial_dem(xv, yv, centers, amp)

        np.save(name, img)

        ax = fig.add_subplot(1, len(args.amplitudes), ax_ct, projection='3d')

        ax.plot_surface(xv, yv, img)
        ax.set_zlim(0, 20)
        ax_ct += 1
    fig.savefig('changing-heights.png')
    return 0

if __name__=="__main__":
    main()