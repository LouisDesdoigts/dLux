
__all__ = ["plot_batch"]

import jax.numpy as np
import matplotlib.pyplot as plt
import equinox as eqx
from matplotlib.patches import Circle

def plot_batch(batch, filename=None, dpi=96, ncols=4, apply_det=False, rmask=False, rmin=None, rmax=None, s=None, single=False):
    nrows = int(np.ceil(2*int(len(batch))/ncols))
    scale = 3
    
    indxs = np.arange(len(batch))
    indxs1 = np.arange(1, 2*len(indxs)+1, 2)
    indxs2 = np.arange(2, 2*len(indxs)+2, 2)
    
    plt.figure(figsize=(scale*ncols, scale*nrows))
    for i in range(len(batch)):
        model = batch[i]
        if not apply_det:
            model = eqx.tree_at(lambda model: model.detector_layers, model, [])
        aperture = model.layers[2].array
        opd = model.layers[4].get_binary_phase()

        if single:
            psf = model.propagate_single(model.wavels, apply_detector=apply_det)
        else:
            psf = model()
            
        c = psf.shape[0]//2
        s = c if s is None else s
        psf = psf[c-s:c+s, c-s:c+s]

        plt.subplot(nrows, ncols, indxs1[i])
        plt.imshow(aperture * opd, cmap='hot')
        plt.xticks([])
        plt.yticks([])

        ax = plt.subplot(nrows, ncols, indxs2[i])
        plt.imshow(psf)
        if rmask:
            ax.add_patch(Circle((s-0.5, s-0.5), rmin, fill=False, color='white'))
            ax.add_patch(Circle((s-0.5, s-0.5), rmax, fill=False, color='white'))
        plt.xticks([])
        plt.yticks([])

    # save frame
    plt.tight_layout()
    if filename is not None:
        filename = '{}'.format(filename)
        plt.savefig(filename, dpi=dpi, facecolor='None')
        plt.close()
        return
    else:
        plt.show()