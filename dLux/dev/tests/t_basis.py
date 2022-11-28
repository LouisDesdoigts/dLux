__author__ = "ataras2"

import dLux as dl
import jax.numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    circ1_pos = [ 0.2,0.3]

    circ1_r = 0.1

    n_pix = 128
    coordinates = dl.utils.get_pixel_coordinates(n_pix, 1. / n_pix)


    # test with normal
    circ1 = dl.CircularAperture(x_offset=circ1_pos[0], y_offset=circ1_pos[1], radius=circ1_r, occulting=False, softening=True)

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(circ1._aperture(coordinates))
    plt.show()


    circ1_basis = dl.Basis(nterms=5, aperture=circ1)

    plt.figure()
    # b = circ1_basis.basis(coordinates=coordinates)
    b = circ1_basis._zernikes(coordinates=coordinates)
    for i in range(b.shape[0]):
        plt.subplot(1,b.shape[0],i+1)
        plt.imshow(np.squeeze(b[i,:,:]))
        
    plt.show()

    plt.figure()
    # b = circ1_basis.basis(coordinates=coordinates)
    b = circ1_basis.basis(coordinates=coordinates)
    for i in range(b.shape[0]):
        plt.subplot(1,b.shape[0],i+1)
        plt.imshow(np.squeeze(b[i,:,:]))
        
    plt.show()
