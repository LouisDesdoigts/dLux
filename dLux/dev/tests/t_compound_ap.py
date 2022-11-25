__author__ = "ataras2"

# Optics
import dLux as dl

# Plotting/visualisation
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

circ1_pos = [ 0.2,0]
circ2_pos = [-0.2,0]

circ1_r = 0.1
circ2_r = 0.1

n_pix = 128
coordinates = dl.utils.get_pixel_coordinates(n_pix, 1. / n_pix)


# test with normal
circ1 = dl.CircularAperture(x_offset=circ1_pos[0], y_offset=circ1_pos[1], radius=circ1_r, occulting=False, softening=True)
circ2 = dl.CircularAperture(x_offset=circ2_pos[0], y_offset=circ2_pos[1], radius=circ2_r, occulting=False, softening=True)

compund_circs = dl.CompoundAperture({'c1' : circ1, 'c2' : circ2}, use_prod=False)
plt.figure()
plt.subplot(1,2,1)
plt.imshow(circ1._aperture(coordinates))
plt.subplot(1,2,2)
plt.imshow(compund_circs._aperture(coordinates))
plt.show()

dl.Basis(nterms=5, aperture=circ1)

# test with overlap
circ1_pos = [ 0.05,0]
circ2_pos = [-0.05,0]

circ1_r = 0.1
circ2_r = 0.1
circ1 = dl.CircularAperture(x_offset=circ1_pos[0], y_offset=circ1_pos[1], radius=circ1_r, occulting=False, softening=True)
circ2 = dl.CircularAperture(x_offset=circ2_pos[0], y_offset=circ2_pos[1], radius=circ2_r, occulting=False, softening=True)

compund_circs = dl.CompoundAperture({'c1' : circ1, 'c2' : circ2}, use_prod=False)
plt.figure()
plt.subplot(1,2,1)
plt.imshow(circ1._aperture(coordinates))
plt.subplot(1,2,2)
plt.imshow(compund_circs._aperture(coordinates))
plt.show()