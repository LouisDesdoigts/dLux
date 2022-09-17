import jax.numpy as np
import matplotlib.pyplot as pyplot
import sys
sys.path.append("dLux/dev")
from layers import *

aperture = SoftEdgedAnnularAperture(1024, .25, -.25, np.pi / 4., 0., 1., 1. / 1024., .5, .1)

pyplot.imshow(aperture._aperture())
pyplot.colorbar()
pyplot.show()
