import sys
sys.path.append("dLux/dev")
from layers import *

help(SoftEdgedCircularAperture)
dir(SoftEdgedCircularAperture)

aperture = SoftEdgedCircularAperture(1024, .25, -.25, np.pi / 4., 0., 1., 1. / 1024., .5)
# aperture = SoftEdgedCircularAperture(1024)

pyplot.imshow(aperture._aperture())
pyplot.colorbar()
pyplot.show()
