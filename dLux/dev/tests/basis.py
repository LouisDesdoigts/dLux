import dLux as dl
import matplotlib as mpl 
import matplotlib.pyplot as plt

mpl.rcParams["text.usetex"] = True

pixels = 128
nterms = 6

coordinates = dl.utils.get_pixel_coordinates(2., 2. / pixels)
