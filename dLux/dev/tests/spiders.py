import dLux as dl
import matplotlib as mpl 
import matplotlib.pyplot as plt

mpl.rcParams["text.usetex"] = True
mpl.rcParams["image.cmap"] = "inferno"

pixels = 128
coordinates = dl.utils.get_pixel_coordinates(pixels, 2. / pixels)

# Uniform Spider Testing
even_soft_unif_spider = dl.UniformSpider(0., 0., 4., .1, 0., softening=True)
even_hard_unif_spider = dl.UniformSpider(0., 0., 4., .1, 0., softening=False)
pos_x_trans_unif_spider = dl.UniformSpider(.5, 0., 4., .1, 0., softening=True)
neg_x_trans_unif_spider = dl.UniformSpider(-.5, 0., 4., .1, 0., softening=True)
pos_y_trans_unif_spider = dl.UniformSpider(0., .5, 4., .1, 0., softening=True)
neg_y_trans_unif_spider = dl.UniformSpider(0., -.5, 4., .1, 0., softening=True)
odd_soft_unif_spider = dl.UniformSpider(0., 0., 3., .1, 0., softening=True)
odd_hard_unif_spider = dl.UniformSpider(0., 0., 3., .1, 0., softening=False)

fig, axes = plt.subplots(2, 4, figsize=(4*4, 2*3))

axes[0][0].set_title("Even Soft. Unif. Spider")
_map = axes[0][0].imshow(even_soft_unif_spider._aperture(coordinates))
fig.colorbar(_map, ax=axes[0][0])

axes[0][1].set_title("Even Soft. Unif. Spider")
_map = axes[0][1].imshow(even_hard_unif_spider._aperture(coordinates))
fig.colorbar(_map, ax=axes[0][1])

axes[0][2].set_title("Even Soft. Pos. x Tans. Unif. Spider")
_map = axes[0][2].imshow(pos_x_trans_unif_spider._aperture(coordinates))
fig.colorbar(_map, ax=axes[0][2])


axes[0][3].set_title("Even Soft. Neg. x Trans. Unif. Spider")
_map = axes[0][3].imshow(neg_x_trans_unif_spider._aperture(coordinates))
fig.colorbar(_map, ax=axes[0][3])

axes[1][0].set_title("Even Soft. Pos. y Trans. Unif. Spider")
_map = axes[1][0].imshow(pos_y_trans_unif_spider._aperture(coordinates))
fig.colorbar(_map, ax=axes[1][0])

axes[1][1].set_title("Even Soft. Neg. y Trans. Unif. Spider")
_map = axes[1][1].imshow(neg_y_trans_unif_spider._aperture(coordinates))
fig.colorbar(_map, ax=axes[1][1])

axes[1][2].set_title("Odd Soft. Unif. Spider")
_map = axes[1][2].imshow(odd_soft_unif_spider._aperture(coordinates))
fig.colorbar(_map, ax=axes[1][2])

axes[1][3].set_title("Odd Hard Unif. Spider")
_map = axes[1][3].imshow(odd_hard_unif_spider._aperture(coordinates))
fig.colorbar(_map, ax=axes[1][3])
plt.show()
