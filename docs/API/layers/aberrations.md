# Aberrations

???+ info "Zernike"
    ::: dLux.layers.aberrations.Zernike

???+ info "ZernikeBasis"
    ::: dLux.layers.aberrations.ZernikeBasis

<!-- # Aberrations: aberrations.py

This module contains the classes that define the behaviour of Aberration Layers in dLux.

There are two public classes:

- `Zernike`
- `ZernikeBasis`

These classes are designed to generate Zernike polynomials dynamically. Typically, users should not need to interact with these classes directly, but rather through the `Aperture` classes.

The Zernike class hold a single Zernike polynomial. It has one main method, `.calculate(coordinates, nsides)` that takes in a set of coordinates and calculates the Zernike polynomial on those coordinates.

The ZernikeBasis class holds a set of Zernike polynomials. It has one main method, `.calculate_basis(coordinates, nsides)` that takes in a set of coordinates and calculates the Zernike polynomial basis on those coordinates.

??? info "Zernike Basis API"
    :::dLux.aberrations.ZernikeBasis

??? info "Zernike API"
    :::dLux.aberrations.Zernike

Let's construct a Zernike Basis:

```python
import dLux as dl
import jax.numpy as np

zernike_basis = dl.ZernikeBasis(np.arange(1, 7))
```

```python
> ZernikeBasis(
>   zernikes=[
>     Zernike(j=1, n=0, m=0, name='Piston', _k=f32[1], _c=f32[1]),
>     Zernike(j=2, n=1, m=1, name='Tilt X', _k=f32[1], _c=f32[1]),
>     Zernike(j=3, n=1, m=-1, name='Tilt Y', _k=f32[2], _c=f32[2]),
>     Zernike(j=4, n=2, m=0, name='Defocus', _k=f32[2], _c=f32[2]),
>     Zernike(j=5, n=2, m=-2, name='Astig X', _k=f32[3], _c=f32[3]),
>     Zernike(j=6, n=2, m=2, name='Astig Y', _k=f32[1], _c=f32[1])
>   ]
> )
```

??? abstract "Plotting code"
    ```python
    import matplotlib.pyplot as plt

    npix = 256
    coords = dl.utils.pixel_coords(npix, 2/npix)
    basis = zernike_basis.calculate_basis(coords)

    plt.figure(figsize=(15, 8))
    for i in range(len(basis)):
        plt.subplot(2, 3, i+1)
        plt.title(f"{zernike_basis.noll_indices[i].name}")
        plt.imshow(basis[i])
        plt.colorbar()
    plt.tight_layout()
    plt.savefig("assets/zernike.png")
    ```

![zernike](../assets/zernike.png) -->