# Coordinates Utility Functions

This module contains a series of functions which are used to calculate coordinate arrays, apply transformations to coordinate arrays, and convert between Cartesian and polar coordinates.

## Polar to Cartesian

Converts a polar coordinate array to Cartesian coordinates.

??? info "Polar to Cartesian API"
    ::: dLux.utils.coordinates.polar_to_cart

## Cartesian to Polar

Converts a Cartesian coordinate array to polar coordinates.

??? info "Cartesian to Polar API"
    ::: dLux.utils.coordinates.cart_to_polar

## Pixel Coords

Calculates the coordinates of the pixel centers for the given input,
    assuming an equal size and pixel scale in all dimensions. Similar to `pixel_coordinates()`, but with a simplified API.

??? info "Pixel Coords API"
    ::: dLux.utils.coordinates.pixel_coords

## Pixel Coordinates

Calculates the coordinates of the pixel centers for the given input, without the underlying assumptions of `pixel_coords()`.

??? info "Pixel Coordinates API"
    ::: dLux.utils.coordinates.pixel_coordinates