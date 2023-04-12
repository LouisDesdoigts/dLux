# Coordinates Utility Functions

This module contains a series of functions that are used to calculate coordinates arrays, apply transformations to coordinate those arrays, and convert between cartesian and polar coordiantes.

## Get Pixel Positions

This calculates the coordinates of pixel centers for an arbitrary number of dimensions, lengths and pixel scales. Each dimension can be shifted by an arbitrary amount using the `offset` parameter. 2D arrays can be output in polar coordinates by setting the `polar` flag.

??? info "Get Pixel Positions API"
    ::: dLux.utils.coordinates.get_pixel_positions

## Polar to Cartesian

Converts a polar coordinate array to cartesian coordinates.

??? info "Polar to Cartesian API"
    ::: dLux.utils.coordinates.polar_to_cartesian

## Cartesian to Polar

Converts a cartesian coordinate array to polar coordinates.

??? info "Cartesian to Polar API"
    ::: dLux.utils.coordinates.cartesian_to_polar

## Translate

Translates a coordinate array by a given amount.

??? info "Translate API"
    ::: dLux.utils.coordinates.translate

## Rotate

Rotates a coordinate array by a given amount.

??? info "Rotate API"
    ::: dLux.utils.coordinates.rotate

## Compress

Compresses a coordinate array by a given amount.

??? info "Compress API"
    ::: dLux.utils.coordinates.compress

## Shear

Shears a coordinate array by a given amount.

??? info "Shear API"
    ::: dLux.utils.coordinates.shear