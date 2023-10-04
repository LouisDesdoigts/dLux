# Geometry

This module is used as for calculating geometric shapes on coordinate arrays, used to generate apertures. Each shape has two functions. One that generates a hard-edged oversampled one for more accurate aperture shapes, but is not differentiable/jittable. These are the `circle`, `square`, etc functions. The other type are differentiable and jittable in order to gain dynamic aperture that can be fit to data. These instead calculate the distance of each pixel from an edge and 'clip' the values between 0-1 for distances above a specified threshold. These functions are denoted with a `soft_` prefix, ie `soft_circle`, `soft_square`, etc.

There are also some other helper functions for scaling and downsampling.

TODO: Little tutorial on generating apertures.

---

???+ info "circle"
    ::: dLux.utils.geometry.circle

???+ info "square"
    ::: dLux.utils.geometry.square

???+ info "rectangle"
    ::: dLux.utils.geometry.rectangle

???+ info "reg_polygon"
    ::: dLux.utils.geometry.reg_polygon

???+ info "spider"
    ::: dLux.utils.geometry.spider

???+ info "combine"
    ::: dLux.utils.geometry.combine

???+ info "soft_circle"
    ::: dLux.utils.geometry.soft_circle

???+ info "soft_square"
    ::: dLux.utils.geometry.soft_square

???+ info "soft_rectangle"
    ::: dLux.utils.geometry.soft_rectangle

???+ info "soft_reg_polygon"
    ::: dLux.utils.geometry.soft_reg_polygon

???+ info "soft_spider"
    ::: dLux.utils.geometry.soft_spider
