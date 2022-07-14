from dLux import *

def toliman_layers(extra_layers=[],
                   in_focus = True,
                   wf_npix = 256, 
                   det_npix = 256,
                   aperture = 0.125,
                   m2 = 0.02,
                   fl = 1.32,
                   fl_shift = 0.,
                   osamp = 5, 
                   det_pixsize = 6.5e-6
                   ):
    """
    Returns Toliman layers
    """
    
    det_pixsize /= osamp
        
    layers = [
        CreateWavefront(wf_npix, aperture),
        TiltWavefront(),
        CircularAperture(wf_npix, rmin=m2/aperture, eps=1e-7),
        NormaliseWavefront()]
    [layers.append(layer) for layer in extra_layers]
    if in_focus:
        layers.append(MFT(det_npix, fl, det_pixsize))
    else:
        layers.append(FresnelProp(det_npix, fl, fl_shift, det_pixsize))
    
    return layers