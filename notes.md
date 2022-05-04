### Notes:
- Currently we only work with even sized arrays
- We need to figure out the centering convetion for psfs (ie central pixel or corner of pixel)
- Offset is now passed through ALL layers so that CreateWavefront does not have the be the first layer, ie you could use the offest term in the MFT
- Add a 'check_model' function to OpticalSystem that checks there are no incongruities in pixelscales with the given wavels
- Add a similar function for the interpolation layer to make sure interpoaltion ratio > 2