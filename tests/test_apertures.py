import dLux 
import jax.numpy as np
import typing

Aperture = typing.TypeVar("Aperture")
Array = typing.TypeVar("Array")



class TestAperturesCommonInterfaces():
    """
    For each type of aperture that has common properties, test it
    """
    def _assert_valid_hard_aperture(aperture, msg=''):
        assert ((aperture == 1.) | (aperture == 0.)).all(), msg
        
    def _assert_valid_soft_aperture(aperture, msg=''):
        assert (aperture <= 1. + np.finfo(np.float32).resolution).all(), msg
        assert (aperture >= 0. - np.finfo(np.float32).resolution).all(), msg
        # there should also exist something in bounds (assuming edges are within coords)
        assert np.logical_and(aperture > 0., aperture < 1.).any(), msg
        
    def test_all_apertures(self, create_square_aperture : callable, create_rectangular_aperture : callable, create_circular_aperture : callable):
        
        constructors = [create_square_aperture,
                        create_rectangular_aperture,
                        create_circular_aperture]
        
        # TODO might need to add error message for when this fails but it checks things very easily
        for ctor in constructors:
            self._test_single_aperture_class(ctor)
    
    def _test_single_aperture_class(self, aperture_fixture):
        
        coords = dLux.utils.get_pixel_coordinates(512, 2./512)
        
        x_offset = 1.
        y_offset = 1.
        centres = [np.array([0., 0.]),
                   np.array([x_offset, 0.]),
                   np.array([0., y_offset]),
                   np.array([x_offset, y_offset])]
        
        rotations = [0, np.pi/2.]
        not_rotatable_apertures = (dLux.apertures.CircularAperture,
                                   dLux.apertures.AnnularAperture)
        
        base_kwargs = {"centre" : None,
                       "softening" : None,
                       "occulting" : None
                       }

        for centre in centres:
            for rotation in rotations:
                for softening in [True, False]:
                    for occulting in [True, False]:
                        actual_kwargs = base_kwargs
                        actual_kwargs["centre"] = centre
                        actual_kwargs["softening"] = softening
                        actual_kwargs["occulting"] = occulting
                        
                        if not isinstance(aperture_fixture(), not_rotatable_apertures):
                            actual_kwargs["rotation"] = rotation
                            
                        
                        aperture = aperture_fixture(**actual_kwargs)._aperture(coords)
                        
                        msg = f'{actual_kwargs}, on ctor {aperture_fixture}'
                        
                        if softening:
                            TestAperturesCommonInterfaces._assert_valid_soft_aperture(aperture, msg)
                        else:
                            TestAperturesCommonInterfaces._assert_valid_hard_aperture(aperture, msg)
                            

# the above for loops essentially run the following tests for each aperture

# class TestSquareAperture():
#     """
#     Provides unit tests for the `Square Aperture` class. 

#     Parameters:
#     -----------
#     utility: SquareApertureUtility
#         Provides default parameter values and coordinate systems. 
#     """


#     def test_constructor(self, create_square_aperture : callable) -> None:
#         """
#         Checks that all of the fields are correctly set. 
#         """
#         # Case default
#         sq_ap = create_square_aperture()

#         # TODO check that these don't actually test anything?
#         # assert sq_ap.occulting == self.utility.occulting
#         # assert sq_ap.softening == np.inf 
#         # assert sq_ap.x_offset == self.utility.x_offset
#         # assert sq_ap.y_offset == self.utility.y_offset
#         # assert sq_ap.width == self.utility.width
#         # assert sq_ap.theta == self.utility.theta 

#         # Case Translated X
#         x_offset = 1.
#         centre = np.array([x_offset, 0.])
#         sq_ap = create_square_aperture(centre = centre)

#         assert (sq_ap.centre == centre).all()

#         # Case Translated Y
#         y_offset = 1.
#         centre = np.array([0., y_offset])
#         sq_ap = create_square_aperture(centre = centre)

#         assert (sq_ap.centre == centre).all()

#         # Case Rotated Clockwise
#         rotation = np.pi / 2.
#         sq_ap = create_square_aperture(rotation = rotation)

#         assert sq_ap.rotation == rotation


#     def test_range_hard(self, create_square_aperture : callable) -> None:
#         """
#         Checks that the apertures fall into the correct range.
#         """
#         coords = dLux.utils.get_pixel_coordinates(512, 2./512)

#         # Case Translated X
#         x_offset = 1.
#         centre = np.array([x_offset, 0.])
#         aperture = create_square_aperture(centre = centre)._aperture(coords)

#         TestSquareAperture._check_valid_hard_aperture(aperture)

#         # Case Translated Y
#         y_offset = 1.
#         centre = np.array([0., y_offset])
#         aperture = create_square_aperture(centre = centre)._aperture(coords)

#         TestSquareAperture._check_valid_hard_aperture(aperture)

#         # Case Rotated 
#         rotation = np.pi / 2.
#         aperture = create_square_aperture(rotation = rotation)._aperture(coords)

#         TestSquareAperture._check_valid_hard_aperture(aperture)

#         # Case Occulting
#         aperture = create_square_aperture(occulting = True)._aperture(coords)

#         TestSquareAperture._check_valid_hard_aperture(aperture)

#         # Case Not Occulting
#         aperture = create_square_aperture(occulting = False)._aperture(coords)

#         TestSquareAperture._check_valid_hard_aperture(aperture)

#     def _check_valid_hard_aperture(aperture):
#         assert ((aperture == 1.) | (aperture == 0.)).all()
        
#     def _check_valid_soft_aperture(aperture):
#         assert (aperture <= 1.).all()
#         assert (aperture >= 0.).all()
#         # there should also exist something in bounds (assuming edges are within coords)
#         assert np.logical_and(aperture > 0., aperture < 1.).any()

#     def test_range_soft(self, create_square_aperture : callable) -> None:
#         """
#         Checks that the aperture falls into the correct range.
#         """
#         coords = dLux.utils.get_pixel_coordinates(512, 2./512)
#         softening = True
        
#         # Case Translated X
#         x_offset = 1.
#         centre = np.array([x_offset, 0.])
#         aperture = create_square_aperture(centre=centre, softening=softening)._aperture(coords)

#         import matplotlib.pyplot as plt
#         TestSquareAperture._check_valid_soft_aperture(aperture)

#         # Case Translated Y
#         y_offset = 1.
#         centre = np.array([0., y_offset])
#         aperture = create_square_aperture(centre=centre, softening=softening)._aperture(coords)

#         TestSquareAperture._check_valid_soft_aperture(aperture)

#         # Case Rotated 
#         rotation = np.pi / 2.
#         aperture = create_square_aperture(rotation=rotation, softening=softening)._aperture(coords)

#         TestSquareAperture._check_valid_soft_aperture(aperture)

#         # Case Occulting
#         aperture = create_square_aperture(centre=centre, softening=softening)._aperture(coords)

#         TestSquareAperture._check_valid_soft_aperture(aperture)

#         # Case Not Occulting
#         aperture = create_square_aperture(centre=centre, softening=softening)._aperture(coords)

#         TestSquareAperture._check_valid_soft_aperture(aperture)