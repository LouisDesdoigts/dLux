from __future__ import annotations
from utilities import Utility, UtilityUser
import jax.numpy as np
import dLux
from optax import GradientTransformation, MultiTransformState, adam
from jax import config
config.update("jax_debug_nans", True)


class BaseUtility(Utility):
    """
    Utility for the Base class.
    """
    param1 : float
    param2 : float


    class A(dLux.base.ExtendedBase):
        """
        Test subclass to test the Base methods
        """
        param : float
        b     : B


        def __init__(self, param, b):
            """
            Constructor for the Base testing class
            """
            self.param = param
            self.b = b


        def model(self):
            """
            Sample modelling function
            """
            return self.param**2 + self.b.param**2


    class B(dLux.base.ExtendedBase):
        """
        Test subclass to test the Base methods
        """
        param : float


        def __init__(self, param):
            """
            Constructor for the Base testing class
            """
            self.param = param


    def __init__(self : Utility):
        """
        Constructor for the Optics Utility.
        """
        self.param1 = 1.
        self.param2 = 1.


    def construct(self : Utility,
                  param1 : float = None,
                  param2 : float = None):
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        param1 = self.param1 if param1 is None else param1
        param2 = self.param2 if param2 is None else param2
        return self.A(param1, self.B(param2))


class ExtendedBaseUtility(BaseUtility):
    """
    Utility for the Base class.
    """
    pass


class TestBase(UtilityUser):
    """
    Tests the Base class.
    """
    utility : BaseUtility = BaseUtility()


    def test_unwrap(self):
        """
        Test the _unwrap method
        """
        base = self.utility.construct()

        # Test unwrapping
        wrapped_a = ['a', ['b', ['c', ['d']]]]
        wrapped_b = [[[['a'], 'b'], 'c'], 'd']
        wrapped_c = ['a', 'b', 'c', 'd']

        assert base._unwrap(wrapped_a) == ['a', 'b', 'c', 'd']
        assert base._unwrap(wrapped_b) == ['a', 'b', 'c', 'd']
        assert base._unwrap(wrapped_c) == ['a', 'b', 'c', 'd']

        # Test path maps
        pmap = {'1' : 'a.b', '2' : 'b.c', '3' : 'c.d', '4' : 'd.e'}
        wrapped_a = ['1', ['2', ['3', ['4']]]]
        wrapped_b = [[[['1'], '2'], '3'], '4']
        wrapped_c = ['1', '2', '3', '4']

        assert base._unwrap(wrapped_a, pmap=pmap) == ['a.b', 'b.c', 'c.d', 'd.e']
        assert base._unwrap(wrapped_b, pmap=pmap) == ['a.b', 'b.c', 'c.d', 'd.e']
        assert base._unwrap(wrapped_c, pmap=pmap) == ['a.b', 'b.c', 'c.d', 'd.e']

        # Test with values
        wrapped_a = ['a', ['b', 'c', 'd']]
        wrapped_b = [['a', 'b', 'c'], 'd']
        wrapped_c = ['a', 'b', 'c', 'd']

        assert base._unwrap(wrapped_a, [1, 2])[1]       == [1, 2, 2, 2]
        assert base._unwrap(wrapped_b, [1, 2])[1]       == [1, 1, 1, 2]
        assert base._unwrap(wrapped_c, [1, 2, 3, 4])[1] == [1, 2, 3, 4]


    def test_format(self):
        """
        test the _format method
        """
        base = self.utility.construct()

        # Test formatting
        path_a = ['a.b', ['b.c', ['c.d', ['d.e']]]]
        path_b = [[[['a.b'], 'b.c'], 'c.d'], 'd.e']
        path_c = ['a.b', 'b.c', 'c.d', 'd.e']

        assert base._format(path_a) == [['a', 'b'], ['b', 'c'], ['c', 'd'], ['d', 'e']]
        assert base._format(path_b) == [['a', 'b'], ['b', 'c'], ['c', 'd'], ['d', 'e']]
        assert base._format(path_c) == [['a', 'b'], ['b', 'c'], ['c', 'd'], ['d', 'e']]

        # Test path maps
        pmap = {'1' : 'a.b', '2' : 'b.c', '3' : 'c.d', '4' : 'd.e'}
        path_a = ['1', ['2', ['3', ['4']]]]
        path_b = [[[['1'], '2'], '3'], '4']
        path_c = ['1', '2', '3', '4']

        assert base._format(path_a, pmap=pmap) == [['a', 'b'], ['b', 'c'], ['c', 'd'], ['d', 'e']]
        assert base._format(path_b, pmap=pmap) == [['a', 'b'], ['b', 'c'], ['c', 'd'], ['d', 'e']]
        assert base._format(path_c, pmap=pmap) == [['a', 'b'], ['b', 'c'], ['c', 'd'], ['d', 'e']]

        # Test with values
        path_a = ['a.b', ['b.c', 'c.d', 'd.e']]
        path_b = [['a.b', 'b.c', 'c.d'], 'd.e']
        path_c = ['a.b', 'b.c', 'c.d', 'd.e']

        assert base._format(path_a, [1, 2])[1]       == [1, 2, 2, 2]
        assert base._format(path_b, [1, 2])[1]       == [1, 1, 1, 2]
        assert base._format(path_c, [1, 2, 3, 4])[1] == [1, 2, 3, 4]


    def test_get(self):
        """
        tests the get method
        """
        # Define parameters and construct base
        param1 = 5.
        param2 = 10.
        base = self.utility.construct(param1, param2)

        # Define paths
        param1_path = 'param'
        param2_path = 'b.param'

        assert base.get(param1_path) == param1
        assert base.get(param2_path) == param2


    def test_set(self):
        """
        tests the set method
        """
        # Define parameters and construct base
        param1 = 5.
        param2 = 10.
        base = self.utility.construct()

        # Define paths
        p1 = 'param'
        p2 = 'b.param'

        # Test set
        new_base = base.set([p1, p2], [param1, param2])
        assert new_base.param   == param1
        assert new_base.b.param == param2


    def test_add(self):
        """
        tests the add method
        """
        # Define parameters and construct base
        param1 = 5.
        param2 = 10.
        base = self.utility.construct()

        # Define paths
        p1 = 'param'
        p2 = 'b.param'

        # Test add
        new_base = base.add([p1, p2], [param1, param2])
        assert new_base.param   == base.get(p1) + param1
        assert new_base.b.param == base.get(p2) + param2


    def test_multiply(self):
        """
        tests the multiply method
        """
        # Define parameters and construct base
        param1 = 5.
        param2 = 10.
        base = self.utility.construct()

        # Define paths
        p1 = 'param'
        p2 = 'b.param'

        # Test multiply
        new_base = base.multiply([p1, p2], [param1, param2])
        assert new_base.param   == base.get(p1) * param1
        assert new_base.b.param == base.get(p2) * param2


    def test_divide(self):
        """
        tests the divide method
        """
        # Define parameters and construct base
        param1 = 5.
        param2 = 10.
        base = self.utility.construct()

        # Define paths
        p1 = 'param'
        p2 = 'b.param'

        # Test divide
        new_base = base.divide([p1, p2], [param1, param2])
        assert new_base.param   == base.get(p1) / param1
        assert new_base.b.param == base.get(p2) / param2


    def test_power(self):
        """
        tests the power method
        """
        # Define parameters and construct base
        param1 = 5.
        param2 = 10.
        base = self.utility.construct(2., 2.)

        # Define paths
        p1 = 'param'
        p2 = 'b.param'

        # Test power
        new_base = base.power([p1, p2], [param1, param2])
        assert new_base.param   == base.get(p1)**param1
        assert new_base.b.param == base.get(p2)**param2


    def test_min(self):
        """
        tests the min method
        """
        # Define parameters and construct base
        param1 = 5.
        param2 = 10.
        base = self.utility.construct()

        # Define paths
        p1 = 'param'
        p2 = 'b.param'

        # Test min
        new_base = base.min([p1, p2], [param1, param2])
        assert new_base.param   == np.minimum(base.get(p1), param1)
        assert new_base.b.param == np.minimum(base.get(p2), param2)


    def test_max(self):
        """
        tests the max method
        """
        # Define parameters and construct base
        param1 = 5.
        param2 = 10.
        base = self.utility.construct()

        # Define paths
        p1 = 'param'
        p2 = 'b.param'

        # Test max
        new_base = base.max([p1, p2], [param1, param2])
        assert new_base.param   == np.maximum(base.get(p1), param1)
        assert new_base.b.param == np.maximum(base.get(p2), param2)


    def test_apply(self):
        """
        tests the  method
        """
        # Define parameters and construct base
        fn1 = lambda x: x + 5.
        fn2 = lambda x: x + 10.
        base = self.utility.construct()

        # Define paths
        p1 = 'param'
        p2 = 'b.param'

        # Test 
        new_base = base.apply([p1, p2], [fn1, fn2])
        assert new_base.param   == base.get(p1) + 5.
        assert new_base.b.param == base.get(p2) + 10.


    def test_apply_args(self):
        """
        tests the apply_args method
        """
        # Define parameters and construct base
        fn = lambda x, a: x + a
        param1 = 5.
        param2 = 10.
        base = self.utility.construct()

        # Define paths
        p1 = 'param'
        p2 = 'b.param'

        # Test apply_args
        new_base = base.apply_args([p1, p2], fn, [(param1,), (param2,)])
        assert new_base.param   == base.get(p1) + param1
        assert new_base.b.param == base.get(p1) + param2


class TestExtendedBase(UtilityUser):
    """
    Tests the ExtendedBase class.
    """
    utility : ExtendedBaseUtility = ExtendedBaseUtility()


    def test_get_args(self):
        """
        tests the get_filter_spec method
        """
        # Define parameters and construct base
        base = self.utility.construct()

        # Define paths
        p1 = 'param'
        p2 = 'b.param'

        # Test paths
        args = base.get_args([p1, p2])
        assert args.param   == True
        assert args.b.param == True


    def test_get_param_spec(self):
        """
        tests the get_param_spec method
        """
        # Define parameters and construct base
        base = self.utility.construct()

        # Define paths & groups
        p1 = 'param'
        p2 = 'b.param'
        groups = ['group1', 'group2']

        # Test paths
        param_spec = base.get_param_spec([p1, p2], groups)
        assert param_spec.param   == groups[0]
        assert param_spec.b.param == groups[1]


    def test_get_optimiser(self):
        """
        tests the get_optimiser method
        """
        # Define parameters and construct base
        base = self.utility.construct()

        # Define paths & groups
        p1 = 'param'
        p2 = 'b.param'
        optimisers = [adam(0), adam(1)] # These are actually arbitrary

        # Test paths
        optim, opt_state = base.get_optimiser([p1, p2], optimisers)
        assert isinstance(optim, GradientTransformation)
        assert isinstance(opt_state, MultiTransformState)


    def test_update_and_model(self):
        """
        tests the update_and_model method
        """
        # Define parameters and construct base
        param1 = 2.
        param2 = 4.
        base = self.utility.construct()

        # Define paths & groups
        p1 = 'param'
        p2 = 'b.param'
        values = [param1, param2]

        # Test paths
        new_base = base.update_and_model("model", [p1, p2], values)
        assert new_base == param1**2 + param2**2

    def test_apply_and_model(self):
        """
        tests the apply_and_model method
        """
        # Define parameters and construct base
        base = self.utility.construct()

        # Define paths & groups
        p1 = 'param'
        p2 = 'b.param'
        fn1 = lambda x: x * 2.
        fn2 = lambda x: x * 4.
        fns = [fn1, fn2]

        # Test paths
        new_base = base.apply_and_model("model", [p1, p2], fns)
        assert new_base == (base.get(p1)*2.)**2 + (base.get(p2)*4.)**2