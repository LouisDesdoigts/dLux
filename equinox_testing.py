import equinox
import jax
import typing
import pytest


A = typing.NewType("A", equinox.Module)
B = typing.NewType("B", equinox.Module)


class A(equinox.Module):
    a : int


    def __init__(self, a : int):
        self.a = a


    def get_a(self) -> int:
        return self.a


    def set_a(self, a : int) -> A:
        return equinox.tree_at(lambda pytree : pytree.a, self, a)


    def add(self, letter : int) -> int:
        return self.get_a() + letter


class B(A):
    b : int


    def __init__(self, a : int, b : int) -> B:
        super().__init__(a)
        self.b = b


    def get_b(self) -> int:
        return self.b


    def set_b(self, b : int) -> B:
        return equinox.tree_at(lambda pytree : pytree.b, self, b)


    def add(self, letter : int) -> int:
        return self.get_b() + self.get_a() + letter


class TestA():
    def test_constructor(self):
        a = A(0)
        assert a.a == 0


    def test_accessor(self):
        a = A(0)
        assert a.get_a() == 0


    def test_mutator(self):
        a = A(0)
        a = a.set_a(1)
        assert a.get_a() == 1


    def test_add(self):
        a = A(0)
        assert a.add(1) == 1


class TestB():
    def test_constructor(self):
        b = B(0, 1)
        assert b.b == 1
        assert b.a == 0


    def test_accessors(self):
        b = B(0, 1)
        assert b.get_a() == 0
        assert b.get_b() == 1


    def test_mutators(self):
        b = B(0, 1)
        b = b.set_a(1)
        b = b.set_b(2)
        assert b.get_a() == 1
        assert b.get_b() == 2


    def test_add(self):
        b = B(0, 1)
        assert b.add(1) == 2



