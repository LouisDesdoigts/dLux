from types import SimpleNamespace

from dLux._exports import reexport


def test_reexport_deduplicates_names():
    first = SimpleNamespace(__all__=("shared",), shared=1)
    second = SimpleNamespace(__all__=("shared", "unique"), shared=2, unique=3)
    namespace = {}

    exported = reexport((first, second), namespace)

    assert exported == ["shared", "unique"]
    assert namespace == {"shared": 2, "unique": 3}
