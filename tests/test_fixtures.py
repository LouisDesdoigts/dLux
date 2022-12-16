import pytest
import enum


class Gender(enum.Enum):
    MALE = 0
    FEMALE = 1


class Person(object):
    name: str
    height: float
    age: int
    gender: int


    def __init__(self, name: str, height: float, age: int, gender: int):
        self.name = name
        self.height = height
        self.age = age
        self.gender = gender


    def __repr__(self) -> str:
        out: str = "I am {}, a {} years old {}, who is {:.2f}m tall." 
        return out.format(self.name, self.age, self.gender, self.height)


@pytest.fixture
def create_john():
    return Person("John", 1.845, 33, Gender.MALE)


@pytest.fixture
def create_charlie():
    return Person("Charlie", 1.543, 564)


def test_john(fixture: callable):
    corr: str = "I am John, a 33 years old Gender.MALE, who is 1.84m tall"
    assert str(fixture) == corr 
