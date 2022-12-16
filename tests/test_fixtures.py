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
        return out.format(self.name, self.age, self.gender.name, self.height)


@pytest.fixture
def create_person():
    def _create_person(
            name: str = "John", 
            height: float = 1.845,
            age: int = 33, 
            gender: int = Gender.MALE):
        return Person(name, height, age, gender)
    return _create_person


def test_john(create_person: callable):
    corr: str = "I am John, a 33 years old MALE, who is 1.84m tall."
    assert str(create_person()) == corr 
