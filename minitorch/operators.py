"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    return float(x * y)


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return float(x + y)


def neg(x: float) -> float:
    return -float(x)


def lt(x: float, y: float) -> bool:
    return x < y


def eq(x: float, y: float) -> bool:
    return x == y


def max(x: float, y: float) -> float:
    return float(x) if x > y else float(y)


def is_close(x: float, y: float) -> bool:
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))

def sigmoid_back(x:float, y:float) -> float:
    return y * sigmoid(x) * (1 - sigmoid(x))


def relu(x: float) -> float:
    x = float(x)
    x = x if x > 0 else 0.0
    return x


def log(x: float) -> float:
    return math.log(x)


def log_back(x: float, y: float) -> float:
    return y / x

def exp(x: float) -> float:
    return math.exp(x)

def exp_back(x: float, y:float) -> float:
    return math.exp(x) * y

def inv(x: float) -> float:
    return 1.0 / x


def inv_back(x: float, y: float) -> float:
    return -1.0 / (x**2) * y


def relu_back(x: float, y: float) -> float:
    return float(y) if x > 0 else 0.0



# Small practice library of elementary higher-order functions.

def map(fn: Callable[[float], float], l: Iterable[float]) -> Iterable[float]:
    return [fn(x) for x in l]


def zipWith(
    fn: Callable[[float, float], float], l1: Iterable[float], l2: Iterable[float]
) -> Iterable[float]:
    return [fn(x, y) for x, y in zip(l1, l2)]


def reduce(
    fn: Callable[[float, float], float], l: Iterable[float], init: float
) -> float:
    return init if not l else reduce(fn, l[1:], fn(init, l[0]))


def negList(l: Iterable[float]) -> Iterable[float]:
    return map(neg, l)


def addLists(l1: Iterable[float], l2: Iterable[float]) -> Iterable[float]:
    return zipWith(add, l1, l2)


def sum(l: Iterable[float]) -> float:
    return reduce(add, l, 0)


def prod(l: Iterable[float]) -> float:
    return reduce(mul, l, 1)
