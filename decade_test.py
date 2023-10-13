import math
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

def get_data():
    """
    The code will generate a set of random `x` values
    """
    # Generate a uniformly distributed set of random numbers in the range from
    # 0 to 2π, which covers a complete sine wave oscillation
    x_values = np.random.uniform(low=0, high=2 * math.pi,
                                size=1000).astype(np.float32)

    # Shuffle the values to guarantee they're not in order
    np.random.shuffle(x_values)

    return x_values


def generator(num_samples=500):
    for i in range(num_samples):
        yield [x_values[i].reshape(1, 1)]

x_values = get_data()
y = generator()
for i in y:
    print(i)