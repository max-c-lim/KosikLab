"""
Old version of flipping y-coordinates:
    yi = y_max - yi

New version:
    yi = y_max - yi + y_min
    "+ y_min" is needed when y_min is not 0
"""

import numpy as np
import matplotlib.pyplot as plt

lower_bound = 150
upper_bound = 1000
n_points = 20

test_points = np.random.randint(low=lower_bound, high=upper_bound, size=n_points)
y_max = max(test_points)
y_min = min(test_points)
colors = np.random.random(n_points*3).reshape((n_points, 3))

# Plot original points
plt.scatter([0]*n_points, test_points, c=colors)

# Plot incorrect flipping
plt.scatter([1] * n_points, y_max - test_points, c=colors)

# Plot correct flipping
plt.scatter([2] * n_points, y_max - test_points + y_min, c=colors)

plt.ylim(0, upper_bound+100)
plt.xlim(-0.5, 2.5)
plt.xticks([0, 1, 2], ["Original", "Incorrect Flip", "Correct Flip"])
plt.show()
