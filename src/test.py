import matplotlib.pyplot as plt
import numpy as np

y = [141.2364,
120.1873,
102.0348,
86.5864,
73.5276,
62.5141,
53.2247,
45.3799,
38.7452,
33.1257,
28.3599,
24.3135,
20.8746,
17.9493,
15.4588,
13.3368,
11.5274,
9.9835,
8.6650,
7.5384,
6.5749,
5.7504,
5.0444,
4.4395,
3.9209]

x = np.arange(1, len(y) + 1)
plt.plot(x, y, marker='o', linestyle='-', color='b')
plt.title('Performance over Time')
plt.xlabel('Time')
plt.ylabel('Performance')
plt.xticks(x)
plt.yticks(y)
plt.grid()
plt.show()

