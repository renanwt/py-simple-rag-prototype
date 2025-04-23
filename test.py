import numpy as np
array = [1.9, 1.3, 4.5, 2.3, 7, 10, 1]
top_k = 3

print(np.argpartition(array, range(top_k))[0:top_k])