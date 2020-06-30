import numpy as np

p = [0.1, 0.1, 0.8]
for _ in range(10):
    a = np.random.choice(len(p), p=p)
    print(a)