import numpy as np
with open('test.pkl','rb') as f:
    data = np.load(f, allow_pickle=True)
    print(data)