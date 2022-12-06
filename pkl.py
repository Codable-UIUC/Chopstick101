import numpy as np
import pickle
with open('c.pkl','rb') as f:
    # data = np.load(f, allow_pickle=True)
    data = pickle.load(f)
    print(data.shape)