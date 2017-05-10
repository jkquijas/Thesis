import numpy as np

def test_perm():
    return np.random.permutation(np.array([1,2,3,4,5]))
def main():
    x=np.random.rand(10)
    print x
    cw=3
    n_cols = len(x)
    for i in range(0, n_cols, cw):
        print x[i:min(i+cw,n_cols)]
        print len(x[i:min(i+cw,n_cols)])

main()
