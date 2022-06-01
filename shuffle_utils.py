from sklearn.model_selection import ShuffleSplit
import numpy as np

def triple_shuffle_split(data_size=10, train_size=0.6, valid_size=0.3, test_size=0.1, random_state=123):

    data_idx = range(data_size)
    epsilon = 1e-5

    total = train_size + valid_size + test_size
    if total != 1.0:
        train_size /= total
        valid_size /= total
        test_size /= total

    # make up for small decimal errors
    test_size = 1.0 - train_size - valid_size

    l_size = 1.0 - test_size
    r_size = test_size
    assert abs(l_size + r_size - 1.0) <= epsilon
    rs1 = ShuffleSplit(n_splits=1, train_size=l_size, test_size=r_size, random_state=random_state) 
    tmp_idx = test_idx = []
    for l, r in rs1.split(data_idx):
        tmp_idx = np.array(data_idx)[list(l)]
        test_idx = np.array(data_idx)[list(r)]

    train_idx = valid_idx = []
    l_size = train_size/(train_size+valid_size)
    r_size = valid_size/(train_size+valid_size)
    assert abs(l_size + r_size - 1.0) <= epsilon

    if random_state is not None:
        random_state += 1
    
    rs2 = ShuffleSplit(n_splits=1, train_size=l_size, test_size=r_size, random_state=random_state) 
    for l, r in rs2.split(tmp_idx):
        train_idx = np.array(tmp_idx)[list(l)]
        valid_idx = np.array(tmp_idx)[list(r)]

    all_idx = []
    all_idx.extend(train_idx)
    all_idx.extend(valid_idx)
    all_idx.extend(test_idx)
    all = [len(train_idx),len(valid_idx),len(test_idx)]
    assert np.sum(all) == len(all_idx) == data_size

    return train_idx, valid_idx, test_idx

if __name__ == '__main__':

    train_idx, valid_idx, test_idx = \
        triple_shuffle_split(data_size=100, train_size=0.8, valid_size=0.1, test_size=0.1, random_state=123)
    assert [len(train_idx),len(valid_idx),len(test_idx)]  == [80,10,10]

    train_idx, valid_idx, test_idx = \
        triple_shuffle_split(data_size=100, train_size=0.6, valid_size=0.3, test_size=0.1, random_state=123)
    assert [len(train_idx),len(valid_idx),len(test_idx)]  == [60,30,10]

    train_idx, valid_idx, test_idx = \
        triple_shuffle_split(data_size=100, train_size=0.5, valid_size=0.3, test_size=0.2, random_state=123)
    assert [len(train_idx),len(valid_idx),len(test_idx)]  == [50,30,20]

    print("done")
