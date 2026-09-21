import os
import urllib.request
import numpy as np

URL = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
DEST = "data/sift-128-euclidean.hdf5"


def download():
    os.makedirs("data", exist_ok=True)
    if os.path.exists(DEST):
        print("already downloaded")
        return
    print("downloading ~500MB")
    urllib.request.urlretrieve(URL, DEST)


def verify():
    from annlib.dataset import load_raw

    train, test, truth = load_raw(DEST)
    assert train.shape == (1_000_000, 128), train.shape
    assert test.shape == (10_000, 128), test.shape
    assert truth.shape == (10_000, 100), truth.shape
    assert train.dtype == np.float32

    # Brute force query 0 against the full database
    # and compare with the first 10 stored neighbour ids.
    d = ((train - test[0]) ** 2).sum(axis=1)
    mine = np.argsort(d)[:10]
    print("mine  ", mine)
    print("stored", truth[0][:10])
    assert set(mine.tolist()) == set(truth[0][:10].tolist())
    print("data verified")


if __name__ == "__main__":
    download()
    verify()