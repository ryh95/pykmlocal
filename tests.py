from pykmlocal import kmeans
import numpy as np

def test_kmeans():
    data = np.array([[0, 0],
                     [0, 1],
                     [1, 1],
                     [-1, 1],

                     [10,10],
                     [11,10],
                     [10,11],
                     [10,10.1],

                     [-100, -100],
                     [-101, -100],
                     [-102, -100],
                     ],
                    dtype=np.float)
    term = (100, 0, 0, 0,  # run for 1 stage (maxTotStage)
     0.10,  # min consec RDL
     0.10,  # min accum RDL
     3,  # max run stages
     0.50,  # init. prob. of acceptance
     10,  # temp. run length
     0.95)
    codebook = kmeans( data, 3, 'hybrid',term)
    # XXX should do test here
    return codebook

def test_kmeans_random():
    data = np.random.randn( 1000, 3 )
    data[500:] += np.array([ 10,10,10])
    codebook1 = kmeans( data, 2)

    codebook2 = kmeans( data, 2)

    codebook3 = kmeans( data, 2)
    allclose = (np.allclose( codebook1, codebook2 ) and
                np.allclose( codebook2, codebook3))
    assert not allclose
    # XXX should do more tests here

test_kmeans()