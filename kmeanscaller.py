from pykmlocal import kmeans
def call_kmeans(data,n,k,algorithm,term):

    # data: nxd

    # term = (100, 0, 0, 0,  # run for 1 stage (maxTotStage)
    #  0.10,  # min consec RDL
    #  0.10,  # min accum RDL
    #  3,  # max run stages
    #  0.50,  # init. prob. of acceptance
    #  10,  # temp. run length
    #  0.95)

    data = data.reshape((n, -1))
    codebook = kmeans( data, k, algorithm,term)
    return codebook