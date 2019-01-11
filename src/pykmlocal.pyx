import numpy as np
from cython.operator cimport dereference as deref
cimport numpy as np
cimport cython

cdef extern from "KMeans.h":
    ctypedef double *KMpoint
    ctypedef double *KMcenter
    ctypedef double *KMdataPoint

cdef extern from "KMterm.h":
    cdef cppclass KMterm:
        KMterm(double, double, double, double,
               double, double, int,
               double, int, double)

cdef extern from "KMlocal.h":
    cdef cppclass KMlocalLloyds:
        KMlocalLloyds(KMfilterCenters, KMterm)
        KMfilterCenters execute()
        KMfilterCenters* execute_to_new()
    cdef cppclass KMlocalSwap:
        KMlocalSwap(KMfilterCenters, KMterm)
        KMfilterCenters execute()
        KMfilterCenters* execute_to_new()
    cdef cppclass KMlocalEZ_Hybrid:
        KMlocalEZ_Hybrid(KMfilterCenters, KMterm)
        KMfilterCenters execute()
        KMfilterCenters* execute_to_new()
    cdef cppclass KMlocalHybrid:
        KMlocalHybrid(KMfilterCenters, KMterm)
        KMfilterCenters execute()
        KMfilterCenters* execute_to_new()

cdef extern from "KMdata.h":
    cdef cppclass KMdata:
        KMdata( int dim, int maxPts )

        KMdataPoint& operator[](int)
        KMdataPoint& get(int)

        void buildKcTree()

# cdef extern from "KMcenters.h":
#     cdef cppclass KMcenters:
#         KMcenters( int, KMdata)
#         int getK()
#         int getDim()
#         KMcenter get(int)

cdef extern from "KMfilterCenters.h":
    #cdef cppclass KMfilterCenters(KMcenters):
    cdef cppclass KMfilterCenters:
        KMfilterCenters( int, KMdata)
        int getK()
        int getDim()
        KMcenter get(int)

cdef class KMLocal:
    cdef KMdata* dataPts
    cdef int k
    cdef int numFeatures

    def __init__(self, np.ndarray[np.float_t, ndim=2] data, int k_):
        cdef int maxPts
        cdef int nPts = 0
        cdef int d
        cdef KMdataPoint p

        maxPts           = data.shape[0]
        self.numFeatures = data.shape[1]

        self.dataPts = new KMdata(self.numFeatures, maxPts) # allocate data storage

        with cython.boundscheck(False):
            for nPts in range( maxPts ):
                #p = self.dataPts[nPts] # XXX this causes trouble with Cython 0.16.
                p = self.dataPts.get(nPts)
                for d in range( self.numFeatures ):
                    p[d] = data[nPts, d]
        self.dataPts.buildKcTree()

        self.k = k_

    def run(self, algorithm, a,  b,  c,  d, mcr,  mar,  mrs, ipa,  trl,  trf):
        # allocate new centers
        cdef KMfilterCenters* ctrs
        ctrs = new KMfilterCenters( self.k, deref(self.dataPts))

        # allocate termination critereon
        cdef KMterm *term = new KMterm(a, b, c, d,    #  run for 1 stage (maxTotStage)
                                  mcr,			#  min consec RDL
                                  mar,			#  min accum RDL
                                  mrs,			#  max run stages
                                  ipa,			#  init. prob. of acceptance
                                  trl,			#  temp. run length
                                  trf)			#  temp. reduction factor
        cdef KMlocalLloyds* kmLloyds
        cdef KMlocalSwap* kmSwap
        cdef KMlocalEZ_Hybrid* kmEZHybrid
        cdef KMlocalHybrid* kmHybrid
        cdef int i
        cdef int j
        cdef KMcenter center
        cdef np.ndarray[np.float_t, ndim=2] codebook

        if algorithm=='lloyd':
             kmLloyds = new KMlocalLloyds(deref(ctrs), deref(term)) # repeated Lloyd's
             ctrs = kmLloyds.execute_to_new()
        elif algorithm == 'swap':
            kmSwap = new KMlocalSwap(deref(ctrs), deref(term))
            ctrs = kmSwap.execute_to_new()
        elif algorithm == 'EZ-hybrid':
            km_ez_hybrid = new KMlocalEZ_Hybrid(deref(ctrs), deref(term))
            ctrs = km_ez_hybrid.execute_to_new()
        elif algorithm == 'hybrid':
            km_hybrid = new KMlocalHybrid(deref(ctrs), deref(term))
            ctrs = km_hybrid.execute_to_new()
        else:
            raise ValueError('unknown algorithm "%s"'%algorithm)

        assert ctrs.getDim() == self.numFeatures
        codebook = np.empty( (ctrs.getK(), ctrs.getDim()), dtype=np.float)
        with cython.boundscheck(False):
            for i in range( ctrs.getK() ):
                center = ctrs.get(i)
                for j in range( ctrs.getDim() ):
                    codebook[i,j] = center[j]

        return codebook

def kmeans( data, k, algorithm,term):
    kml = KMLocal( data, k )
    codebook = kml.run(algorithm,term[0],term[1],term[2],term[3],term[4],term[5],term[6],term[7],term[8],term[9])
    return codebook
