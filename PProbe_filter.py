from __future__ import division
#generic imports
import sys,math,ast,copy
import numpy as np
#utility class of agnostic functions that should work
#under all python versions with/without scipy


class Filters:
      def __init__(self,verbose=False):
            self.verbose = verbose

      def cutoff_rejects(self,data_array,data_columns):
            if self.verbose:
                  print "FLAGGING INPUT PEAKS BY PRE-DEFINED CUTOFFS"
            #takes data array with all data (raw, pca scaled, info) and rejects based on dictionary of cutoffs
            #returns boolean array 1=reject
            #mostly useful for training
            reject_mask = np.zeros(data_array.shape[0],dtype=np.bool)
            reject_cutoffs={'ccSf':(-999.9,999.9),'ccWf':(-999.9,999.9),'ccS2':(-999.9,999.9),'ccW2':(-999.9,999.9),
                            'ccSifi':(-999.9,999.9),'ccSifo':(-999.9,999.9),'ccSi2i':(-999.9,999.9),'ccSi2o':(-999.9,999.9),
                            'ccSifr':(-999.9,999.9),'ccSi2r':(-999.9,999.9),'ccWif':(-999.9,999.9),'ccWi2':(-999.9,999.9),
                            'ccSf60':(-999.9,999.9),'sdSf60':(-999.9,999.9),'ccS260':(-999.9,999.9),'sdS260':(-999.9,999.9),
                            'vf':(0.0,500.0),'v2':(0.0,500.0),'charge':(-60,60),
                            'RX0':(-10.0,10.0),'RX1':(-10.0,10.0),'RX2':(-10.0,10.0),'RX3':(-10.0,10.0),'RX4':(-10.0,10.0),
                            'RX5':(-10.0,10.0),'RX6':(-10.0,10.0),'RX7':(-10.0,10.0),'RX8':(-10.0,10.0),
                            'RX9':(-10.0,10.0),'RX10':(-10.0,10.0),'RX11':(-10.0,10.0),'RX12':(-10.0,10.0),'RX13':(-10.0,10.0),
                            'RX14':(-10.0,10.0),'RX15':(-10.0,10.0),'RX16':(-10.0,10.0),'RX17':(-10.0,10.0),'RX18':(-10.0,10.0),
                            'res':(0.5,5.01),'fofc_sigi':(0,999.9),'2fofc_sigi':(0,999.9),'fofc_sigo':(1.0,999.9),
                            '2fofc_sigo':(0.2,999.9),'dmove':(0.0,3.5)}
            for column in data_columns:
                  if column in reject_cutoffs:
                        lowc,highc = reject_cutoffs[column]
                        reject_sel = np.logical_or(np.array(data_array[column],dtype=np.float64) < lowc, 
                                                   np.array(data_array[column],dtype=np.float64) > highc)
                        reject_count = np.count_nonzero(reject_sel)
                        if reject_count > 0 and self.verbose:
                              print "     FLAGGED %6s PEAKS WITH %10s CUTOFF %4.2f %4.2f" % (reject_count,column,lowc,highc)
                        #combine_rejects
                        reject_mask = np.logical_or(reject_sel,reject_mask)
            total_rej = np.count_nonzero(reject_mask)
            if self.verbose:
                  print "TOTAL FLAGS %10s %4.3f" % (total_rej,float(total_rej)/data_array.shape[0])
            return reject_mask



      def feat_filt(self,data_array,sigscale=1.0):
            #2 sigma window for 28 features/calculations from [Sun Mar 11 21:16:48 2018', 'traset_res_scale.npy']
            #can be scaled with "sigscale"
            #returns filter_mask with counts of # of outliers for w,s,o,m.
            #6 counts usually selects 2-3% of each correct population
            npeaks = data_array.shape[0]
            npop = 4
            filter_mask = np.zeros((npeaks,npop),dtype=np.int8)
            w_cut={"ccSf":( 0.112, 0.724),
                   "ccWf":( 0.512, 0.958),
                   "ccS2":(-0.032, 0.524),
                   "ccW2":( 0.440, 0.893),
                   "ccSifi":(-0.048, 0.149),
                   "ccSifo":(-0.052, 0.149),
                   "ccSi2i":(-0.091, 0.178),
                   "ccSi2o":(-0.091, 0.171),
                   "ccSifr":( 0.058, 0.659),
                   "ccSi2r":(-0.054, 0.460),
                   "ccWif":(-0.023, 0.144),
                   "ccWi2":( 0.014, 0.192),
                   "ccSf60":( 0.044, 0.682),
                   "sdSf60":( 0.002, 0.051),
                   "ccS260":(-0.062, 0.472),
                   "sdS260":( 0.004, 0.059),
                   "score":(-15.932,-1.692),
                   "llgS":(-11.705,-0.581),
                   "llgW":(-2.168, 7.506),
                   "chiS":( 6.219,42.524),
                   "chiW":(-0.220,25.500),
                   "cscore":(-5.545,-1.112),
                   "cllgS":(-3.917,-0.528),
                   "cllgW":(-1.006, 3.217),
                   "cchiS":(-0.403,11.770),
                   "cchiW":(-1.864, 3.530),
                   "charge":(-24.584,11.531),
                   "c1":( 2.270, 3.109)}



            s_cut={"ccSf":( 0.613, 1.0),
                   "ccWf":( 0.663, 1.0),
                   "ccS2":( 0.560, 1.0),
                   "ccW2":( 0.650, 1.0),
                   "ccSifi":(-0.120, 0.366),
                   "ccSifo":(-0.132, 0.401),
                   "ccSi2i":(-0.014, 0.244),
                   "ccSi2o":(-0.030, 0.268),
                   "ccSifr":( 0.494, 0.959),
                   "ccSi2r":( 0.463, 0.923),
                   "ccWif":(-0.211, 0.116),
                   "ccWi2":(-0.162, 0.116),
                   "ccSf60":( 0.553, 0.982),
                   "sdSf60":(-0.001, 0.039),
                   "ccS260":( 0.512, 0.950),
                   "sdS260":(-0.001, 0.048),
                   "score":(-1.711,26.877),
                   "llgS":(-10.639,25.104),
                   "llgW":(-17.380, 6.678),
                   "chiS":(-8.610,36.390),
                   "chiW":(-5.252,112.992),
                   "cscore":( 0.130, 5.541),
                   "cllgS":(-1.340, 3.727),
                   "cllgW":(-3.638, 0.355),
                   "cchiS":(-5.175, 7.262),
                   "cchiW":(-0.819, 9.159),
                   "charge":(-12.533,23.682),
                   "c1":( 2.927, 4.122)}


            o_cut={"ccSf":( 0.315, 0.915),
                   "ccWf":( 0.466, 1.045),
                   "ccS2":( 0.219, 0.849),
                   "ccW2":( 0.436, 1.032),
                   "ccSifi":(-0.054, 0.178),
                   "ccSifo":(-0.066, 0.199),
                   "ccSi2i":(-0.041, 0.194),
                   "ccSi2o":(-0.048, 0.197),
                   "ccSifr":( 0.212, 0.848),
                   "ccSi2r":( 0.158, 0.775),
                   "ccWif":(-0.087, 0.136),
                   "ccWi2":(-0.046, 0.174),
                   "ccSf60":( 0.247, 0.883),
                   "sdSf60":(-0.001, 0.046),
                   "ccS260":( 0.190, 0.795),
                   "sdS260":( 0.001, 0.061),
                   "score":(-11.858,14.312),
                   "llgS":(-12.308, 8.626),
                   "llgW":(-12.696, 6.559),
                   "chiS":(-2.657,37.654),
                   "chiW":(-4.195,56.375),
                   "cscore":(-4.053, 4.857),
                   "cllgS":(-3.942, 3.262),
                   "cllgW":(-3.532, 2.048),
                   "cchiS":(-6.994,12.239),
                   "cchiW":(-2.603, 7.996),
                   "charge":(-20.909,18.515),
                   "c1":( 2.298, 4.204)}

            m_cut={"ccSf":( 0.399, 1.047),
                   "ccWf":( 0.537, 1.139),
                   "ccS2":( 0.101, 0.837),
                   "ccW2":( 0.345, 1.011),
                   "ccSifi":(-0.122, 0.285),
                   "ccSifo":(-0.131, 0.316),
                   "ccSi2i":(-0.072, 0.219),
                   "ccSi2o":(-0.080, 0.245),
                   "ccSifr":( 0.327, 0.982),
                   "ccSi2r":( 0.078, 0.752),
                   "ccWif":(-0.201, 0.148),
                   "ccWi2":(-0.104, 0.147),
                   "ccSf60":( 0.326, 1.040),
                   "sdSf60":(-0.006, 0.045),
                   "ccS260":( 0.057, 0.766),
                   "sdS260":(-0.001, 0.048),
                   "score":(-8.952,24.356),
                   "llgS":(-22.624,40.431),
                   "llgW":(-23.280,25.684),
                   "chiS":(-3.199,55.078),
                   "chiW":(-27.625,151.287),
                   "cscore":(-5.409,-1.456),
                   "cllgS":(-3.294, 3.721),
                   "cllgW":( 0.865, 6.427),
                   "cchiS":( 2.370,23.121),
                   "cchiW":(-2.537, 7.317),
                   "charge":(-31.919,14.130),
                   "c1":( 1.641, 2.733)}
            pops=("WAT","SO4","OTH","ML1")
            data_columns = list(set(w_cut.keys()) & set(data_array.dtype.names))
            #print "-->Filtering %s peaks on %s columns" % (data_array.shape[0],len(data_columns))
            for cind,column in enumerate(data_columns):
                  wlowc,whighc = w_cut[column]
                  slowc,shighc = s_cut[column]
                  olowc,ohighc = o_cut[column]
                  mlowc,mhighc = m_cut[column]
                  filter_mask[:,0]= filter_mask[:,0] + np.logical_or(data_array[column] < wlowc*sigscale,data_array[column] > whighc*sigscale)
                  filter_mask[:,1]= filter_mask[:,1] + np.logical_or(data_array[column] < slowc*sigscale,data_array[column] > shighc*sigscale)
                  filter_mask[:,2]= filter_mask[:,2] + np.logical_or(data_array[column] < olowc*sigscale,data_array[column] > ohighc*sigscale)
                  filter_mask[:,3]= filter_mask[:,3] + np.logical_or(data_array[column] < mlowc*sigscale,data_array[column] > mhighc*sigscale)
                  #for i in range(4):
                  #      reject_count = np.count_nonzero(filter_mask[:,i] > 5)
                  #      print "     CUMREJ %6s PEAKS AS OUTLIERS ON %10s FOR %s" % (reject_count,column,pops[i])

            return filter_mask
