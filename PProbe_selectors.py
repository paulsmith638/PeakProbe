from __future__ import division
import sys,math,ast,copy
import numpy as np
class Selectors:
     """
     class of functions for selecting data by
     population (obss = observed sulfate or phostphate, obsw = observed water)
     by resolution cutoffs
     by omit, batch, etc
     raw data must have a data column defined as "ori" and another as "omit"
     """
     def __init__(self,raw_data):
          #selects all that is not XXX
          self.sw_bool = np.apply_along_axis(lambda x: x  != 'XXX',0,np.core.defchararray.strip(raw_data['ori']))
          #selects sulfate, then phosphate, then combines
          self.obss_bool = np.apply_along_axis(lambda x: x == 'SO4',0,np.core.defchararray.strip(raw_data['ori']))
          self.obsp_bool = np.apply_along_axis(lambda x: x == 'PO4',0,np.core.defchararray.strip(raw_data['ori']))
          self.obss_bool = self.union_mask((self.obss_bool,self.obsp_bool))
          #selects water
          self.obsw_bool = np.apply_along_axis(lambda x: x == 'HOH'  ,0,np.core.defchararray.strip(raw_data['ori']))
          #selects data flagged for omit, bad structures, etc.
          if raw_data['omit'].dtype == '|S32':
               self.omit_bool = np.array(raw_data['omit'] == 'True').astype(np.bool)
          if raw_data['omit'].dtype == 'bool':
               self.omit_bool = raw_data['omit']

          #group data into populations
          self.included_data_bool = self.intersect_mask((self.sw_bool,np.invert(self.omit_bool)))
          self.inc_obss_bool = self.intersect_mask((self.included_data_bool,self.obss_bool))
          self.inc_obsw_bool = self.intersect_mask((self.included_data_bool,self.obsw_bool))
          #std dataset has 19 features of floating point data (6 others are info, resolution, etc.)
          self.std_view_col = ['ccSf','ccWf','ccS2','ccW2','ccSifi','ccSifo','ccSi2i','ccSi2o','ccSifr','ccSi2r',
                               'ccWif','ccWi2','ccSf60','sdSf60','ccS260','sdS260','vf','v2','charge']
          self.std_view_formats = (np.float64,np.float64,np.float64,np.float64,np.float64,
                                   np.float64,np.float64,np.float64,np.float64,np.float64,
                                   np.float64,np.float64,np.float64,np.float64,np.float64,
                                   np.float64,np.float64,np.float64,np.float64)
          #some numerical routines don't like structured arrays, give raw dtype
          self.raw_dtype = np.dtype(str(len(self.std_view_formats))+"f8")
          self.std_view_dtype = np.dtype(zip(self.std_view_col,self.std_view_formats))
          self.pca_view_col = ['RX0','RX1','RX2','RX3','RX4','RX5',
                               'RX6','RX7','RX8','RX9','RX10','RX11',
                               'RX12','RX13','RX14','RX15','RX16','RX17','RX18']
          self.pca_view_dtype = np.dtype(zip(self.pca_view_col,self.std_view_formats))
          self.alldata_pca_col = ['RX0','RX1','RX2','RX3','RX4','RX5',
                                  'RX6','RX7','RX8','RX9','RX10','RX11',
                                  'RX12','RX13','RX14','RX15','RX16','RX17','RX18',
                                  'ori','res','id','bin','batch','omit',
                                  'solc','fofc_sigi','2fofc_sigi','fofc_sigo','2fofc_sigo','dmove','score']
          self.alldata_pca_formats = [np.float64,np.float64,np.float64,np.float64,np.float64,
                                      np.float64,np.float64,np.float64,np.float64,np.float64,
                                      np.float64,np.float64,np.float64,np.float64,np.float64,
                                      np.float64,np.float64,np.float64,np.float64,
                                      'S16',np.float64,'S16',np.int16,np.int16,'S32',
                                      np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64]
          self.alldata_pca_dtype = np.dtype(zip(self.alldata_pca_col,self.alldata_pca_formats))




     #combine boolean selection masks by logical AND
     def intersect_mask(self,mask_list):
          comb_mask = np.ones(mask_list[0].shape,dtype=np.bool)
          for mask in mask_list:
               comb_mask = np.logical_and(comb_mask,mask)
          return comb_mask

     #combine boolean selection masks by logical OR
     def union_mask(self,mask_list):
          comb_mask = np.zeros(mask_list[0].shape,dtype=np.bool)
          for mask in mask_list:
               comb_mask = np.logical_or(comb_mask,mask)
          return comb_mask


     def reso_select_mask(self,raw_data,lowcut,highcut):
          lressel_bool = np.apply_along_axis(lambda x: x > lowcut,0,raw_data['res'])
          hressel_bool = np.apply_along_axis(lambda x: x < highcut,0,raw_data['res'])
          return self.intersect_mask((lressel_bool,hressel_bool))
