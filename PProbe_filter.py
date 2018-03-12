from __future__ import division
#generic imports
import sys,math,ast,copy
import numpy as np
from PProbe_util import Util
from PProbe_contacts import Contacts
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

      

      def peak_fc(self,data_array):
            #function to assign a class for various peak flags
            # 0 = no flags
            # 1 = special position
            # 2 = very bad contacts
            # 3 = bad contacts and one close contact
            # 4 = bad contacts
            # 5 = one close contact
            # 6 = weak by 2fofc
            # 7 = remote, far from any contact

            flags_col=['weak','remote','close','special','badc','sadc']
            flags_fmt = [np.bool_,np.bool_,np.bool_,np.bool_,np.bool_,np.bool_]
            flags_dtype = np.dtype(zip(flags_col,flags_fmt))
            flags_arr = np.zeros(data_array.shape[0],dtype=flags_dtype)

            #flag weak peaks with low 2fofc level (scaled)
            solvent_content = np.clip(data_array['solc'],0.2,0.8)
            sig_scale = 0.5*np.sqrt(0.5/(1.0 - solvent_content))
            flags_arr['weak'] = data_array['2fofc_sigo'] < sig_scale

            csum = np.zeros(data_array.shape[0],dtype=np.int16)
            for ccol in ('ol','om','oh','sl','sm','sh'):
                  csum = csum+data_array[ccol]
            flags_arr['remote'] = csum == 0 #no contacts to anything
            flags_arr['special'] = data_array['sp'] > 1 #likely special position

            #screen for bad contacts
            vbarr = np.zeros(data_array.shape[0],dtype=np.bool_) #very bad
            mbarr = np.zeros(data_array.shape[0],dtype=np.bool_) #moderate bad
            s1 = data_array['wl'] >= 2
            s2 = np.logical_and(data_array['wl'] == 1,data_array['wm'] > 1)
            s3 = np.logical_and(data_array['wl'] == 1,data_array['st'] > 8)
            s4 = data_array['wm'] >= 3
            vbcut = np.logical_or(s1,np.logical_or(s2,np.logical_or(s3,s4)))
            vbarr[vbcut] = True
            mbarr[vbcut] = True

            s1 = data_array['wl'] >= 1
            s2 = np.logical_and(data_array['wl'] == 1,data_array['wm'] >= 1)
            s3 = data_array['st'] > 4
            s4 = data_array['wt'] > 2
            mbcut = np.logical_or(s1,np.logical_or(s2,np.logical_or(s3,s4)))
            mbarr[mbcut] = True
          
            flags_arr['badc'] = vbcut
            flags_arr['sadc'] = mbcut
            flags_arr['close'] = data_array['c1'] < 2.2

            #assign flags
            flag_class = np.zeros(flags_arr.shape[0],dtype=np.int16)
            sp_sel = flags_arr['special'] == True
            badc_sel = flags_arr['badc'] == True
            sadc_sel = flags_arr['sadc'] == True
            close_sel = flags_arr['close'] == True
            remote_sel = flags_arr['remote'] == True
            weak_sel = flags_arr['weak'] == True

            #assign in rev order of precidence (remote < special, special implies badc, etc.)
            flag_class[remote_sel] = 7
            flag_class[weak_sel] = 6
            flag_class[close_sel] = 5
            flag_class[sadc_sel] = 4
            flag_class[np.logical_and(sadc_sel,close_sel)] = 3
            flag_class[badc_sel] = 2
            flag_class[sp_sel] = 1
            
            return flag_class

      def peak_sat(self,data_array,selector=None):
            #input is array of all data (or one entire pdb), not selected
            #as all potential contacts must be included
            #selector is boolean mask for which flags will be output
            #likely options for selector are all predicted sulfate
            #flags
            #1 = solo peak
            #2 = part of a large cluster
            #3 = principal peak of local cluster, also master of large cluster
            #4 = principal peak of local cluster
            #5 = satellite peak of local cluster (should be rejected or re-examined)
            #6 = self satellite peak, close to special position
            ppcont = Contacts()
            all_id_hash={}
            for index,peak in enumerate(data_array):
                  all_id_hash[peak['id']] = index
            cd,rcd = ppcont.cluster_assembly(data_array)
            selall = np.ones(data_array.shape[0],dtype=np.bool_)
            #get satellites for all peaks
            sat_dict = self.screen_satellite(data_array,selall,all_id_hash,cd,rcd)
            if selector is None:
                  selector = selall
            #mark according to selector (even if not in selector)
            for peak in data_array[selector]:
                  satlist = sat_dict[peak['id']]
                  if len(satlist) == 0:#solo peak
                        data_array['mf'][all_id_hash[peak['id']]] = 1
                        continue
                  if len(satlist) > 5:#large cluster
                        data_array['mf'][all_id_hash[peak['id']]] = 2
                        for satpeak in satlist[1::]:
                              spind = all_id_hash[satpeak[0]]
                              data_array['mf'][spind] = 2  
                        continue
                  if peak['id'] == satlist[0][0]: #best ed score in cluster
                        clust_master = cd[peak['id']]
                        if peak['id'] == clust_master:
                              #cluster master peak (not score related)
                              data_array['mf'][all_id_hash[peak['id']]] = 3
                        else:
                              #just best score in cluster, not master
                              data_array['mf'][all_id_hash[peak['id']]] = 4
                        for satpeak in satlist[1::]:
                              #mark all others as satellites, regardless of score
                              #print "PEAK %s is satellite of %s" % (satpeak[0],peak['id'])
                              spind = all_id_hash[satpeak[0]]
                              if satpeak[0] == peak['id']:
                                    data_array['mf'][spind] = 6
                              else:
                                    data_array['mf'][spind] = 5
            return sat_dict

      def screen_satellite(self,data_array,selector,all_id_hash,master_clust_dict,master_rev_clust_dict):
            #takes selected data and collects local clusters by sorted distance matrix
            if self.verbose:
                  print "Checking for Satellite Peaks:"
            ppcont = Contacts()
            sat_dict = {}
            for peak in data_array[selector]:
                  cpeak_clust = ppcont.check_peak_cluster(peak['id'],data_array,all_id_hash,master_clust_dict,master_rev_clust_dict)
                  if 'none_X_00000' in cpeak_clust:
                        sat_dict[peak['id']] = []
                  if len(cpeak_clust) > 1:
                        s_cpeaks=[]
                        for cpeakid in cpeak_clust:
                              cpeak = data_array[all_id_hash[cpeakid]]
                              cscore = cpeak['score']
                              s_cpeaks.append([cpeakid,cpeak['score']])
                        s_cpeaks.sort(key=lambda x: x[1],reverse=True)
                        sat_dict[peak['id']] = s_cpeaks
            return sat_dict


            
