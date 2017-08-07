from __future__ import division
import sys,re,copy
import numpy as np
#cctbx imports
import iotbx.pdb
from mmtbx import monomer_library
from libtbx import group_args
from cctbx import miller
from scitbx.array_family import flex
from iotbx import reflection_file_utils
from mmtbx.command_line import geometry_minimization
from mmtbx import monomer_library
import mmtbx.refinement.geometry_minimization
from cctbx import maptbx
import scitbx.rigid_body
import mmtbx.ncs
import mmtbx.refinement.real_space.individual_sites as rsr
import mmtbx.refinement.real_space.rigid_body as rigid
import iotbx.ncs
import mmtbx.maps.correlation
from iotbx.ccp4_map import write_ccp4_map
#PProbe imports
from PProbe_peak import PeakObj
from PProbe_struct import StructData
from PProbe_ref import RSRefinements
from PProbe_util import Util
from PProbe_cctbx import CctbxHelpers
from PProbe_realspace import RealSpace
from PProbe_forest import RfContacts

class FeatureExtraction:
      def __init__(self):
            pass

      def generate_peak_list(self,pdb_code,peak_pdb_hier,set_chain=False,renumber=False):
      #this function takes a list of peaks from a peak search as a pdb
      #and outputs a list of dictionaries with info and coordinates
      #if chainid is False, original chainids are preserved
            pput = Util()
            peak_list = []
            pdb = peak_pdb_hier
            for model in pdb.models():
                  for chain in model.chains():
                        if set_chain:
                              chainid=set_chain
                        else:
                              chainid = chain.id.strip()
                        for resgroups in chain.residue_groups():
                              for atomgroups in resgroups.atom_groups():
                                    for atom in atomgroups.atoms():
                                          if renumber:
                                                resid=str(len(peak_list) + 1)
                                          else:
                                                resid = resgroups.resseq.strip()
                                          coord = atom.xyz
                                          db_id = pput.gen_db_id(pdb_code,chainid,resid)
                                          peak_list.append(dict(db_id=db_id,resid=resid,chainid=chainid,coord=coord))
            return peak_list



      def generate_peak_object(self,peak,symmetry,orig_pdb_hier,strip_pdb_hier,peak_pdb_hier,struct_data,write_maps=False):
          db_id = peak['db_id']
          chainid = peak['chainid']
          coord = peak['coord']
          resid = peak['resid']
          pdb_code = db_id[0:4]
          bound = 5.0 #size of new box: +/- bound, must be an integer or half integer
          grid_last = int(bound*4+1)
          #create a peak object with all maps and pdb data
          peak_object = PeakObj(pdb_code,symmetry,orig_pdb_hier,strip_pdb_hier,peak_pdb_hier,struct_data,chainid,resid,coord,bound)
          ref_object=RSRefinements(peak_object)
          if write_maps:
                pptbx = CctbxHelpers()
                pptbx.write_local_map(peak_object.local_map_fofc,db_id+"_local_fofc",peak_object)
                pptbx.write_local_map(peak_object.local_map_2fofc,db_id+"_local_2fofc",peak_object)
                pptbx.write_local_map(peak_object.inv_map_fofc,db_id+"_inv_fofc",peak_object)
                pptbx.write_local_map(peak_object.inv_map_2fofc,db_id+"_inv_2fofc",peak_object)
                pptbx.write_local_map(peak_object.shaped_map_2fofc,db_id+"_shaped_2fofc",peak_object)
                pptbx.write_local_map(peak_object.shaped_map_fofc,db_id+"_shaped_fofc",peak_object)
          return peak_object,ref_object

                
      def basic_features(self,features,peak_object):
        #add data on rotations,peak heights, volumes, local environment, etc.
          pput = Util()
          features['resid'] = peak_object.resid
          features['chainid'] = peak_object.chainid
          features['vol_fofc'] = peak_object.peak_volume_fofc
          features['vol_2fofc'] = peak_object.peak_volume_2fofc
          features['orires'] = pput.original_residue(peak_object)
          features['charge'] = pput.charge(peak_object)
          features['fofc_sig_in'] = peak_object.peak_fofc
          features['2fofc_sig_in'] = peak_object.peak_2fofc
          features['fofc_sig_out'] = peak_object.density_at_point(peak_object.local_map_fofc,features['wat_fofc_ref_xrs'],features['wat_fofc_coord_out'])
          features['2fofc_sig_out'] = peak_object.density_at_point(peak_object.local_map_2fofc,features['wat_fofc_ref_xrs'],features['wat_fofc_coord_out'])
          new_coord = features['wat_fofc_coord_out']
          features['dmove'] = np.linalg.norm(np.subtract(new_coord,(5.0,5.0,5.0)))
          #collect all lists of contacts
          features['contacts'] = peak_object.contacts
          features['strip_contacts'] = peak_object.strip_contacts
          features['peak_contacts'] = peak_object.peak_contacts
          #s/w contacts added by rsr methods


      def analyze_features(self,features):
            #tally short contacts
            tal_str,tal_arr = self.tally_contacts(features)
            features['ctally'] = tal_arr
            features['cstr'] = tal_str #string array for later storage?
            t1 = tal_arr[0] + tal_arr[5] #all short in original structure
            t2 = tal_arr[2] + tal_arr[7] #all short to refined sulfate
            t3 = tal_arr[3] + tal_arr[8] #all short to refined water
            #get three shortest heavy, non w/s contact distances
            oricont_nohsw = self.no_hsw_contacts(features['contacts'])
            short_c = [6.01,6.01,6.01] #out of range
            max_ind = np.clip(len(oricont_nohsw),0,3)
            for cindex in np.arange(max_ind):
                  short_c[cindex] = oricont_nohsw[cindex]['distance']


            contact_feat = short_c + [tal_arr[0],tal_arr[5],tal_arr[2],tal_arr[7],tal_arr[3],tal_arr[8],t1,t2,t3]
            features['contact_feat'] = contact_feat


            #later functions expect arrays, so create 1 row array for single peak
            feat_col = ('c1','c2','c3','ol','om','sl','sm','wl','wm','t1','t2','t3')
            feat_fmt = (np.float32,np.float32,np.float32,np.int16,np.int16,np.int16,np.int16,np.int16,np.int16,np.int16,np.int16,np.int16)
            feat_arr = np.zeros(1,dtype=np.dtype(zip(feat_col,feat_fmt)))
            for faind,facol in enumerate(feat_col):
                  feat_arr[facol][0] = contact_feat[faind]


            solvent_content = np.clip(features['solc'],0.2,0.8)
            sig_scale = 0.5*np.sqrt(0.5/(1.0 - solvent_content))

            peak_flags = {'weak':False,'special':False,'remote':False,'sadc':False,
                          'badc':False,'close':False,'sfp':False}

            if features['2fofc_sig_out'] < sig_scale:
                  peak_flags['weak'] = True
            if np.nansum(tal_arr[0:11]) == 0: #no atomic contacts
                  peak_flags['remote'] = True
            if tal_arr[15] > 1:
                  peak_flags['special'] = True
            vbcut,mbcut = self.close_contacts(feat_arr)
            if vbcut:
                  peak_flags['badc'] = True
            if mbcut:
                  peak_flags['sadc'] = True
            if contact_feat[0] < 1.6:
                  peak_flags['close'] = True
                  
            features['fp_prob'] = RfContacts(np.array(contact_feat)).pclass1            

            features['peak_flags'] = peak_flags

      def no_hsw_contacts(self,cont_dict_list):
            if len(cont_dict_list) > 0:
                  new_list = []
                  for index,contact in enumerate(cont_dict_list): #list of dictionaries
                        keep = True
                        res = contact['resname'] #
                        atname  = contact['name']
                        at1 = atname[0] #really element
                        if at1 == 'H': #omit hydrogen
                              keep = False
                        if res == 'SO4' or res == 'PO4' or res == 'HOH':
                              keep = False
                        if keep:
                              new_list.append(contact)
                  return new_list
            else:
                  return cont_dict_list
      


      def tally_contacts(self,features):
            """
            Tally Contacts and to filter out bad peaks
            Peaks refined too close to protein atoms will have many contacts less than 1A to atoms in the strip hierarchy
            """
            dist_cutoffs = {"low":(0.00,1.1),"mid":(1.1,1.7),"high":(1.7,4.5)}
            contact_tally = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            special = np.count_nonzero(list(cont['special'] == True for cont in features['w_contacts']))
            contact_tally[-1] = special
            close_dist = [9.99,9.99,9.99,9.99,9.99]
            for index,cdict in enumerate(('contacts','strip_contacts','s_contacts','w_contacts','peak_contacts')):
                  if len(features[cdict]) > 0:
                        close_dist[index] = "%3.2f" % features[cdict][0]['distance']
                        for index2,cutoffs in enumerate(("low","mid","high")):
                              lcut = dist_cutoffs[cutoffs][0]
                              hcut = dist_cutoffs[cutoffs][1]
                              cut_contacts =np.count_nonzero(list((cont['distance'] >= lcut and cont['distance'] < hcut) for cont in features[cdict])) 
                              cut_contacts = np.clip(cut_contacts,0,9)
                              contact_tally[5*index2+index] = cut_contacts
            #return a 16char string that encodes this mess along with the numbers
            tally_string = "".join(('{:1s}'.format(str(x)) for x in contact_tally))
            assert len(tally_string) == 16
            return tally_string,contact_tally

      def close_contacts(self,contact_features):
            #contact features is 
            # ['c1','c2','c3','ol','om','sl','sm','wl','wm','calc1','calc2','calc3']
            #cutoffs for very bad contacts
            vbarr = np.zeros(contact_features.shape[0],dtype=np.bool_)
            mbarr = np.zeros(contact_features.shape[0],dtype=np.bool_)
            s1 = contact_features['wl'] > 3
            s2 = np.logical_and(contact_features['wl'] == 2,contact_features['wm'] > 2)
            s3 = np.logical_and(contact_features['wl'] == 3,contact_features['t2'] > 8)
            s4 = contact_features['wm'] > 4
            vbcut = np.logical_or(s1,np.logical_or(s2,np.logical_or(s3,s4)))
            vbarr[vbcut] = True
            mbarr[vbcut] = True
            #cutoffs for mild bad contacts
            s1 = contact_features['wl'] > 1
            s2 = np.logical_and(contact_features['wl'] == 1,contact_features['wm'] > 1)
            s3 = np.logical_and(contact_features['wl'] == 1,contact_features['t2'] > 8)
            s4 = contact_features['wm'] > 2
            mbcut = np.logical_or(s1,np.logical_or(s2,np.logical_or(s3,s4)))
            mbarr[mbcut] = True
            return vbarr,mbarr


