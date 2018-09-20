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
import mmtbx.maps.correlation
from iotbx.ccp4_map import write_ccp4_map
#PProbe imports
from PProbe_peak import PeakObj
from PProbe_struct import StructData
from PProbe_ref import RSRefinements
from PProbe_util import Util
from PProbe_cctbx import CctbxHelpers
from PProbe_realspace import RealSpace

class FeatureExtraction:
      def __init__(self):
            pass

      def generate_peak_list(self,pdb_code,peak_pdb_hier,model_id,set_chain=False,renumber=False):
      #this function takes a list of peaks from a peak search as a pdb
      #and outputs a list of dictionaries with info and coordinates
      #if chainid is False, original chainids are preserved
            pput = Util()
            peak_list = []
            pdb = peak_pdb_hier
            for model in pdb.models():
                  for chain in model.chains():
                        ori_chain = chain.id.strip()
                        if set_chain:
                              out_chain=set_chain
                        else:
                              out_chain = ori_chain
                        for resgroups in chain.residue_groups():
                              for atomgroups in resgroups.atom_groups():
                                    for atom in atomgroups.atoms():
                                          awl = atom.fetch_labels()
                                          resname=awl.resname.strip()
                                          name=awl.name.strip()
                                          altloc = awl.altloc.strip()
                                          ori_resid = resgroups.resseq.strip()
                                          if renumber:
                                                out_resid=str(len(peak_list) + 1)
                                          else:
                                                out_resid = ori_resid
                                          coord = atom.xyz
                                          resat = resname+"_"+ori_chain+ori_resid+"_"+name
                                          db_id = pput.gen_db_id(pdb_code,out_chain,out_resid)
                                          unat,unal,unrg = pput.gen_unids(awl,model=model_id)
                                          pdict = self.gen_pdict()
                                          pdict["db_id"] = db_id
                                          pdict["model"] = model_id
                                          pdict["resid"]=out_resid
                                          pdict["chainid"]=out_chain
                                          pdict["coord"]=coord
                                          pdict["unat"]=unat
                                          pdict["unal"]=unal
                                          pdict["unrg"]=unrg
                                          pdict["ori_chain"]=ori_chain
                                          pdict["ori_resid"]=ori_resid
                                          pdict["resat"]=resat
                                          peak_list.append(pdict)
            return peak_list



      def generate_peak_object(self,peak,symmetry,orig_pdb_hier,strip_pdb_hier,peak_pdb_hier,struct_data,write_maps=False):
          db_id = peak['db_id']
          chainid = peak['chainid']
          coord = peak['coord']
          resid = peak['resid']
          pdb_code = db_id[0:4]
          unal = peak['unal']
          bound = 5.0 #size of new box: +/- bound, must be an integer or half integer
          #create a peak object with all maps and pdb data
          peak_object = PeakObj(pdb_code,unal,symmetry,orig_pdb_hier,strip_pdb_hier,peak_pdb_hier,struct_data,chainid,resid,coord,bound)
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
            features['coord'] = peak_object.coord
            features['vol_fofc'] = peak_object.peak_volume_fofc
            features['vol_2fofc'] = peak_object.peak_volume_2fofc
            features['fofc_sig_in'] = peak_object.peak_fofc
            features['2fofc_sig_in'] = peak_object.peak_2fofc
            features['fofc_sig_out'] = peak_object.density_at_point(peak_object.local_map_fofc,features['wat_fofc_ref_xrs'],features['wat_fofc_coord_out'])
            features['2fofc_sig_out'] = peak_object.density_at_point(peak_object.local_map_2fofc,features['wat_fofc_ref_xrs'],features['wat_fofc_coord_out'])
            #rescale density level to account for lack of variance in solvent region
            features['2fofc_sigo_scaled'] = pput.scale_density(features['2fofc_sig_out'],features['solc'])
            features['fofc_sigo_scaled'] = pput.scale_density(features['fofc_sig_out'],features['solc'])
            new_coord = features['wat_fofc_coord_out']
            features['dmove'] = np.linalg.norm(np.subtract(new_coord,(5.0,5.0,5.0)))


      def gen_pdict(self):
            #initializes a dictionary with default values for a pdict
            #not all peaks use all values
            #many are copied to/from numpy
            #default values are assigned and reflect expected types
            pdict_proto = {"2fofc_sig_in":0.0 ,
                           "2fofc_sigo_scaled":0.0,
                           "2fofc_sig_out":0.0,
                           "ambig":0,
                           "anc_cont":[],
                           "anc_for":[],
                           "anchor":{},
                           "batch":0,
                           "bin":0,
                           "c1":0.0,
                           "cc":0,
                           "cchiS":0.0,
                           "cchiW":0.0,
                           "chainid":"",
                           "charge":0.0,
                           "chiS":0.0,
                           "chiW":0.0,
                           "clash":False,
                           "cllgS":0.0,
                           "cllgW":0.0,
                           "clust_cent":0.0,
                           "clust_mem":[],
                           "clust_pair":[],
                           "clust_rank":0,
                           "clust_score":0,
                           "contacts":[],
                           "cont_db":{},
                           "coord":(0.0,0.0,0.0),
                           "cscore":0.0,
                           "db_id":"",
                           "dmove":0.0,
                           "edc":0,
                           "fc":0,
                           "filter_mask":[0,0,0,0],
                           "fofc_sig_in":0.0,
                           "fofc_sigo_scaled":0.0,
                           "fofc_sig_out":0.0,
                           "label":0,
                           "llgS":0.0,
                           "llgW":0.0,
                           "master_dict":{},
                           "mf":0,
                           "mflag":0,
                           "mm_contacts":[],
                           "mod_cont":[],
                           "model":0,
                           "modexp_clust":[],
                           "mod_for":[],
                           "oh":0,
                           "ol":0,
                           "om":0,
                           "omit":0,
                           "omit_contacts":[],
                           "ori_chain":"",
                           "orires":"",
                           "ori_resid":"",
                           "pdb_code":"",
                           "peak_contacts":[],
                           "peak_unal_db":{},
                           "pick":0,
                           "pick_name":"",
                           "prob":0.0,
                           "prob_data":np.zeros((3,4)),
                           "ptype":"",
                           "rc":0,
                           "resat":"",
                           "resid":"",
                           "resid":0,
                           "resolution":"",
                           "resolution":0.0,
                           "s_contacts":[],
                           "score":0.0,
                           "scr1":0.0,
                           "scr2":0.0,
                           "scr3":0.0,
                           "sl":0,
                           "sm":0,
                           "so4_2fofc_mean_cc60":0.0,
                           "so4_2fofc_ref_oricoords":list(("X",(0.0,0.0,0.0)) for i in range(5)),
                           "so4_2fofc_stdev_cc60":0.0,
                           "so4_cc_2fofc_in":0.0,
                           "so4_cc_2fofc_inv_in":0.0,
                           "so4_cc_2fofc_inv_out":0.0,
                           "so4_cc_2fofc_inv_rev":0.0,
                           "so4_cc_2fofc_out":0.0,
                           "so4_cc_fofc_in":0.0,
                           "so4_cc_fofc_inv_in":0.0,
                           "so4_cc_fofc_inv_out":0.0,
                           "so4_cc_fofc_inv_rev":0.0,
                           "so4_cc_fofc_out":0.0,
                           "so4_fofc_coord_out":(),
                           "so4_fofc_mean_cc60":0.0,
                           "so4_fofc_stdev_cc60":0.0,
                           "solc":0.0,
                           "sol_contacts":[],
                           "sol_mod":[],
                           "sp":0,
                           "st":0,
                           "status":-1,
                           "strip_contacts":[],
                           "tflag":0,
                           "unal":0,
                           "unat":0,
                           "unrg":0,
                           "vol_2fofc":0.0,
                           "vol_fofc":0.0,
                           "warnings":[],
                           "wat_2fofc_ref_oricoords":(),
                           "wat_cc_2fofc_in":0.0,
                           "wat_cc_2fofc_inv":0.0,
                           "wat_cc_2fofc_out":0.0,
                           "wat_cc_fofc_in":0.0,
                           "wat_cc_fofc_inv":0.0,
                           "wat_cc_fofc_out":0.0,
                           "wat_fofc_coord_out":(),
                           "w_contacts":[],
                           "wl":0,
                           "wm":0,
                           "worst_mm":{},
                           "wt":0}
            
            return pdict_proto
