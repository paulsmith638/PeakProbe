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
                                          peak_list.append(dict(db_id=db_id,
                                                                model=model_id,
                                                                resid=out_resid,
                                                                chainid=out_chain,
                                                                coord=coord,
                                                                unat=unat,
                                                                unal=unal,
                                                                unrg=unrg,
                                                                ori_chain=ori_chain,
                                                                ori_resid=ori_resid,
                                                                resat=resat))
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


