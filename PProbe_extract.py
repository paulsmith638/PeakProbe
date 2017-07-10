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
from PProbe_realspace import RealSpace


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
            ref_object.write_map(peak_object.local_map_fofc,"local_fofc")
            ref_object.write_map(peak_object.local_map_2fofc,"local_2fofc")
            ref_object.write_map(peak_object.inv_map_fofc,"inv_fofc")
            ref_object.write_map(peak_object.inv_map_2fofc,"inv_2fofc")
            ref_object.write_map(peak_object.shaped_map_2fofc,"shaped_2fofc")
            ref_object.write_map(peak_object.shaped_map_fofc,"shaped_fofc")
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
          features['score'] = [-1.0] #scoring later
