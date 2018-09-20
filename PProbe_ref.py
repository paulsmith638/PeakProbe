from __future__ import division
import sys,math,re,random,copy
from mmtbx import monomer_library
import mmtbx.restraints
import mmtbx.utils
from cctbx import maptbx
import scitbx.rigid_body
import mmtbx.refinement.real_space.individual_sites as rsr
import mmtbx.refinement.real_space.rigid_body as rigid
import mmtbx.maps.correlation
from cctbx import sgtbx



class RSRefinements:
    def __init__(self,peak_object):

        self.peak_obj = peak_object
        self.pdb_code = self.peak_obj.pdb_code
        self.chainid = self.peak_obj.chainid
        self.resid = self.peak_obj.resid
        self.fofc_map_data = self.peak_obj.local_map_fofc
        self.twofofc_map_data = self.peak_obj.local_map_2fofc
        self.fofc_inv_map_data = self.peak_obj.inv_map_fofc
        self.twofofc_inv_map_data = self.peak_obj.inv_map_2fofc

        #self.twofofc_map_data = self.peak_obj.round_map_2fofc
        #self.map_data = self.peak_obj.local_map
        #self.map_data = self.peak_obj.make_round_map(5.0).round_map
        self.grid_last = self.peak_obj.grid_last
        self.symmetry = self.peak_obj.symmetry

        
    def calculate_cc(self,pdb_hier,xrs,sel_str,map_data):
        selection = pdb_hier.atom_selection_cache().selection(sel_str)
        # dummy D_min set to current setup
        cc_obj = mmtbx.maps.correlation.from_map_and_xray_structure_or_fmodel(xray_structure=xrs,map_data=map_data,d_min=2.0)
        cc = cc_obj.cc(selection=selection) 
        return cc

    def refine_rbr(self,pdb_hier,pdb_xrs,map_data):
        #Rigid Body Refinement = no restraints manager needed
        try:
            ref_out = rigid.refine_mz(map_data,pdb_hier,pdb_xrs,d_min=2.0,log="/dev/null")
  	    ref_xrs = ref_out.xray_structure
            ref_hier = ref_out.pdb_hierarchy
        except:
            print "DATA RBR of chainid,resid "+self.chainid+self.resid+" FAILED!"
            
        return ref_xrs,ref_hier

    def refine_rsr(self,pdb_obj,pdb_hier,pdb_xrs,map_data,restraints):
        #restraints manager required, generated as part of structure data
        #res_sigma is target for coordinate restraints 
        #refinement updates coordinates, so make a copy first
        input_xrs = copy.deepcopy(pdb_xrs)
        input_hier = copy.deepcopy(pdb_hier)
        try:
            #weight of 5.0 allows good movement into density 
            #speeds up by skipping weight calculation
            ref_out = rsr.easy(map_data,input_xrs,input_hier,restraints,w=5.0,max_iterations=100)
            ref_xrs = ref_out.xray_structure
            ref_hier = ref_out.pdb_hierarchy
        except:
            print "RSR of chainid,resid "+self.chainid+str(self.resid)+" FAILED!"
            #return unrefined structure (a bit dangerous)
            ref_xrs = input_xrs
            ref_hier = input_hier
        return ref_xrs,ref_hier


    def write_pdb_file(self,pdb_hier,filename):
        pdb_hier.write_pdb_file(
            file_name = filename,
            append_end=True)
