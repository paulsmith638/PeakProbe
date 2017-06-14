from __future__ import division
import sys,math,re,random,copy
import numpy as np
import iotbx.pdb
from mmtbx import monomer_library
from libtbx import group_args
from cctbx import miller
from scitbx.array_family import flex
from libtbx.utils import user_plus_sys_time
import mmtbx.monomer_library
import mmtbx.monomer_library.pdb_interpretation
import mmtbx.restraints
import mmtbx.utils
import mmtbx.secondary_structure
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
from scitbx.array_family import flex
from iotbx.ccp4_map import write_ccp4_map
from cctbx import sgtbx


class mini_ref:
    def __init__(self,PeakObj):
        self.peak_obj = PeakObj
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

    def write_map(self,input_map_data,outstr):
        write_ccp4_map(
            file_name=self.pdb_code+"_"+self.chainid+"_"+str(self.resid)+"_"+outstr+".ccp4",
            unit_cell=self.peak_obj.so4_xrs.unit_cell(),
            space_group=sgtbx.space_group_info("P1").group(),
            gridding_first=(0,0,0),
            gridding_last=(self.grid_last,self.grid_last,self.grid_last),
            map_data=input_map_data,
            labels=flex.std_string(["local_map_from_"+self.chainid+"_"+str(self.resid)])
            )
    def write_pdb_file(self,pdb_hier,filename):
        pdb_hier.write_pdb_file(
            file_name = filename,
            append_end=True)

        
    def calculate_cc(self,pdb_hier,xrs,sel_str,map_data):
        selection = pdb_hier.atom_selection_cache().selection(sel_str)
        # dummy D_min set to current setup
        cc_obj = mmtbx.maps.correlation.from_map_and_xray_structure_or_fmodel(xray_structure=xrs,map_data=map_data,d_min=2.0)
        cc = cc_obj.cc(selection=selection) 
        return cc

    def generate_geo_res(self,pdb_obj,pdb_xrs,res_sel,res_sigma):
        raw_records=pdb_obj.as_pdb_string()
        #this needs work, where are sigmas for normal restraints?
        #res_sigma is for coordinate harmonic res only?
        # are sigmas only useful for normalized restraints?
        processed_pdb = monomer_library.pdb_interpretation.process(
            mon_lib_srv               = monomer_library.server.server(),
            ener_lib                  = monomer_library.server.ener_lib(),
            file_name                 = None,
            raw_records               = raw_records,
            crystal_symmetry          = self.symmetry,
            force_symmetry            = True)

        geometry = processed_pdb.geometry_restraints_manager(
            show_energies      = False,
            plain_pairs_radius = 5.0)

        self.restraints_manager = mmtbx.restraints.manager(geometry      = geometry,
                                                           normalization = False)
        #object that stores coordinates
        sites_start = pdb_xrs.sites_cart()
        #proxies needed for restraints?
        chain_proxy=processed_pdb.all_chain_proxies
        restraints_selection = flex.bool(sites_start.size(), True)
        select_str=res_sel
        selection=chain_proxy.selection(select_str)
        #must select the "i" of an "i-j" pair?
        isel=selection.iselection()
        self.restraints_manager.geometry.generic_restraints_manager.reference_manager.add_coordinate_restraints(
            sites_cart=sites_start.select(isel),
            selection=isel,
            sigma=res_sigma)
        #print dir(self.restraints_manager.geometry)

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
            ref_out = rsr.easy(map_data,input_xrs,input_hier,restraints,w=None,max_iterations=100)
            ref_xrs = ref_out.xray_structure
            ref_hier = ref_out.pdb_hierarchy
        except:
            print "RSR of chainid,resid "+self.chainid+str(self.resid)+" FAILED!"
            #return unrefined structure (a bit dangerous)
            ref_xrs = input_xrs
            ref_hier = input_hier
        return ref_xrs,ref_hier


