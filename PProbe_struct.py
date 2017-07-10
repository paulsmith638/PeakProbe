from __future__ import division
import sys,copy,os
import numpy as np
#CCTBX IMPORTS
import iotbx.pdb
from mmtbx import monomer_library
from cctbx import miller
from cctbx import crystal
from cctbx import xray
from scitbx.array_family import flex
import mmtbx.utils
from iotbx import reflection_file_utils
from mmtbx.command_line import geometry_minimization
from mmtbx import monomer_library
import mmtbx.refinement.geometry_minimization
from cctbx import maptbx
from iotbx.ccp4_map import write_ccp4_map
from cctbx import sgtbx
from cctbx import uctbx
import mmtbx.utils
from mmtbx.refinement import print_statistics
from iotbx import reflection_file_utils
from cctbx.array_family import flex
from cctbx import maptbx
#PProbe imports
from PProbe_util import Util as pputil

null_log = open(os.devnull,'w')


#our class gets passed the following:
#pdb root = pdb file we're working with
#map_coeffs = filename of MTZ weighted 2fofc and fofc map coeffs
#symmetry = cctbx symmetry object of original structure (sg, unit cell)
# 3 different pdb hierarchies:
#   1) original structure with all atoms/waters/etc. "orig_pdb"
#   2) structure stripped of water/sulfate/phospahte "strip_pdb"
#   3) pdb format of fofc peaks from #2 "peaks_pdb"


class StructData:
    def __init__(self,pdb_code,symmetry,orig_pdb_hier,strip_pdb_hier,peak_pdb_hier,map_file):
        #instantiate utility class
        self.pput = pputil()
        self.pdb_code = pdb_code
        self.orig_symmetry = symmetry
        self.orig_pdb_hier = orig_pdb_hier 
        self.orig_xrs = self.orig_pdb_hier.extract_xray_structure(crystal_symmetry=self.orig_symmetry)
        self.strip_pdb_hier = orig_pdb_hier 
        self.peak_pdb_hier = peak_pdb_hier 
        self.map_file = map_file
        self.fofcsig = self.make_fofc_map() #also makes map data
        self.twofofcsig = self.make_2fofc_map() #also makes map data
        self.so4_restraints_1sig = self.make_so4_restraints(1.0)
        self.so4_restraints_01sig = self.make_so4_restraints(0.1)
        self.wat_restraints_1sig = self.make_wat_restraints(1.0)
        self.wat_restraints_01sig = self.make_wat_restraints(0.1)
        self.solvent_content = self.get_solvent_content()
        self.dist_mat = self.make_dist_mask()
        self.round_mask = self.make_round_mask(self.dist_mat,2.0)#fixed radius of 2.0A
        self.shaped_mask = self.make_shaped_mask(self.dist_mat)

    def get_solvent_content(self):
        masks_obj = mmtbx.masks.bulk_solvent(self.orig_xrs,True,solvent_radius = 1.1,shrink_truncation_radius=1.0,grid_step=0.25)
        return masks_obj.contact_surface_fraction

    def make_fofc_map(self):
        # FOFC map of the whole original asu
        map_coeff = reflection_file_utils.extract_miller_array_from_file(
            file_name = self.map_file,
            label     = "FOFCWT,PHFOFCWT",
            type      = "complex",
            log       = null_log)
        map_sym = map_coeff.crystal_symmetry()
        fft_map = map_coeff.fft_map(resolution_factor=0.25)
        mapsig = np.nanstd(fft_map.real_map_unpadded().as_numpy_array())
        fft_map.apply_sigma_scaling()
        self.fofc_map_data = fft_map.real_map_unpadded()
        return mapsig

    def make_2fofc_map(self):
        # 2FOFC map of the whole original asu
        map_coeff = reflection_file_utils.extract_miller_array_from_file(
            file_name = self.map_file,
            label     = "2FOFCWT,PH2FOFCWT",
            type      = "complex",
            log       = null_log)
        map_sym= map_coeff.crystal_symmetry()
        fft_map = map_coeff.fft_map(resolution_factor=0.25)
        mapsig = np.nanstd(fft_map.real_map_unpadded().as_numpy_array())
        fft_map.apply_sigma_scaling()
        self.twofofc_map_data = fft_map.real_map_unpadded()
        return mapsig
    
    #create a generic SO4 restraints object, re-used for every refinement

    def make_so4_restraints(self,sigma):
        #hack to make one set of restraints for so4 and copy many times
        self.bound = 5.0 #this might change later
        self.resid = 1
        occ,b_fac = 1.0, 35.0 #dummy values
        #create a new sulfate PDB from raw txt with "S" at the center of a 10A cubic P1 cell
        pdb_string="CRYST1%9.3f%9.3f%9.3f  90.00  90.00  90.00 P 1            1\n" % \
            (2.0*self.bound,2.0*self.bound,2.0*self.bound)
        x1,y1,z1 = 5.0,5.0,5.0
        x2,y2,z2 = x1+0.190,y1+1.032,z1-1.013
        x3,y3,z3 = x1-1.373,y1+0.045,z1+0.487
        x4,y4,z4 = x1+0.251,y1-1.323,z1-0.569
        x5,y5,z5 = x1+0.915,y1+0.239,z1+1.118
        so4_dict = {"sx":x1,"sy":y1,"sz":z1,"o1x":x2,"o1y":y2,"o1z":z2,"o2x":x3,"o2y":y3,"o2z":z3,
                    "o3x":x4,"o3y":y4,"o3z":z4,"o4x":x5,"o4y":y5,"o4z":z5}
        pdb_entry = ""
        pdb_entry=pdb_entry+self.write_atom(1,"S","","SO4","X",self.resid,"",so4_dict['sx'],
                                            so4_dict['sy'],so4_dict['sz'],occ,b_fac,"S","")
        pdb_entry=pdb_entry+self.write_atom(2,"O1","","SO4","X",self.resid,"",so4_dict['o1x'],
                                            so4_dict['o1y'],so4_dict['o1z'],occ,b_fac,"O","")
        pdb_entry=pdb_entry+self.write_atom(3,"O2","","SO4","X",self.resid,"",so4_dict['o2x'],
                                            so4_dict['o2y'],so4_dict['o2z'],occ,b_fac,"O","")
        pdb_entry=pdb_entry+self.write_atom(4,"O3","","SO4","X",self.resid,"",so4_dict['o3x'],
                                            so4_dict['o3y'],so4_dict['o3z'],occ,b_fac,"O","")
        pdb_entry=pdb_entry+self.write_atom(5,"O4","","SO4","X",self.resid,"",so4_dict['o4x'],
                                            so4_dict['o4y'],so4_dict['o4z'],occ,b_fac,"O","")
        sulfate_pdb = pdb_string+pdb_entry
        #print sulfate_pdb
        self.so4_pdb = iotbx.pdb.input(source_info=None, lines=flex.split_lines(sulfate_pdb))
        self.so4_hier = self.so4_pdb.construct_hierarchy()
        self.so4_xrs = self.so4_pdb.xray_structure_simple()
        raw_records=self.so4_pdb.as_pdb_string()
        processed_pdb = monomer_library.pdb_interpretation.process(
            mon_lib_srv               = monomer_library.server.server(),
            ener_lib                  = monomer_library.server.ener_lib(),
            file_name                 = None,
            raw_records               = raw_records,
            crystal_symmetry          = self.so4_pdb.crystal_symmetry(),
            force_symmetry            = True)

        geometry = processed_pdb.geometry_restraints_manager(
            show_energies      = False,
            plain_pairs_radius = 5.0)

        restraints_manager = mmtbx.restraints.manager(geometry=geometry,normalization = False)
        #object that stores coordinates
        sites_start = self.so4_xrs.sites_cart()
        #proxies needed for restraints?
        chain_proxy=processed_pdb.all_chain_proxies
        restraints_selection = flex.bool(sites_start.size(), True)
        select_str="Element S"
        selection=chain_proxy.selection("Element S")
        #must select the "i" of an "i-j" pair?
        isel=selection.iselection()
        # new restraints system, should add a harmoinc restraint to S
        restraints_manager.geometry.add_reference_coordinate_restraints_in_place(
            #processed_pdb.all_chain_proxies,
            pdb_hierarchy=processed_pdb.all_chain_proxies.pdb_hierarchy,
            selection=isel,
            sigma=sigma)
        return restraints_manager

    def make_wat_restraints(self,sigma):
        #hack to make one set of restraints for wat 
        self.bound = 5.0 #this might change later
        self.resid = 1
        occ,b_fac = 1.0, 35.0
        pdb_string="CRYST1%9.3f%9.3f%9.3f  90.00  90.00  90.00 P 1            1\n" % \
            (2.0*self.bound,2.0*self.bound,2.0*self.bound)
        #x1,y1,z1 = coord1[0:3]
        x1,y1,z1=5.0,5.0,5.0
        pdb_entry = ""
        pdb_entry=pdb_entry+self.write_atom(1,"O","","HOH","X",self.resid,"",x1,y1,z1,occ,b_fac,"O","")
        wat_pdb=pdb_string+pdb_entry
        self.water_pdb = iotbx.pdb.input(source_info=None, lines=flex.split_lines(wat_pdb))
        self.water_hier = self.water_pdb.construct_hierarchy()
        self.water_xrs = self.water_pdb.xray_structure_simple()
        raw_records=self.water_pdb.as_pdb_string()
        processed_pdb = monomer_library.pdb_interpretation.process(
            mon_lib_srv               = monomer_library.server.server(),
            ener_lib                  = monomer_library.server.ener_lib(),
            file_name                 = None,
            raw_records               = raw_records,
            crystal_symmetry          = self.water_pdb.crystal_symmetry(),
            force_symmetry            = True)

        geometry = processed_pdb.geometry_restraints_manager(
            show_energies      = False,
            plain_pairs_radius = 5.0)

        restraints_manager = mmtbx.restraints.manager(geometry      = geometry,
                                                           normalization = False)
        #object that stores coordinates
        sites_start = self.water_xrs.sites_cart()
        #proxies needed for restraints?
        chain_proxy=processed_pdb.all_chain_proxies
        restraints_selection = flex.bool(sites_start.size(), True)
        select_str="Element O"
        selection=chain_proxy.selection("Element O")
        #must select the "i" of an "i-j" pair?
        isel=selection.iselection()
        # new restraints system, should add a harmoinc restraint to S
        restraints_manager.geometry.add_reference_coordinate_restraints_in_place(
            #processed_pdb.all_chain_proxies,
            pdb_hierarchy=processed_pdb.all_chain_proxies.pdb_hierarchy,
            selection=isel,
            sigma=sigma)
        return restraints_manager

    def write_atom(self,serial,name,alt,resname,chain,resid,ins,x,y,z,occ,temp,element,charge):
        return "ATOM  %5d %4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s\n" % \
          (serial,name,alt,resname,chain,resid,ins,x,y,z,occ,temp,element,charge)  
            

    """
    functions for map masks on the standard grid, compute once for every structure
    """
    def make_dist_mask(self):
        ref_map_grid = self.pput.new_grid((0.0,0.0,0.0),5.0)
        dist_grid = np.zeros(ref_map_grid.shape)
        return np.apply_along_axis(np.linalg.norm,1,ref_map_grid)

    def make_round_mask(self,dist_mat,radius):
        mask = np.less_equal(dist_mat,np.ones(dist_mat.shape)*float(radius))
        return np.array(mask,dtype=np.int)

    def make_shaped_mask(self,dist_mat):
        shape_func = lambda x: np.exp(-(0.5*(np.clip(x-0.1,0,np.inf)))**6.0)
        shaped_mask = np.apply_along_axis(shape_func,0,dist_mat)
        return shaped_mask

                         
