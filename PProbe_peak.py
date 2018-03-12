from __future__ import division
#GENERIC IMPORTS
import sys,os,re,copy
import numpy as np
#CCTBX IMPORTS
import iotbx.pdb
from mmtbx import monomer_library
from libtbx import group_args
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
import mmtbx.utils
from mmtbx.refinement import print_statistics
from iotbx import reflection_file_utils
from cctbx.array_family import flex
from cctbx import maptbx

#PProbe IMPORTS
from PProbe_util import Util
from PProbe_cctbx import CctbxHelpers

class PeakObj:
    """
    A class for a "peak" object with all associated data for a particular peak
    This class gets instantiated for every peak, and can be a bit demanding for cpu and memory
    Constructor takes the following:
         StructData object that contains pdb_code, all pdbs/xrs/maps
         Peak Specific info: chainid,coord,bound, option to specify stripped pdb
    Initialization does the following:
         1)makes local maps on the standard 0.5A grid around the peak
         2)makes versions of these maps (shaped, round, etc.)

    """
    def __init__(self,pdb_code,unid,symmetry,orig_pdb_hier,strip_pdb_hier,peak_pdb_hier,struct_data,chainid,resid,coord,bound):
        #instantiate utility classes
        self.pput = Util()
        self.ppctx = CctbxHelpers()
        #bind some util functions here
        self.write_atom = self.pput.write_atom

        #attach references to structure data for use in this class
        self.pdb_code = struct_data.pdb_code
        self.orig_symmetry = struct_data.orig_symmetry
        self.orig_pdb_hier = struct_data.orig_pdb_hier
        self.orig_xrs = struct_data.orig_xrs
        self.strip_pdb_hier = strip_pdb_hier
        self.peak_pdb_hier = struct_data.peak_pdb_hier 
        self.struct_data = struct_data

        self.chainid = chainid #single letter string
        self.resid = int(resid)
        self.coord = coord #tuple of floats
        self.bound = bound
        self.grid_last = int(self.bound*4+1)
        self.unid = unid

        #copy pdb,hier,xrs in standard settings
        self.so4_pdb = copy.deepcopy(self.struct_data.std_so4_pdb)
        self.symmetry = self.so4_pdb.crystal_symmetry()
        self.so4_hier = copy.deepcopy(self.struct_data.std_so4_hier)
        self.so4_xrs = copy.deepcopy(self.struct_data.std_so4_xrs)
        self.wat_pdb = copy.deepcopy(self.struct_data.std_wat_pdb)
        self.wat_hier = copy.deepcopy(self.struct_data.std_wat_hier)
        self.wat_xrs = copy.deepcopy(self.struct_data.std_wat_xrs)



        #make local maps
        self.local_map_fofc, self.peak_volume_fofc = self.make_local_map(self.struct_data.fofc_map_data)
        self.local_map_2fofc, self.peak_volume_2fofc = self.make_local_map(self.struct_data.twofofc_map_data)
        self.shaped_map_fofc = self.make_shaped_map(self.local_map_fofc)
        self.shaped_map_2fofc = self.make_shaped_map(self.local_map_2fofc)
        self.inv_map_fofc = self.make_round_map(self.local_map_fofc,2.0,True)
        self.inv_map_2fofc = self.make_round_map(self.local_map_2fofc,2.0,True)

        #set peak heights of initial peak
        self.peak_fofc = self.density_at_point(self.struct_data.fofc_map_data,self.orig_xrs,self.coord)
        self.peak_2fofc = self.density_at_point(self.struct_data.twofofc_map_data,self.orig_xrs,self.coord)

    def density_at_point(self,map_data,xrs,point):
        site_frac = xrs.unit_cell().fractionalize(site_cart=point)
        return map_data.tricubic_interpolation(site_frac)

    def make_local_map(self,map_data,volume_radius=2.0):
        # first make a real map of the whole original asu
        #self.new_map_values = flex.double()
        #populate a list of cartesian sites in original coordinates
        center = self.coord
        volume = 0.0
        solvent_content = self.struct_data.solvent_content
        #sanity check, avoid overcorrection?
        solvent_content = np.clip(solvent_content,0.2,0.8)

        #function to correct for solvent content on sigma values
        # Thanks Tom T!  Assuming near zero variance in solvent region,
        # setting total map variance to 1.0 implies the following:
        # 1.0 = (RMSp^2*vP + RMSs^2*vS)**0.5 = (RMSp^2*vP + 0.0^2*vS)**0.5
        # s.t. 1.0 = (RMSp^2*vP)**0.5 = (RMSp^2*(1-vS))**0.5, or
        # RMSp = 1.0/sqrt(1-Vs), i.e. the true RMS in the protein region
        # is scaled up to account for the lack of variance in the solvent region

        volume_sig_scale = 1.0/np.sqrt(1.0 - solvent_content)
        new_grid_points=self.pput.new_grid(self.coord,self.bound)
        new_map_values = flex.double()
        for point in new_grid_points:
            site_frac = self.orig_xrs.unit_cell().fractionalize(site_cart=point)
            value_at_point = map_data.tricubic_interpolation(site_frac)
            new_map_values.append(value_at_point)
            if (self.calcbond_lengths(center,point) <= volume_radius):
                #count number of gridpoints above scaled_sigma as a "peak volume"
                if (value_at_point >= volume_sig_scale):
                    volume += 1.0
        #map values are a 1d array, now reshape to our new cell on 0.5A grid
        new_map_values.reshape(flex.grid(self.grid_last,self.grid_last,self.grid_last))
        return new_map_values.as_double(),volume

    def find_peaks(self,input_map):
        map_array = input_map.as_numpy_array()
        max_index= np.array(np.where(map_array == np.amax(map_array)))
        max_coords = 0.5*max_index
        dist = np.sqrt((max_coords[0] - 5.0)**2 + (max_coords[1]-5.0)**2 + (max_coords[2] - 5.0)**2)
        return np.amin(dist)
        

    def make_shaped_map(self,square_map):
        #modifies map by a steep falloff function around map center to 
        #kill strong density that may cause so4/water to be pulled away
        #from it's starting location during refinement
        #typically a problem if next to a heavy metal site
        #convert to flex
        map_mask_double = flex.double(self.struct_data.shaped_mask.astype(np.float64))
        map_mask = map_mask_double.as_double()
        map_copy = copy.deepcopy(square_map)
        #scalar multiplication of orginal map and mask
        shaped_map =  map_copy*map_mask
        return shaped_map.as_double()

    def make_round_map(self,square_map,radius=5.0,invert=False):
        map_mask_double = flex.double(self.struct_data.round_mask.astype(np.float64))
        map_mask = map_mask_double.as_double()
        map_copy = copy.deepcopy(square_map)
        #scalar multiplication of orginal map and mask
        round_map =  map_copy*map_mask
        if invert:
            #values closte to zero give very huge and problematic results
            #so here is a function that behaves like an inverse
            #for values above 0.1, but then plateaus off
            inv_map_points=flex.double(square_map.size())
            dampfunc = lambda x: (1.0/((1.0+np.exp(-100*(x-0.1)))))
            for index,value in enumerate(square_map):
                if abs(value) > 0.00001:
                    dval = dampfunc(abs(value))
                    if value > 0.0:
                        inv_map_points[index]=dval/value + 20.0*(1.0-dval)
                    else:
                        inv_map_points[index]=-(dval/abs(value) + 20.0*(1.0-dval))
                else:
                    inv_map_points[index]=25.0
            inv_map_points.reshape(flex.grid(self.grid_last,self.grid_last,self.grid_last))   
            return inv_map_points*map_mask
        else:
            return round_map.as_double()

   

    def calcbond_lengths(self,coord1,coord2):
        return np.linalg.norm(coord2-coord1)


            

