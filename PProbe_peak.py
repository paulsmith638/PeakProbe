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
    def __init__(self,pdb_code,symmetry,orig_pdb_hier,strip_pdb_hier,peak_pdb_hier,struct_data,chainid,resid,coord,bound,strip_pdb=None):
        #instantiate utility class
        self.pput = Util()
        #attach references to structure data for use in this class
        self.pdb_code = struct_data.pdb_code
        self.orig_symmetry = struct_data.orig_symmetry
        self.orig_pdb_hier = struct_data.orig_pdb_hier
        self.orig_xrs = struct_data.orig_xrs
        #if pdb was stripped, then analyzed, keep versions straight
        if not strip_pdb:
            self.strip_pdb_hier = orig_pdb_hier 
        else:
            self.strip_pdb_hier = strip_pdb
        self.peak_pdb_hier = struct_data.peak_pdb_hier 
        self.struct_data = struct_data

        self.chainid = chainid #single letter string
        self.resid = int(resid)
        self.coord = coord #tuple of floats
        self.bound = bound
        self.grid_last = int(self.bound*4+1)
        
        #make local maps
        self.local_map_fofc, self.peak_volume_fofc = self.make_local_map(self.struct_data.fofc_map_data)
        self.local_map_2fofc, self.peak_volume_2fofc = self.make_local_map(self.struct_data.twofofc_map_data)
        #set peak heights of initial peak
        self.peak_fofc = self.density_at_point(self.struct_data.fofc_map_data,self.orig_xrs,self.coord)
        self.peak_2fofc = self.density_at_point(self.struct_data.twofofc_map_data,self.orig_xrs,self.coord)

        self.shaped_map_fofc = self.make_shaped_map(self.local_map_fofc)
        self.shaped_map_2fofc = self.make_shaped_map(self.local_map_2fofc)
        self.inv_map_fofc = self.make_round_map(self.local_map_fofc,2.0,True)
        self.inv_map_2fofc = self.make_round_map(self.local_map_2fofc,2.0,True)
        #build test pdbs of solvent molecules
        self.place_so4()
        self.place_water()
        self.symmetry = self.so4_pdb.crystal_symmetry()
        #must be called after the local map and place routines 
        self.map_model_box()

    def map_model_box(self):
        #used to use extract map_model, but that caused problems
        # here, just create a dummy pdb, put water at the peak site in the original coordinate system
        # the selection syntax is likely aware of symmetry, and will generate a list of all local atoms
        pdb_dummy=self.write_atom(1,"O","","HOH","ZZ",9999,"",self.coord[0],self.coord[1],self.coord[2],1.0,35.0,"O","")
        orig_str = self.orig_pdb_hier.as_pdb_string(write_scale_records=False, append_end=False, 
                                                    interleaved_conf=0, atoms_reset_serial_first_value=1, atom_hetatm=True, sigatm=False, 
                                                    anisou=False, siguij=False, output_break_records=False)
        comb = orig_str+pdb_dummy
        dummy_pdb = iotbx.pdb.input(source_info=None, lines=flex.split_lines(comb))
        dummy_hier = dummy_pdb.construct_hierarchy()
        dummy_xrs = dummy_pdb.xray_structure_simple()
        comb_sel_str = "chain ZZ and resid 9999" 
        comb_peak_atom = dummy_hier.atom_selection_cache().selection(string = comb_sel_str)
        select_within = dummy_xrs.selection_within(5.0,comb_peak_atom)
        atoms = dummy_hier.atoms()
        local_atoms = atoms.select(select_within)
        #find contacts in this local system
        self.contacts=[]
        for atom in local_atoms:
            distance = atom.distance(self.coord)
            if distance < 6.0:
                awl=atom.fetch_labels()
                resname=awl.resname.strip()
                chain=str(atom.chain().id).strip()
                element=atom.element.strip()
                name=atom.name.strip()
                altloc = awl.altloc.strip()
                resid = awl.resseq.strip()
                contact={"name":name,"chain":chain,"element":element,"distance":distance,
                         "resname":resname,"altloc":altloc,"resid":resid}
                self.contacts.append(contact)
        self.contacts.sort(key = lambda x: x['distance'])

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
        if solvent_content < 0.2:
            solvent_content = 0.2
        if solvent_content > 0.8:
            solvent_content = 0.8

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
        #coord - center
        x1 = float(coord1[0])
        y1 = float(coord1[1])
        z1 = float(coord1[2])
        x2 = float(coord2[0])
        y2 = float(coord2[1])
        z2 = float(coord2[2])
        vector1x = x2-x1
        vector1y = y2-y1
        vector1z = z2-z1
        lengthvector1 = np.sqrt(vector1x*vector1x + vector1y*vector1y + vector1z*vector1z)
        return(lengthvector1) 

    def place_so4(self,b_fac=35.0,occ=1.00):
         #create a dummy pdb, put sulfate
        pdb_string="CRYST1%9.3f%9.3f%9.3f  90.00  90.00  90.00 P 1            1\n" % \
            (2.0*self.bound,2.0*self.bound,2.0*self.bound)
        coord1 = self.coord
        #x1,y1,z1 = coord1[0:3]
        x1,y1,z1=5.0,5.0,5.0
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
        

    def place_water(self,b_fac=35.0,occ=1.00):
        #create a dummy pdb, put water
        pdb_string="CRYST1%9.3f%9.3f%9.3f  90.00  90.00  90.00 P 1            1\n" % \
            (2.0*self.bound,2.0*self.bound,2.0*self.bound)
        coord1 = self.coord
        x1,y1,z1=5.0,5.0,5.0
        #x1,y1,z1 = coord1[0:3]
        pdb_entry = ""
        pdb_entry=pdb_entry+self.write_atom(1,"O","","HOH","X",self.resid,"",x1,y1,z1,occ,b_fac,"O","")
        wat_pdb=pdb_string+pdb_entry
        #print wat_pdb
        self.water_pdb = iotbx.pdb.input(source_info=None, lines=flex.split_lines(wat_pdb))
        self.water_hier = self.water_pdb.construct_hierarchy()
        self.water_xrs = self.water_pdb.xray_structure_simple()


    def write_atom(self,serial,name,alt,resname,chain,resid,ins,x,y,z,occ,temp,element,charge):
        return "%-6s%5d %4s%1s%3s%2s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s\n" % \
            ("ATOM",serial,name,alt,resname,chain,resid,ins,x,y,z,occ,temp,element,charge)  

            
    def new_grid(self,coord,bound): 
        """
        returns coordinates for a new map grid
        coord is center of coordinates, bound is the bounding box
        grid is +/- bound in number of points
        currently fixed at 21 to give 0.5A grid (2A, 0.25fft)
        """
        npoints = 21
        center = coord
        origin = (coord[0]-bound,coord[1]-bound,coord[2]-bound)
        endpoint = (coord[0]+bound, coord[1]+bound, coord[2]+bound)
        gridx = np.linspace(origin[0],endpoint[0],npoints)
        gridy = np.linspace(origin[1],endpoint[1],npoints)
        gridz = np.linspace(origin[2],endpoint[2],npoints)
        mesh = np.meshgrid(gridx,gridy,gridz,indexing='ij')
        grid = np.vstack(mesh).reshape(3,-1).T
        return grid


