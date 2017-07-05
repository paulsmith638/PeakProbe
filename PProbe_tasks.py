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
from PProbe_extract import FeatureExtraction
from PProbe_selectors import Selectors
from PProbe_classify import ClassifierFunctions


"""
Things needed here:
make maps from ED data
strip pdb in various ways
setup all hierarchies
tasks for training, analysis, rejecting etc.
"""



class PProbeTasks:
    def __init__(self):
        pass


    def setup_datafiles(self):
        pass




    def feature_ex_by_pdbid(self,pdbid,write_ref_pdb=False,write_maps=False):
        pdb_code = pdbid
        db_filename = pdb_code+"_data.db"
        peaks_features = []
        # file_name for original structure with all atoms
        orig_pdb_file = "/home/paul/PDB/data/structures/all/pdb/pdb"+pdb_code+".ent.gz"
        # original file stripped of water, phosphate, and sulfate
        strip_pdb_file = "/home/paul/PDB/phenix_data/"+pdb_code+"_strip.pdb" #just for symmetry info
        # SET SOURCE FOR PEAKS HERE
        peak_pdb_file = "/home/paul/PDB/phenix_data/python/"+pdb_code+"_sw_peaks.pdb" 
        # file name for mtz format 2fofc and fofc map coeffs
        map_file = "/home/paul/PDB/phenix_data/"+pdb_code+"_peaks_maps.mtz"
        orig_pdb_obj = iotbx.pdb.input(orig_pdb_file)
        strip_pdb_obj = iotbx.pdb.input(strip_pdb_file)
        peak_pdb_obj = iotbx.pdb.input(peak_pdb_file)
        # construct all 3 hierarchies
        orig_pdb_hier = orig_pdb_obj.construct_hierarchy()
        strip_pdb_hier = strip_pdb_obj.construct_hierarchy()
        peak_pdb_hier = peak_pdb_obj.construct_hierarchy()
        symmetry = strip_pdb_obj.crystal_symmetry() #passes the original symmetry
        #generates an object that contains complete asu maps and geometry restraints for SO4 and water
        struct_data=StructData(pdb_code,symmetry,orig_pdb_hier,strip_pdb_hier,peak_pdb_hier,map_file)
        #get a list of dictionaries for each peak
        ppfeat = FeatureExtraction()
        peak_list = ppfeat.generate_peak_list(pdb_code,peak_pdb_hier)
        pput = Util()
        lookup_resolution = pput.assign_resolution(pdb_code)
        for peak in peak_list:
            if peak['chainid'] == "S" or peak['chainid'] == "W":
                features = {}
                features['db_id'] = peak['db_id']
                peak_object,ref_object = ppfeat.generate_peak_object(peak,symmetry,orig_pdb_hier,strip_pdb_hier,peak_pdb_hier,struct_data,write_maps=write_maps)
                rsr_object = RealSpace(peak_object,ref_object,features,ressig=False)
                rsr_object.refinement(peak,peak_object,ref_object,write_pdb=write_ref_pdb,outstr="")
                rsr_object.rotations(peak_object,ref_object,write_pdb=write_ref_pdb)
                features = rsr_object.features

                features['resolution'] = lookup_resolution
                features['bin'] = pput.assign_bin(features['resolution'])
                features['omit'] = 0#boolean, omit if "1"
                features['pdb_code']= pdb_code
                ppfeat.basic_features(features,peak_object)
                features['solc'] = struct_data.solvent_content
                peaks_features.append(features)
        #returns a list of lists each containing result values, objects, etc.
        return peaks_features

    def feature_ex_by_input(self,resolution,input_pdb,input_strip_pdb,peaks_pdb,input_map_mtz,
                            write_ref_pdb=False,
                            pdb_out_str='',
                            ressig=False,
                            write_maps=False,
                            setchain=False,
                            renumber=False):
        pdb_code = 'user'
        peaks_features = []
        # file_name for original structure with all atoms
        orig_pdb_file = input_pdb
        # original file stripped of selected heteroatoms
        strip_pdb_file = input_strip_pdb
        # SET SOURCE FOR PEAKS HERE
        peak_pdb_file = peaks_pdb
        # file name for mtz format 2fofc and fofc map coeffs
        map_file = input_map_mtz
        orig_pdb_obj = iotbx.pdb.input(orig_pdb_file)
        strip_pdb_obj = iotbx.pdb.input(strip_pdb_file)
        peak_pdb_obj = iotbx.pdb.input(peak_pdb_file)
        # construct all 3 hierarchies
        orig_pdb_hier = orig_pdb_obj.construct_hierarchy()
        strip_pdb_hier = strip_pdb_obj.construct_hierarchy()
        peak_pdb_hier = peak_pdb_obj.construct_hierarchy()
        symmetry = strip_pdb_obj.crystal_symmetry() #passes the original symmetry
        #generates an object that contains complete asu maps and geometry restraints for SO4 and water
        struct_data=StructData(pdb_code,symmetry,orig_pdb_hier,strip_pdb_hier,peak_pdb_hier,map_file)
        #get a list of dictionaries for each peak
        ppfeat = FeatureExtraction()
        peak_list = ppfeat.generate_peak_list(pdb_code,peak_pdb_hier,setchain=setchain,renumber=renumber)
        pput = Util()
        for peak in peak_list:
            features = {}
            features['db_id'] = peak['db_id']
            peak_object,ref_object = ppfeat.generate_peak_object(peak,symmetry,orig_pdb_hier,strip_pdb_hier,peak_pdb_hier,struct_data,write_maps=write_maps)
            rsr_object = RealSpace(peak_object,ref_object,features,ressig=ressig)
            rsr_object.refinement(peak,peak_object,ref_object,write_pdb=write_ref_pdb,outstr=pdb_out_str)
            rsr_object.rotations(peak_object,ref_object,write_pdb=write_ref_pdb)
            features = rsr_object.features
            features['resolution'] = resolution
            features['bin'] = pput.assign_bin(features['resolution'])
            features['omit'] = 0 #boolean, omit if "1"
            features['pdb_code']= pdb_code
            ppfeat.basic_features(features,peak_object)
            features['solc'] = struct_data.solvent_content
            peaks_features.append(features)
        #returns a list of lists each containing result values, objects, etc.
        return peaks_features




    def score_features(self,features,plot=False):
        raw_data = np.sort(features,order=['res','id'])
        selectors = Selectors(raw_data)
        ppcf = ClassifierFunctions()
        norm_data = ppcf.standardize_data(raw_data,post_pca=False)
        pca_data = ppcf.pca_xform_data(norm_data,plot=True)
        pca_scaled_data = ppcf.standardize_data(pca_data,post_pca=True)
        results_array = ppcf.initialize_results(pca_scaled_data)
        ppcf.discriminant_analysis(pca_scaled_data,results_array,plot=False)
        ppcf.score_breakdown(pca_scaled_data,results_array)
        return results_array

        
