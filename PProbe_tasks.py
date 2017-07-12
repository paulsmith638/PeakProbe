from __future__ import division
import sys,re,copy,os
import numpy as np
#cctbx imports
import iotbx.pdb
from iotbx import phil
#from mmtbx import monomer_library
#from libtbx import group_args
#from cctbx import miller
from scitbx.array_family import flex
from iotbx import reflection_file_utils
from iotbx import crystal_symmetry_from_any
from libtbx.utils import Sorry
#from mmtbx.command_line import geometry_minimization
#import mmtbx.refinement.geometry_minimization
#from cctbx import maptbx
#import scitbx.rigid_body
#import mmtbx.ncs
#import mmtbx.refinement.real_space.individual_sites as rsr
#import mmtbx.refinement.real_space.rigid_body as rigid
#import iotbx.ncs
#import mmtbx.maps.correlation
#from iotbx.ccp4_map import write_ccp4_map
#PProbe imports
from PProbe_peak import PeakObj
from PProbe_struct import StructData
from PProbe_ref import RSRefinements
from PProbe_util import Util
from PProbe_realspace import RealSpace
from PProbe_extract import FeatureExtraction
from PProbe_selectors import Selectors
from PProbe_classify import ClassifierFunctions
from PProbe_dataio import DataIO


class PProbeTasks:
    def __init__(self,cmdline=False,phenix_params=False):
        #we need one or the other
        assert cmdline or phenix_params
        #setup some dev options
        self.dev_opts = {'set_chain':False,'pdb_out_str':"",
                         'renumber':False,'write_ref_pdb':True,
                         'write_maps':True,'ressig':False}

        #command line expects 5 args as listlike (NB: filenames) 
        #   resolution,input_pdb,input_strip_pdb,peaks_pdb,input_map_mtz 
        #phenix params should be a file formatted by phil
        if cmdline:
            self.score_res,self.model_pdb,self.strip_pdb,self.peaks_pdb,self.map_coeff = cmdline[0:5]
            self.out_prefix = str(self.model_pdb)
        if phenix_params:
            phil_f = open(phenix_params,'r')
            phil_str = phil_f.read()
            phil_f.close()
            master_phil = phil.parse(phil_str)
            self.phe_par = master_phil.extract()
            self.model_pdb = self.phe_par.input.pdb.model_pdb
            self.strip_pdb = self.phe_par.input.pdb.strip_pdb
            self.peaks_pdb = self.phe_par.input.pdb.peaks_pdb
            self.map_coeff = self.phe_par.input.input_map.map_coeff_file
            self.score_res = self.phe_par.input.parameters.score_res
            self.out_prefix =self.phe_par.output.output_file_name_prefix
            """
            fix dev options here
            """
    def feature_extract(self):
        # Master Feature Extraction for User Classification

        pdb_code = 'user'
        peaks_features = []
        #here we go:
        map_file = self.map_coeff
        orig_pdb_obj = iotbx.pdb.input(self.model_pdb)
        strip_pdb_obj = iotbx.pdb.input(self.strip_pdb)
        peak_pdb_obj = iotbx.pdb.input(self.peaks_pdb)
        # construct all 3 hierarchies
        orig_pdb_hier = orig_pdb_obj.construct_hierarchy()
        strip_pdb_hier = strip_pdb_obj.construct_hierarchy()
        peak_pdb_hier = peak_pdb_obj.construct_hierarchy()
        #recheck symmetry
        syms = []
        for datain in [self.model_pdb,self.strip_pdb,self.peaks_pdb,self.map_coeff]:
            cs = crystal_symmetry_from_any.extract_from(datain)
            if(cs is not None): 
                syms.append(cs)
                for test_sym in syms:
                    if (not test_sym.is_similar_symmetry(cs)):
                        raise Sorry("Crystal symmetry mismatch between different files.")
        if len(syms) == 0:
            raise Sorry("No Crystal Symmetry Found!")
        else:
            symmetry = syms[0]
        #generates an object that contains complete asu maps and geometry restraints for SO4 and water
        struct_data=StructData(pdb_code,symmetry,orig_pdb_hier,strip_pdb_hier,peak_pdb_hier,map_file)
        #get a list of dictionaries for each peak
        ppfeat = FeatureExtraction()
        peak_list = ppfeat.generate_peak_list(pdb_code,peak_pdb_hier,set_chain=self.dev_opts['set_chain'],
                                              renumber=self.dev_opts['renumber'])
        pput = Util()
        ppio = DataIO()
        print "Extrating Features from %d Peaks: " % len(peak_list)
        for peak in peak_list:
            features = {}
            features['db_id'] = peak['db_id']
            peak_object,ref_object = ppfeat.generate_peak_object(peak,symmetry,orig_pdb_hier,strip_pdb_hier,
                                                                 peak_pdb_hier,struct_data,
                                                                 write_maps=self.dev_opts['write_maps'])
            rsr_object = RealSpace(peak_object,ref_object,features,ressig=self.dev_opts['ressig'])
            rsr_object.refinement(peak,peak_object,ref_object,write_pdb = self.dev_opts['write_ref_pdb'],
                                  outstr=self.dev_opts['pdb_out_str'])
            rsr_object.ref_contacts()
            rsr_object.rotations(peak_object,ref_object,write_pdb=self.dev_opts['write_ref_pdb'])
            #features dictionary
            #features = rsr_object.features
            features['resolution'] = self.score_res
            features['bin'] = pput.assign_bin(features['resolution'])
            features['omit'] = 0 #boolean, omit if "1", for training
            features['pdb_code']= pdb_code
            ppfeat.basic_features(features,peak_object)
            features['solc'] = struct_data.solvent_content
            ppio.store_contacts(features)
            ppfeat.analyze_features(features)

            peaks_features.append(features)

            #generate some output
            if features['pchar'] == "badw":
                peak_status = "BAD PEAK -- model error?"
            elif features['pchar'] == "bads":
                peak_status = "TIGHT FIT -- SO4 unlikely"
            elif features['pchar'] == "2far":
                peak_status = "FAR AWAY -- spurious?"
            elif features['pchar'] == "weak":
                peak_status = "WEAK 2FoFc -- suspect?"
            elif features['pchar'] == "splp":
                peak_status = "Possible Special Position"
            elif features['pchar'] == "good":
                peak_status = "PASSED"
            peak_ss = "".join((str(x) for x in features['panal']))
            peak_tally = peak_ss[0:5]+"-"+peak_ss[5:10]+"-"+peak_ss[10:15]+"-"+peak_ss[15]
            if len(features['strip_contacts']) > 0:
                closest_contact = features['strip_contacts'][0]
                cc_name,cc_res,cc_resid,cc_chain,cc_dist = closest_contact['name'],closest_contact['resname'],\
                                                           closest_contact['resid'],closest_contact['chain'],\
                                                           closest_contact['distance']

                cc_id = cc_chain+str(cc_resid)+"_"+cc_res+"_"+cc_name
            else:
                cc_id = "Cdist>6.0"
                   
            print "   Peak %s 2FoFc: %4.2f FoFc %4.2f is %3.2fA from %14s %s STATUS: %s" % (features['db_id'],features['2fofc_sig_out'],
                                                                                    features['fofc_sig_out'],cc_dist,cc_id,peak_tally,
                                                                                    peak_status)
            #print "      ",features['panal']
        #returns a list of dictionaries each containing result values, objects, etc.
        print "Finished Feature Extraction"
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

        
