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
                         'renumber':False,'write_ref_pdb':False,
                         'write_maps':False,'ressig':False,
                         'write_contacts':False}

        #command line expects 5 args as listlike (NB: filenames) 
        #   resolution,input_pdb,input_strip_pdb,peaks_pdb,input_map_mtz 
        #   output_code is optional
        #phenix params should be a file formatted by phil
        #should be processed properly by phil . . . 
        if cmdline:
            self.score_res,self.model_pdb,self.strip_pdb,self.peaks_pdb,self.map_coeff = cmdline[0:5]
            if len(cmdline) == 6:
                if len(cmdline[5]) > 3:
                    self.out_prefix = cmdline[5]
            else:
                self.out_prefix = 'user'
        if phenix_params:
            phil_f = open(phenix_params,'r')
            phil_str = phil_f.read()
            phil_f.close()
            master_phil = phil.parse(phil_str)
            self.phe_par = master_phil.extract()
            self.model_pdb = self.phe_par.input.pdb.model_pdb[0]
            self.strip_pdb = self.phe_par.input.pdb.strip_pdb[0]
            self.peaks_pdb = self.phe_par.input.pdb.peaks_pdb[0]
            self.map_coeff = self.phe_par.input.input_map.map_coeff_file[0]
            self.score_res = self.phe_par.input.parameters.score_res[0]
            self.out_prefix  = self.phe_par.output.output_peak_prefix[0][0:4]
            """
            fix dev options here
            """
    def feature_extract(self):
        # Master Feature Extraction for User Classification
        pdb_code = self.out_prefix
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
            if self.dev_opts['write_contacts']:
                ppio.store_contacts(features)
            ppfeat.analyze_features(features)

            peaks_features.append(features)

            #generate some output
            peak_status=""
            for flag,value in features['peak_flags'].iteritems():
                if value == True:
                    peak_status = peak_status+" "+flag

            if peak_status == "":
                peak_status = "OKAY"

            #identify closest contact in stripped pdb
            if len(features['strip_contacts']) > 0:
                closest_contact = features['strip_contacts'][0]
                cc_name,cc_res,cc_resid,cc_chain,cc_dist = closest_contact['name'],closest_contact['resname'],\
                                                           closest_contact['resid'],closest_contact['chain'],\
                                                           closest_contact['distance']

                cc_id = cc_chain+str(cc_resid)+"_"+cc_res+"_"+cc_name
            else:
                cc_id = "Cdist>6.0"
                cc_dist =9.99
            pfmt = lambda val: "{0!s:^5s}".format(val)[0:4]
            print "   Peak %s 2FoFc: %s FoFc %s is %sA from %12s STATUS: %s" % (features['db_id'],pfmt(features['2fofc_sig_out']),
                                                                                pfmt(features['fofc_sig_out']),pfmt(cc_dist),
                                                                                cc_id,peak_status)
        print "Finished Feature Extraction"
        return peaks_features




    def score_features(self,features,plot=False):
        print "-"*79
        print "CLASSIFICATION RUN %s PEAKS:" % features.shape[0]
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

    def classify_peaks(self,results_array,cfeat_array):
        print "-"*79
        print "ASSIGNING PEAK CLASSES:"

        ppcf = ClassifierFunctions()
        flag_class = ppcf.peak_fc(cfeat_array)
        edc = ppcf.peak_edc(results_array)
        c_class = ppcf.peak_cc(results_array,cfeat_array)

        warnd = {'spl':' SPL_POS?','rem':' REMOTE','bc':' BAD_CONT','rej':' REJECTED',
                 'close':' 1_BAD_CONT','chk':' INSPECT_MANUALLY','weak':' WEAK_2FOFC',
                 'mdl':' MODEL_ERROR?--chk_ROT/ALT','unk': ' AMBIGUOUS'}
     
        fmt_scr = lambda result: " ".join(('{:5.2f}'.format(x)[0:5] for x in (result['llgS'],result['llgW'],
                                                                     result['chiS'],result['chiW'])))

        ori_class = np.zeros(cfeat_array.shape[0],dtype=np.int16)
        ori_class[cfeat_array['ori'] == 'HOH'] = 1
        ori_class[cfeat_array['ori'] == 'SO4'] = 2
        ori_class[cfeat_array['ori'] == 'PO4'] = 2
        ori_class[cfeat_array['ori'] == 'XXX'] = 3
        ori_class[ori_class == 0] = 4 #other
        #     #user_P_00001 19.07 -12.8 12.91 126.9   2 1 1 1 0
        print "   PPROBE       SCORES       FITS       CLASSES"                       
        print "Peak ID tag   llgS  llgW  chiS  shiW   I P E C F  Results"           
        pred = np.array(results_array['score'] > 0,dtype=np.int16)
        for index,peak in enumerate(cfeat_array):
            ori = ori_class[index]
            p_edc = edc[index]
            p_cc = c_class[index]
            p_fc = flag_class[index]
            
            scoref = fmt_scr(results_array[index])
            warn = ""
            """
            if peak['flagc'] == 1:
                warn = warn+warnd['spl']+warnd['chk']
            if peak['flagc'] == 2:
                warn = warn+warnd['bc']+warnd['rej']
            if peak['flagc'] == 3:
                warn = warn+warnd['bc']+warnd['rej']
            if peak['flagc'] == 4:
                warn = warn+warnd['bc']+warnd['chk']
            if peak['flagc'] == 5:
                warn = warn+warnd['close']+warnd['chk']
            if peak['flagc'] == 6:
                warn = warn+warnd['rem']+warnd['chk']
            if peak['weak']:
                warn = warn+warnd['weak']
            if peak['edc'] == 3:
                warn = warn+warnd['mdl']
            if peak['edc'] == 4:
                warn = warn+warnd['unk']

            if peak['edc'] == 1:
                pred = 'SO4'
            if peak['edc'] == 2:
                pred = 'HOH'
            if peak['edc'] == 3:
                pred = 'ERR'
            if peak['edc'] == 4:
                pred = 'UNK'

            if warn == "":
                if ori == pred:
                    warn = " GOOD->keep"
                elif ori == "XXX":
                    if pred == "SO4":
                        warn = " NEW->SO4"
                    if pred == "HOH":
                        warn = " NEW->HOH"
            if ori == "SO4" and pred == 'HOH':
                warn = warn+" BAD->unlikely SO4"
            if ori == "HOH" and pred == 'SO4':
                warn = warn+" BAD->check model"
            """         

                
            print "%s %s %3s %1d %1d %1d %1d %s" % (peak['id'],scoref,ori,pred[index],p_edc,p_cc,p_fc,warn)
            #print cfeat_arr[index]

        






