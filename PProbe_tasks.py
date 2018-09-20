from __future__ import division
import sys,re,copy,os,ast,time
import numpy as np

class PhenixTasks:
    #tasks requiring cctbx, run under "phenix.python"
    def __init__(self,cmdline=False,phenix_params=False):
        #cctbx imports
        global iotbx,phil,flex,reflection_file_utils,crystal_symmetry_from_any,Sorry,easy_pickle
        import iotbx.pdb 
        from iotbx import phil
        from scitbx.array_family import flex
        from iotbx import reflection_file_utils
        from iotbx import crystal_symmetry_from_any
        from libtbx.utils import Sorry
        from libtbx import easy_pickle
        #PProbe imports
        global PPpeak,PPstruct,PPref,PPutil,PPreal,PPfeat,PPsel,PPio,PPcont
        from PProbe_peak import PeakObj as PPpeak
        from PProbe_struct import StructData as PPstruct
        from PProbe_ref import RSRefinements as PPref
        from PProbe_util import Util as PPutil
        from PProbe_realspace import RealSpace as PPreal
        from PProbe_extract import FeatureExtraction as PPfeat
        from PProbe_selectors import Selectors as PPsel
        from PProbe_dataio import DataIO as PPio
        from PProbe_contacts import Contacts as PPcont
        self.ppfeat = PPfeat()
        self.pput = PPutil()
        self.ppio = PPio()
        self.ppcont = PPcont(phenix_python=True)

        #we need one or the other
        assert cmdline or phenix_params
        #setup some dev options
        self.dev_opts = {'set_chain':False,'pdb_out_str':"",
                         'renumber':False,'write_ref_pdb':False,
                         'write_maps':False,'ressig':False,
                         'write_contacts':False,'proc_sol':True}

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
            self.phe_par = None
            self.cmdline = " ".join(par for par in cmdline)
        if phenix_params:
            phil_f = open(phenix_params,'r')
            phil_str = phil_f.read()
            phil_f.close()
            pprobe_phil = phil.parse(phil_str)
            self.phe_par = pprobe_phil.extract()
            extract_asstr = self.phe_par.pprobe.extract[0]
            extract = extract_asstr[0].lower() == "t"
            if extract:
                self.model_pdb = self.phe_par.input.pdb.model_pdb[0]
                self.strip_pdb = self.phe_par.input.pdb.strip_pdb[0]
                self.peaks_pdb = self.phe_par.input.pdb.peaks_pdb[0]
                self.map_coeff = self.phe_par.input.input_map.map_coeff_file[0]
            mdict_file = self.phe_par.input.model_param.model_dict_file[0]
            self.master_dict = self.ppio.read_master_dict(input_dfile=mdict_file)
            scres = self.phe_par.input.parameters.score_res
            if scres is not None:
                self.score_res = self.phe_par.input.parameters.score_res[0]
            self.out_prefix  = self.phe_par.output.output_peak_prefix[0][0:4]
            # I can't figure out this phil parse thing, so bad hack here
            for option in self.phe_par.input.parameters.map_omit_mode:
                if option[0] == "*":
                    self.omit_mode = option[1::]

            """
            fix dev options here
            """
            if self.omit_mode == "valsol":
                self.dev_opts['proc_sol'] = False
            self.cmdline=""



    def feature_extract(self):
        # Master Feature Extraction 
        ppfeat = self.ppfeat
        ppcont = self.ppcont
        pput = self.pput
        ppio = self.ppio
        pdb_code = self.out_prefix

        #here we go:

        #read in all data/params
        map_file = self.map_coeff
        orig_pdb_obj = iotbx.pdb.input(self.model_pdb)
        strip_pdb_obj = iotbx.pdb.input(self.strip_pdb)
        peak_pdb_obj = iotbx.pdb.input(self.peaks_pdb)

        # construct all 3 hierarchies
        orig_pdb_hier = orig_pdb_obj.construct_hierarchy()
        strip_pdb_hier = strip_pdb_obj.construct_hierarchy()
        peak_pdb_hier = peak_pdb_obj.construct_hierarchy()
        orig_pdb_hier.remove_hd()
        strip_pdb_hier.remove_hd()
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
        struct_data=PPstruct(pdb_code,symmetry,orig_pdb_hier,strip_pdb_hier,peak_pdb_hier,map_file,self.score_res)

        null_hier = struct_data.gen_null_peak()
        omit_hier = ppcont.omited_peaks(orig_pdb_hier,strip_pdb_hier)
        solvent_hier = ppcont.solvent_peaks(orig_pdb_hier)
        merge_hier = ppcont.merge_hier([strip_pdb_hier,omit_hier,solvent_hier,peak_pdb_hier],symmetry)

        #get a list of dictionaries for each peak type
        input_peak_list = ppfeat.generate_peak_list(pdb_code[0:4],peak_pdb_hier,4,set_chain=self.dev_opts['set_chain'],renumber=self.dev_opts['renumber'])
        omit_peak_list = ppfeat.generate_peak_list("orip",omit_hier,2)
        sol_peak_list = ppfeat.generate_peak_list("solp",solvent_hier,3)
        null_peak_dict = ppfeat.generate_peak_list("null",null_hier,5)[0] #null unal = -8861610501908601326
        natoms = merge_hier.atoms().size()
        print "   Collecting contacts for %d atoms" % natoms
        #generates all contacts, very big dictionary of lists
        contact_by_unal = ppcont.peak_allcont(merge_hier,symmetry)

        #creates master peak dictionary indexed by unique identifier "unal"
        peak_unal_db = {}
        #null peak for later analysis and storing program parameters,inputs,etc.
        null_peak_dict['info'] = {"param":self.phe_par,"cmdline":self.cmdline,"symmetry":symmetry}
        null_peak_dict['inputs'] = {"ori_hier":orig_pdb_hier}
        peak_unal_db[null_peak_dict['unal']] = null_peak_dict
        for plist in [input_peak_list,omit_peak_list,sol_peak_list]:
            for pdict in plist:
                pdict['peak_unal_db'] = peak_unal_db #keep reference to all peaks in the peak itself
                pdict['cont_db'] = contact_by_unal #reference to all contacts
                pdict['master_dict'] = self.master_dict #reference to paster dictionary
                peak_unal_db[pdict['unal']] = pdict
        
        omit_unat = list(peak['unat'] for peak in omit_peak_list)
        sol_unat = list(peak['unat'] for peak in sol_peak_list)
        omit_sol_unat = list(set(omit_unat) & set(sol_unat))

        sol_peaks_toadd = []
        added_sol_unat = []
        print "\nExtracting Features from %d Input Peaks: " % len(input_peak_list)        
        print "     peak_id      Density Level (unscaled)           Peak Env           Neighbor Macro       Assc. Solvent"

        #loop through each peak and extract density data

        for pdict in input_peak_list:
            p_unal = pdict['unal']
            close_solvent = ppcont.prune_cont(pdict['cont_db'][p_unal],omit_models=[1,2,4],uniquify=True,unires=True,omit_null=True,cutoff=3.5)
            cls_count = 0
            if self.dev_opts['proc_sol']: #process associated solvent atoms?
                for cont in close_solvent:
                    if cont['unat'] in omit_sol_unat and cont['unat'] not in added_sol_unat:
                        sol_peaks_toadd.append(cont['unal'])
                        added_sol_unat.append(cont['unat'])
                        cls_count = cls_count + 1
            processed = self.peak_process(pdict,symmetry,orig_pdb_hier,strip_pdb_hier,peak_pdb_hier,struct_data,
                                          write_maps=self.dev_opts['write_maps'])
            if processed == "ERROR":
                pdict['status'] = 1
                continue
            else:
                pdict['status'] = 0
            outstr = ppio.initial_report(pdict)
            if cls_count > 0:
                outstr = outstr + " --> added %d close solvent" % cls_count
            print outstr

        print "\nExtracting Features from %d Solvent Model Peaks: " % len(sol_peaks_toadd)   
        for p_unal in sol_peaks_toadd:
            pdict = peak_unal_db[p_unal]
            processed = self.peak_process(pdict,symmetry,orig_pdb_hier,strip_pdb_hier,peak_pdb_hier,struct_data,
                                          write_maps=self.dev_opts['write_maps'])
            
            if processed == "ERROR":
                pdict['status'] = 1
                continue
            pdict['status'] = 5
        sol_noproc = list(pdict['unal'] for pdict in peak_unal_db.values() if (pdict['model'] == 3 and pdict['unat'] not in added_sol_unat))
        for unal in sol_noproc:
            pdict = peak_unal_db[unal]
            processed = self.peak_process(pdict,symmetry,orig_pdb_hier,strip_pdb_hier,peak_pdb_hier,struct_data,full_proc=False)
            if processed == "ERROR":
                pdict['status'] = 1
            else:
                pdict['status'] = 6

        print "   --> DONE!"
        print "Finished Feature Extraction"
        print "-"*79
        return peak_unal_db




    def peak_process(self,pdict,symmetry,orig_pdb_hier,strip_pdb_hier,
                     peak_pdb_hier,struct_data,write_maps=False,full_proc=True):
        ppfeat = self.ppfeat
        pput = self.pput
        ppio = self.ppio
        ppcont = self.ppcont
        p_unal = pdict['unal']
        pdict['pdb_code']= self.out_prefix
        if pdict['model'] in  [1,2]:
            pdict['status'] = 7
        pdict['warnings'] = []
        pdict['resolution'] = struct_data.resolution
        pdict['bin'] = struct_data.res_bin
        pdict['omit'] = 0 #boolean, omit if "1", for training/cv

        #process contacts for all peaks
        ppcont.process_contacts(pdict)

        #stop here if peak does not have omit data (maps,etc)
        if not full_proc:
            return

        peak_object,ref_object = ppfeat.generate_peak_object(pdict,symmetry,orig_pdb_hier,strip_pdb_hier,
                                                             peak_pdb_hier,struct_data,write_maps=self.dev_opts['write_maps'])

        rsr_object = PPreal(peak_object,ref_object,pdict,pdict['strip_contacts'],ressig=self.dev_opts['ressig'])
        rsr_object.refinement(pdict,peak_object,ref_object,write_pdb = self.dev_opts['write_ref_pdb'],
                              outstr=self.dev_opts['pdb_out_str'])
        rsr_object.ref_contacts()
        ppcont.parse_contacts(pdict)
        rsr_object.rotations(peak_object,ref_object,write_pdb=self.dev_opts['write_ref_pdb'])
        pdict['solc'] = struct_data.solvent_content
        ppfeat.basic_features(pdict,peak_object)


        #look for clashes and solvent model
        pdict['clash'] = ppcont.cflag_pass1(pdict)
        


            



class PProbeTasks:
    #other tasks that might conflict with phenix (matplotlib, scipy, etc.)
    def __init__(self):
        global PPclass,PPfilt,PPcont,PPsel,PPtrain,PPutil,PPKDE,PPio,PPout
        from PProbe_classify import ClassifierFunctions as PPclass
        from PProbe_filter import Filters as PPfilt
        from PProbe_contacts import Contacts as PPcont
        from PProbe_selectors import Selectors as PPsel
        from PProbe_util import Util as PPutil
        from PProbe_kde import KDE as PPKDE
        from PProbe_dataio import DataIO as PPio
        from PProbe_output import Output as PPout

    def post_process(self,all_peak_db,data_array,peak_feat_list):
        if len(peak_feat_list) == 0:
            return
        #sig_flags flags outliers by absolute cutoffs, np_array index
        #sat_dict is dict of (id,score) for possible satellite peaks, by db_id
        #prob arrays, labels are np array indexed
        #updates existing dictionary
        ppclass = PPclass(verbose=False)
        ppio = PPio()
        ppfilt = PPfilt(verbose=False)
        #score by 2D histogram matching

        self.kde_score(data_array)
        #reassign flags based on histogram derived probabilities
        data_array['rc'] = ppclass.score_flags(data_array)
        sig_flags = ppfilt.cutoff_rejects(data_array,data_array.dtype.names)
        sigr_unal = list(data_array[sig_flags]['unal'])
        n_outliers = len(sigr_unal)
        if n_outliers > 0:
            print "-->FLAGGED %s PEAK(S) AS OUTLIERS (Status = 2)" % n_outliers
        
        ppio.update_from_np(all_peak_db,data_array)
        ppclass.score_peaks(all_peak_db)
        for pind,pdict in enumerate(peak_feat_list):

            if pdict['unal'] in sigr_unal:
                pdict['status'] = 2
            #remove pointers from dictionary to hierarchies,xrs, etc. (easier/smaller to pickle)
            for key in ["so4_fofc_ref_hier","so4_i2fofc_ref_hier","so4_2fofc_ref_hier","wat_2fofc_shift",
                        "wat_fofc_ref_xrs","so4_in_hier","wat_in_hier","so4_ifofc_ref_hier","wat_fofc_ref_hier",
                        "so4_ifofc_ref_xrs","so4_2fofc_ref_xrs","wat_2fofc_ref_xrs","wat_2fofc_ref_hier",
                        "so4_2fofc_shift","so4_fofc_ref_xrs","so4_i2fofc_ref_xrs"]:
                pdict.pop(key,None)

    def data_process(self,master_array):
        ppclass = PPclass(verbose=False)
        ppfilt = PPfilt()
        ppcont = PPcont(phenix_python=False)
        ppio = PPio()
        mdict = ppio.read_master_dict()
        print "PROCESSING FEATURES"
        master_array.sort(order=['res','id'])
        selectors = PPsel(master_array)
        #clip resolution to trained region
        master_array['res'] = np.clip(master_array['res'],0.60,4.999).astype(np.float16)
        proc_array = np.zeros(master_array.shape[0],dtype=selectors.proc_array_dtype)
        for col in ['id','ori','res','ccSf','ccWf','ccS2','ccW2','ccSifi','ccSifo','ccSi2i','ccSi2o',
                    'ccSifr','ccSi2r','ccWif','ccWi2','ccSf60','sdSf60','ccS260','sdS260','vf','v2','2fofc_sigo']:
            proc_array[col] = master_array[col]
        ppclass.standardize_data(proc_array,post_pca=False)
        ppclass.pca_xform_data(proc_array)
        ppclass.standardize_data(proc_array,post_pca=True)
        for column in selectors.pca_view_col:
            proc_array[column] = np.nan_to_num(proc_array[column])
        print "     Processed features for %s peaks" % master_array.shape[0]
        ppclass.density_da(proc_array,master_array)
        ppclass.contact_da(master_array)
        ppclass.standardize_data(master_array,post_pca=True,composite=True)
        ppclass.chi_prob(master_array)
        print "     Scored ED/CC for %s peaks" % master_array.shape[0]
        master_array['rc']  = ppclass.score_flags(master_array)
        master_array['fc']  = ppclass.peak_fc(master_array)
        master_array['edc'] = ppclass.peak_edc(master_array)
        master_array['cc']  = ppclass.peak_cc(master_array)


    def validate_peaks(self,all_peak_db):
        # updates peak status from a dictionary of peaks
        # then updates status by confirming w and s
        # using these confirmed peaks as anchors when necessary, update contacts
        # rescore until no status changes
        ppclas = PPclass()
        ppcont = PPcont(phenix_python=False)
        ppio = PPio()
        peak_list = list(pdict['unal'] for pdict in all_peak_db.values() if pdict['model'] == 4)
        sol_list = list(pdict['unal'] for pdict in all_peak_db.values() if pdict['model'] == 3)
        ppcont.associate_models(all_peak_db,peak_list,sol_list)
        ppcont.cluster_analysis(all_peak_db)
        #for pdict in all_peak_db.values():
        #    if pdict['status'] > 100 or pdict['status'] in [-1,1,3,6,7]:
        #        continue
        #    else:
        #        if pdict['model'] in [3,4] and pdict['status'] not in [-1,1,2,3,6,7]:
        #            if (pdict['edc'] in [0,4,5] or #suspicious density
        #                pdict['fc'] in [2,3,6] or #bad clash or very weak density
        #                pdict['cc'] in [0,4,5]): #junk peak
        #                pdict['status'] = 4
        converged = False
        iterations = 0
        while not converged:
            updated_peaks= ppclas.update_anchors(all_peak_db,allow_cross=True)
            iterations = iterations + 1
            if updated_peaks > 0 and iterations < 5:
                print "Updated status for %s peak(s),rescoring cycle %s . . . " % (updated_peaks,iterations)
                input_feat = list(pdict  for pdict in all_peak_db.values() if pdict['model'] == 4 and pdict['status'] not in [1,3,6,7])
                if len(input_feat) > 0:
                    master_array=ppio.extract_raw(input_feat)
                    self.data_process(master_array)
                    self.post_process(all_peak_db,master_array,input_feat)
                ori_feat = list(pdict  for pdict in all_peak_db.values() if (pdict['model'] == 3 and pdict['status'] not in [1,3,6,7]) )
                if len(ori_feat) > 0:
                    assp_array=ppio.extract_raw(ori_feat)
                    self.data_process(assp_array)
                    self.post_process(all_peak_db,assp_array,ori_feat)
                print "-"*79
            else:
                converged = True

        print "DONE -- UPDATING MODELS AND CONTACTS"
        ppclas.update_all_models(all_peak_db)
        ppclas.bad_contacts(all_peak_db)
        ppclas.score_peaks(all_peak_db)
        ppclas.ws_pass1(all_peak_db)
        #for pdict in all_peak_db.values():
        #    if pdict['status'] not in [1,3,6,7] and pdict['model'] in [3,4]:
        #        ppclas.peak_summary(pdict)
    def kde_score(self,master_array):
        #master_array = np.sort(master_array,order=['id'])
        ppio = PPio()
        pput = PPutil()
        mdict = ppio.read_master_dict()
        ppkde = PPKDE(mdict,verbose=True)
        kde_probs = ppkde.kde_score(master_array)
        master_array['kde'] = kde_probs
        master_array['lab'] = ppkde.kde_label(master_array)
        for pind,probs in enumerate(kde_probs):
            master_array['pick'][pind] = pput.pick_from_prob(probs)

    def train_model(self,master_array,master_dictionary,tplot=True,train_steps=None,write_data=False,output_root="train"):
        from PProbe_train import TrainingFunctions as PPtrain
        #standard training run
        #takes master array and runs indicated training steps
        #master dictionary
        trafunc = PPtrain()
        ppfilt = PPfilt()
        ppio = PPio()

        #sort by resolution, then peak id
        master_array.sort(order=['res','id'])
        selectors = PPsel(master_array)
        master_array['res'] = np.clip(master_array['res'],0.6,5.0).astype(np.float16)
        proc_array = np.zeros(master_array.shape[0],dtype=selectors.proc_array_dtype)
        for col in ['id','ori','res','ccSf','ccWf','ccS2','ccW2','ccSifi','ccSifo','ccSi2i','ccSi2o',
                    'ccSifr','ccSi2r','ccWif','ccWi2','ccSf60','sdSf60','ccS260','sdS260','vf','v2','2fofc_sigo']:
            proc_array[col] = master_array[col]

        if train_steps is None:
            train_steps = ['contacts','density','composite','kde']

        # each step generates model coefficients, then reloads a classifier object with new values

        #train contact distributions
        if 'contacts' in train_steps:
            master_dictionary['contacts'] = trafunc.contact_jsu(master_array,plot=tplot)
            ppclass = PPclass(input_dict=master_dictionary,verbose=True)
            ppclass.contact_da(master_array)

        #density features are resolution dependendent and highly correlated
        #require scaling, decorrelation, scaling, then training discriminatory
        #outputs 
        if 'density' in train_steps:
            #scaling pre_pca
            master_dictionary['respre'] =  trafunc.calculate_res_scales(proc_array,plot=tplot)  
            ppclass = PPclass(input_dict=master_dictionary,verbose=True)
            ppclass.standardize_data(proc_array,post_pca=False)
            #calculate resolution dependent PCA transformation matrix coefficients
            master_dictionary['pca'] = trafunc.calc_res_pca(proc_array,plot=tplot)
            ppclass = PPclass(input_dict=master_dictionary,verbose=True)
            #do actual PCA transformation
            ppclass.pca_xform_data(proc_array)
            #calculates resolution dependent scales for PCA transformed data
            master_dictionary['respost'] = trafunc.calculate_res_scales(proc_array,post_pca=True,plot=tplot)
            ppclass = PPclass(input_dict=master_dictionary,verbose=True)
            ppclass.standardize_data(proc_array,post_pca=True)
            #trains discriminant analysis system on scaled decorrelated features
            master_dictionary['density']=trafunc.calc_jsu_coeff(proc_array,plot=True)
            ppclass = PPclass(input_dict=master_dictionary,verbose=True)
            ppclass.density_da(proc_array,master_array)

        if 'composite' in train_steps:
            #assumes score and cscore are up to date!
            master_dictionary['composite'] = trafunc.composite_jsu_coeff(master_array,plot=tplot)
            ppclass = PPclass(input_dict=master_dictionary,verbose=True)
            #rescales score and cscore 
            ppclass.standardize_data(master_array,post_pca=True,composite=True)
            master_dictionary['chiD'] = trafunc.chiD_fit(master_array,plot=True)
            
        if 'kde' in train_steps:
            ppkde = PPKDE(master_dictionary,train=True)
            master_dictionary['kde'] = ppkde.train_kde(master_array,master_dictionary)
            if tplot:
                ppkde = PPKDE(master_dictionary,train=False)
                ppkde.plot_kde(outstr=output_root)
        

        #save data processesed with training model
        if write_data:
            np.save(output_root+"train_proc.npy",master_array)
            
        #write master dictionary with all model parameters
        ppio.write_master_dict(master_dictionary)


    def data_vs_hist(self,master_array,gen_prior=False,outstr="test"):
        ppio = PPio()
        mdict = ppio.read_master_dict()
        ppkde = PPKDE(mdict)
        d1 = master_array['score']
        d2 = master_array['cscore']
        to_plot = np.invert(master_array['mf'] % 100 > 19)
        if gen_prior:
            probin = master_array['kde'][:,2,:]
            prior,counts = ppkde.calc_pdb_prior(master_array,probin)
            labels = master_array['lab']
        else:
            labels = ppkde.kde_label(master_array)
            prior=None
        ppkde.plot_data_on_grid(d1[to_plot],d2[to_plot],labels[to_plot],prior=prior,outstr=outstr)


    def peak_report(self,all_peak_db,output_list = None,omit_models=[]):
        ppio = PPio()
        if output_list is None: #output every GD peak
            output_list = list(pdict['unal'] for pdict in all_peak_db.values() if pdict['model'] not in omit_models)
        else:
            output_list = list(unal for unal in output_list if all_peak_db[unal]['model'] not in omit_models)
        output_list = sorted(output_list,key = lambda unal: all_peak_db[unal]['resat'])
        print "-"*79
        print "-"*79
        print "OUTPUT REPORT for %d PEAKS:" % len(output_list)
        for unal in output_list:
            ppio.peak_report(all_peak_db,unal)
            print "-"*79
            
