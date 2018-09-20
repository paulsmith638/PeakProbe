import sys,os,copy,math,ast
import numpy as np
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=1e6, edgeitems=1e6)
from PProbe_selectors import Selectors
from PProbe_util import Util
from PProbe_filter import Filters
ppsel = Selectors(None)


class DataIO:
     def __init__(self,phenix_python=True):
          global Contacts
          self.alldata_input_col = ['ccSf','ccWf','ccS2','ccW2','ccSifi','ccSifo','ccSi2i','ccSi2o','ccSifr','ccSi2r','ccWif','ccWi2',
                                    'ccSf60','sdSf60','ccS260','sdS260','ori','vf','v2','charge','res','id','bin','batch','omit',
                                    'solc','fofc_sigi','2fofc_sigi','fofc_sigo','2fofc_sigo','dmove','cstr']
          self.alldata_input_formats = (np.float64, np.float64, np.float64, np.float64,
                                        np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64,
                                        np.float64, np.float64, np.float64, np.float64,'S16',np.float64,np.float64,np.float64,np.float64,'S16',
                                        np.int16,np.int16,np.bool,np.float64, np.float64, np.float64, np.float64, np.float64, np.float64,'S16')
          self.alldata_input_dtype = np.dtype(zip(self.alldata_input_col,self.alldata_input_formats))


          self.np_raw_col_names = self.alldata_input_col
          self.np_raw_col_formats = self.alldata_input_formats
          self.np_raw_dtype = self.alldata_input_dtype


          self.common_solvent = ['HOH','SO4','GOL','EDO','PO4','NAG','ACT','PEG','DMS','MPD','MES','TRS','PG4','FMT','PLP','EPE','PGE',
                                 'ACY','BME','CIT','SAH','IMD','NO3','1PE','IPA','H4B','BOG','MRD','FLC','TLA','CO3','GSH','P6G','MAN',
                                 'SCN','MLI','CAC','NDG','BGC','PLM','GLC','RET','SUC','POP','EOH','AKG','BEN','MAL','DTT','NH4','AZI',
                                 'LDA','HED','BCT','GAL','PYR','DIO','MYR','LMT','BTB','SIN','NHE','C8E','ADN','PE4','OLC','FUC','DOD',
                                 'SPM','MLA','CDL','U10','TAR','ADE','2PE','BEZ','PGO','SIA','BTN','BEF','HEZ','F6P','OGA','PAR','MLT',
                                 'IMP','OLA','TAM','SO3','PEP','IPH','CLR','PMP','NO2','HC4','ORO','OXL','CAM','CMO','BNG','BMA','B3P',
                                 'STU','PG0','FOL','CYN','VO4','MBO','CPT','CHD','PPV','MPO','DMU','ALF','PGA','SFG','H2S','CXS','15P',
                                 '12P','URE','SRT','P33','CYC','BLA','MOH','LAT','BCR','BCN','A2G','GAI']

          self.common_elem = ['BA','BR','HG','I','CO','CD','NI','CU','FE','K','MN','NA','CA','CL','MG','ZN']

          self.common_oth = ['DTT', 'MAL', 'EOH', 'SUC', 'SCN', 'P6G', 'GSH', 'CO3', 'CIT', 'BOG', 'NO3', 'IMD', 'BME', 'ACY', 
                             'PGE', 'PG4', 'TRS', 'MPD', 'DMS', 'PEG', 'ACT', 'EDO', 'GOL', 'CL',  'BR',  'AZI', 'GNP', 'BGC', 
                             'BEN', 'H4B', 'SF4', 'GLC', 'RET', '1PE', 'ACP', 'CAC', 'FLC', 'EPE', 'AKG', 'LDA', 'SAM', 'POP', 
                             'F3S', 'NAI', 'MLI', 'NDG', 'THP', 'HED', 'NH4', 'TLA', 'FES', 'HEC', 'MRD', 'UNL', 'IPA', 'PLP', 
                             'MES', 'NCO', 'PLM', 'MAN']

          self.common_met = ['MG','CA','ZN','MN','NI','CD','CO','FE']



          self.pdict_to_numpy = {"cchiS":"cchiS",
                 "cchiW":"cchiW",
                 "chiS":"chiS",
                 "chiW":"chiW",
                 "cllgS":"cllgS",
                 "cllgW":"cllgW",
                 "cscore":"cscore",
                 "llgS":"llgS",
                 "llgW":"llgW",
                 "prob":"prob",
                 "prob_data":"kde",
                 "pick":"pick",
                 "score":"score",
                 "oh":"oh",
                 "ol":"ol",
                 "om":"om",
                 "sl":"sl",
                 "sm":"sm",
                 "sp":"sp",
                 "st":"st",
                 "wl":"wl",
                 "wm":"wm",
                 "wt":"wt",
                 "2fofc_sigo_scaled":"2fofc_sigo",
                 "c1":"c1",
                 "charge":"charge",
                 "fofc_sigo_scaled":"fofc_sigo",
                 "so4_2fofc_mean_cc60":"ccS260",
                 "so4_2fofc_stdev_cc60":"sdS260",
                 "so4_cc_2fofc_inv_in":"ccSi2i",
                 "so4_cc_2fofc_inv_out":"ccSi2o",
                 "so4_cc_2fofc_inv_rev":"ccSi2r",
                 "so4_cc_2fofc_out":"ccS2",
                 "so4_cc_fofc_inv_in":"ccSifi",
                 "so4_cc_fofc_inv_out":"ccSifo",
                 "so4_cc_fofc_inv_rev":"ccSifr",
                 "so4_cc_fofc_out":"ccSf",
                 "so4_fofc_mean_cc60":"ccSf60",
                 "so4_fofc_stdev_cc60":"sdSf60",
                 "vol_2fofc":"v2",
                 "vol_fofc":"vf",
                 "wat_cc_2fofc_inv":"ccWi2",
                 "wat_cc_2fofc_out":"ccW2",
                 "wat_cc_fofc_inv":"ccWif",
                 "wat_cc_fofc_out":"ccWf",
                 "batch":"batch",
                 "bin":"bin",
                 "cc":"cc",
                 "db_id":"id",
                 "edc":"edc",
                 "fc":"fc",
                 "label":"lab",
                 "mf":"mf",
                 "mflag":"mflag",
                 "model":"model",
                 "label":"lab",                
                 "omit":"omit",
                 "orires":"ori",
                 "rc":"rc",
                 "resolution":"res",
                 "solc":"solc",
                 "status":"status",
                 "unal":"unal",
                 "dmove":"dmove",
                 "scr1":"scr1",
                 "scr2":"scr2",
                 "scr3":"scr3",
                 "tflag":"tf"}

          if phenix_python:
               from PProbe_contacts import Contacts
               self.prune_cont = Contacts.prune_cont
               self.ppcont = Contacts()

     def extract_raw(self,features_list):
          pput = Util()
          master_array=self.initialize_master(len(features_list))
          #generates a master numpy array for all calculations from list of feature dictionaries
          for index,fdict in enumerate(features_list):
               if 'fofc_sigo_scaled' not in fdict.keys():
                    fdict['fofc_sigo_scaled'] = pput.scale_density(fdict['fofc_sig_out'],fdict['solc'])
               if '2fofc_sigo_scaled' not in fdict.keys():
                    fdict['2fofc_sigo_scaled'] = pput.scale_density(fdict['2fofc_sig_out'],fdict['solc'])
               for dname,np_name in self.pdict_to_numpy.iteritems():
                    master_array[np_name][index] = fdict.get(dname,0)
          return master_array

     def update_from_np(self,all_peak_db,master_array):
          #dictionary is reversible in this case
          ppfilt = Filters(verbose=False)
          filter_mask = ppfilt.feat_filt(master_array)
          for pind,peak in enumerate(master_array):
               unal = peak['unal']
               for dname,np_name in self.pdict_to_numpy.iteritems():
                    all_peak_db[unal][dname] = peak[np_name]
               #add a filter value
               all_peak_db[unal]['filter_mask'] = list(filter_mask[pind])

     def initialize_master(self,num_peaks):
          """
          Define the master numpy array that holds all peak information for numerical analysis 
          including features, calculated values, contact lists, etc.
          FIELD       Data Type     Description
          unal        i8         info    -- unal for each input peak, completely unique 64bit int
          id          string16   info    --  id for every peak --> abcd_X_00000 abcd=pdbid or "user",X = chainid, 00000 = serial number
          ccSf        f4         feature -- RScorr for S refined against fofc
          ccWf        f4         feature -- RScorr for W refined against fofc
          ccS2        f4         feature -- RScorr for S refined against 2fofc
          ccW2        f4         feature -- RScorr for W refined against 2fofc
          ccSifi      f4         feature -- RScorr for S against pseudoinverse fofc, unmoved from coords from ccSf
          ccSifo      f4         feature -- RScorr for S against pseudoinverse fofc after refinement
          ccSi2i      f4         feature -- RScorr for S against pseudoinverse 2fofc, unmoved from coords from ccSf
          ccSi2o      f4         feature -- RScorr for S against pseudoinverse 2fofc after refinement
          ccSifr      f4         feature -- RScorr for S against normal fofc after refinement against pseudoinv fofc
          ccSi2r      f4         feature -- RScorr for S against normal 2fofc after refinement against pseudoinv 2fofc
          ccWif       f4         feature -- RScorr for W against pseudoinverse fofc
          ccWi2       f4         feature -- RScorr for W against pseudoinverse 2fofc
          ccSf60      f4         feature -- Mean RScorr for S after rotation 60degrees around each of 4 atomic axes, fofc
          sdSf60      f4         feature -- Mean StDev of ccSf60
          ccS260      f4         feature -- Mean RScorr for S after rotation 60degrees around each of 4 atomic axes, 2fofc
          sdSf60      f4         feature -- Mean StDev of ccS260
          vf          f4         feature -- Num of 0.5A map gridpoints above sol-scaled 1sigma fofc map (conv from int)
          v2          f4         feature -- Num of 0.5A map gridpoints above sol-scaled 1sigma 2fofc map (conv from int)
          charge      f4         feature -- sum of logodds ratios for atomic contacts, approximates electrostat potential
          fofc_sigo   f4         feature -- fofc sigma height of peak after refinement, solc-corrected (not used in classifier)
          2fofc_sigo  f4         feature -- 2fofc sigma height of peak after refinement, solc-corrected
          dmove       f4         feature -- ang dist moved in refinement (only for filtering)
          ol          i1         feature -- num very close contacts peak to original structure below 1.1A
          om          i1         feature -- num very close contacts peak to original structure below 1.7A
          oh          i1         feature -- num very close contacts peak to original structure below 4.5A
          wl          i1         feature -- num very close contacts ref W to ori structure below 1.1A
          wm          i1         feature -- num very close contacts ref W to ori structure below 1.7A
          wt          i1         feature -- sum wl, wm
          sl          i1         feature -- num very close contacts ref S (any atom) to ori structure below 1.1A
          sm          i1         feature -- num very close contacts ref S (any atom) to ori structure below 1.7A
          st          i1         feature -- sum sl,sm
          sp          i1         feature -- number of self contacts in peak structure (special positions)
          c1          f4         feature -- ang dist to first non-H, non W/S contact in original structure
          res         f4         general feature -- resolution of peak (same for all within one pdb)
          bin         i1         general feature -- resolution bin (1-9, roughly eq vol recip space, 1=high, 9=low)
          solc        f4         general feature -- structure solvent content, used for map variance correction
          ori         string     info  -- original residue at peak position
          batch       i2         info -- randomly assigned integer, usually 0-999 for CV/testing
          omit        bool       info -- flag to omit for training (bad r-fact, etc.)
          prob        f4         calc -- function of score and cscore, top level classifier
          score       f4         calc -- llr S vs W for electron density features
          llgS        f4         calc -- llg for S vs random for electron density feature
          llgW        f4         calc -- llg for W vs random for electron density feature
          chiS        f4         calc -- chisq stat S vs distributions for scaled electron density features
          chiW        f4         calc -- chisq stat W vs distributions for sclaed electron density features
          cscore      f4         calc -- llr S vs W for contact features
          cllgS       f4         calc -- llg for S vs random for contact feature
          cllgW       f4         calc -- llg for W vs random for contact feature
          cchiS       f4         calc -- chisq for S vs dist for contact feature
          cchiW       f4         calc -- chisq for W vs dist for contact feature
          kde         3x4f4      calc -- scores from kde
          pick        i2         calc -- current pick from prob
          fc          i1         info -- flag class (special, weak, bad contacts, etc.)
          edc         i1         info -- electron density class (quality of fit to ED calc values)
          cc          i1         info -- contact class (quality of fit to contact calc values)
          mf          i4         info -- peak cluster class (multiple peaks for one solvent molecule, etc.)
          rc          i1         info -- results class (S/W assignment and how many stats below cutoffs)
          status      i4         info -- status int 
          tflag       i4         info -- temporary flag
          scr1        f4         scr  -- column for storing/pasing intermediate results
          scr2        f4         scr  -- column for storing/pasing intermediate results
          scr3        f4         scr  -- column for storing/pasing intermediate results

          """

          self.master_cols = ["cchiS","cchiW","chiS","chiW","cllgS","cllgW","cscore","llgS","llgW","prob","kde","pick","score","oh","ol","om","sl","sm","sp","st","wl","wm","wt","2fofc_sigo","c1","charge","fofc_sigo","ccS260","sdS260","ccSi2i","ccSi2o","ccSi2r","ccS2","ccSifi","ccSifo","ccSifr","ccSf","ccSf60","sdSf60","v2","vf","ccWi2","ccW2","ccWif","ccWf","batch","bin","cc","id","edc","fc","lab","mf","mflag","model","omit","ori","rc","res","solc","status","unal","dmove","scr1","scr2","scr3","tf"]


          self.master_fmts = ["f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","(3,4)f4","i2","f4","i1","i1","i1","i1","i1","i1","i1","i1","i1","i1","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","i2","i1","i1","S16","i1","i1","i1","i4","i1","i1","?","S3","i1","f4","f4","i4","i8","f4","f4","f4","f4","i4"]



          self.master_dtype = np.dtype(zip(self.master_cols,self.master_fmts))

          #initialize array all zeros and return
          master_data=np.zeros(num_peaks,dtype=self.master_dtype)
          return master_data

     

     def read_master_dict(self,input_dfile=None):
#          try:
          if input_dfile is None:
               dfile = open("pprobe_master.dict",'r')
               dstring = dfile.read().strip()
               self.master_dict = ast.literal_eval(dstring)
               dfile.close()
          else:
               dfile = open(input_dfile,'r')
               self.master_dict = ast.literal_eval(dfile.read())
               dfile.close()
#          except:
#              sys.exit("PROBLEM WITH MASTER PARAM FILE: %s -- TERMINATING!" % dfile)
          return self.master_dict

     def write_master_dict(self,master_dict,output_dfile=None):
          
          if output_dfile is not None:
               outfile = output_dfile
          else:
               outfile = "pprobe_master.dict"
          f = open(outfile,'w')
          f.write(str(master_dict))
          f.close()



     def initial_report(self,peak_features):
          #for peak data output during feature extraction
          clash = peak_features['clash']
          closest_peak = peak_features['peak_contacts'][0]
          peak_3A = self.prune_cont(peak_features['peak_contacts'],cutoff=3.0,omit_null=True)
          closest_solvent = peak_features['sol_contacts'][0]
          closest_mac = peak_features['mm_contacts'][0]
          #local peaks / clusters?
          if len(peak_3A) == 0:
               cl_peak_out = "Isolated Peak"
          elif len(peak_3A) > 2:
               cl_peak_out = "Cluster of %2d" % len(peak_3A)
          elif len(peak_3A) ==  1:
               cl_peak_out = " 1 Close peak"
          else:
               cl_peak_out = "Near %2d peaks" % len(peak_3A)
          if "self" in list(cont['ctype'] for cont in peak_3A):
               cl_peak_out = "%13s" % "Special Pos?"

          #closest solvent contact
          if closest_solvent['distance'] > 2.5:
               cl_sol_out = "%-20s" % "NONE"
          else:
               cl_sol_out = "%2.1fA from %-10s" % (closest_solvent['distance'],closest_solvent['resat'])

          #closest non-solvent contact
          if closest_mac['distance'] > 4.5:
               cl_mod_out = "%-22s" % "far from model"
          elif clash:
               cl_mod_out = "%2.1fA CLASH %-11s" % (closest_mac['distance'],closest_mac['resat'][0:11])
          else:
               cl_mod_out = "%2.1fA from %-12s" % (closest_mac['distance'],closest_mac['resat'])

          #oneliner output
          outstr = "PEAK %s FoFc: %5.2f 2FoFc: %5.2f Contacts: %s %s %s" % (peak_features['db_id'],
                                                                              peak_features['fofc_sig_out'],
                                                                              peak_features['2fofc_sig_out'],
                                                                              cl_peak_out,cl_mod_out,cl_sol_out)
          return outstr

