import sys,os,copy,math,ast
import numpy as np
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=1e6, edgeitems=1e6)
from PProbe_selectors import Selectors
from PProbe_util import Util
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

          self.common_elem = [' BA',' BR',' HG','  I',' CO',' CD',' NI',' CU',' FE','  K',' MN',' NA',' CA',' CL',' MG',' ZN']
          if phenix_python:
               from PProbe_contacts import Contacts
               self.prune_cont = Contacts.prune_cont
               self.ppcont = Contacts()

     def extract_raw(self,features_list):
          pput = Util()
          master_array=self.initialize_master(len(features_list))
          #translates items from feature dictionary to master array column type
          f2m = {'so4_cc_fofc_out':'ccSf',
                 'so4_cc_2fofc_out':'ccS2',
                 'wat_cc_fofc_out':'ccWf',
                 'wat_cc_2fofc_out':'ccW2',
                 'so4_cc_fofc_inv_in':'ccSifi',
                 'so4_cc_fofc_inv_out':'ccSifo',
                 'so4_cc_fofc_inv_rev':'ccSifr',
                 'so4_cc_2fofc_inv_in':'ccSi2i',
                 'so4_cc_2fofc_inv_out':'ccSi2o',
                 'so4_cc_2fofc_inv_rev':'ccSi2r',
                 'wat_cc_fofc_inv':'ccWif',
                 'wat_cc_2fofc_inv':'ccWi2',
                 'so4_fofc_mean_cc60':'ccSf60',
                 'so4_fofc_stdev_cc60':'sdSf60',
                 'so4_2fofc_mean_cc60':'ccS260',
                 'so4_2fofc_stdev_cc60':'sdS260',
                 'fofc_sigo_scaled':'fofc_sigo',
                 '2fofc_sigo_scaled':'2fofc_sigo',
                 'charge':'charge',
                 'db_id':'id',
                 'ol':'ol',
                 'om':'om',
                 'oh':'oh',
                 'wl':'wl',
                 'wm':'wm',
                 'sl':'sl',
                 'sm':'sm',
                 'sp':'sp',
                 'c1':'c1',
                 'pc1d':'pc1d',
                 'pc1id':'pc1id',
                 'pc2d':'pc2d',
                 'pc2id':'pc2id',
                 'pc3d':'pc3d',
                 'pc3id':'pc3id',
                 'solc':'solc',
                 'orires':'ori',
                 'bin':'bin',
                 'resolution':'res',
                 'omit':'omit',
                 'vol_fofc':'vf',
                 'vol_2fofc':'v2',
                 'dmove':'dmove',
                 'unal':'unal'}
          for index,fdict in enumerate(features_list):
               if 'fofc_sigo_scaled' not in fdict.keys():
                    fdict['fofc_sigo_scaled'] = pput.scale_density(fdict['fofc_sig_out'],fdict['solc'])
               if '2fofc_sigo_scaled' not in fdict.keys():
                    fdict['2fofc_sigo_scaled'] = pput.scale_density(fdict['2fofc_sig_out'],fdict['solc'])
               for column in f2m.keys():
                    master_array[f2m[column]][index] = fdict[column]
          return master_array


     def initialize_master(self,num_peaks):
          """
          Define the master numpy array that holds all peak information for numerical analysis 
          including features, calculated values, contact lists, etc.
          FIELD       Data Type     Description
          unal        i8         desc    -- unal for each input peak, completely unique 64bit int
          id          string     desc    --  id for every peak --> abcd_X_00000 abcd=pdbid or "user",X = chainid, 00000 = serial number
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
          v2          f4         feature -- Num of 0.5A map gridpoints above sol-scaled 1sigma 2fofc map (conf from int)
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
          pc1id       string     feature -- id of closest other peak (cluster analysis)
          pc1d        f4         feature -- dist between self and pc1id
          pc2id       string     feature -- id of 2nd closest other peak (cluster analysis)
          pc2d        f4         feature -- dist between self and pc2id
          pc3id       string     feature -- id of 3rd closest other peak (cluster analysis)
          pc3d        f4         feature -- dist between self and pc3id
          presc       string     feature -- identifier string for closest structural contact
          presd       f4         feature -- dist from presc to peak

          res         f4         general feature -- resolution of peak (same for all within one pdb)
          bin         i1         general feature -- resolution bin (1-9, roughly eq vol recip space, 1=high, 9=low)
          solc        f4         general feature -- structure solvent content, used for map variance correction

          ori         string     label -- original residue at peak position
          batch       i2         label -- randomly assigned integer, usually 0-999 for CV/testing
          omit        bool       label -- flag to omit for training (bad r-fact, etc.)

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
          
          fc          i1         calc -- flag class (special, weak, bad contacts, etc.)
          edc         i1         calc -- electron density class (quality of fit to ED calc values)
          cc          i1         calc -- contact class (quality of fit to contact calc values)
          mf          i1         calc -- peak cluster class (multiple peaks for one solvent molecule, etc.)
          rc          i1         calc -- results class (S/W assignment and how many stats below cutoffs)

          scr1        f4         scr -- column for storing/pasing intermediate results
          scr2        f4         scr -- column for storing/pasing intermediate results
          scr3        f4         scr -- column for storing/pasing intermediate results

          """

          self.master_cols = ["unal","id","ccSf","ccWf","ccS2","ccW2","ccSifi","ccSifo","ccSi2i",
                              "ccSi2o","ccSifr","ccSi2r","ccWif","ccWi2","ccSf60","sdSf60","ccS260",
                              "sdS260","vf","v2","charge","fofc_sigo","2fofc_sigo","dmove","ol",
                              "om","oh","wl","wm","wt","sl","sm","st","sp",
                              "c1","pc1id","pc1d","pc2id","pc2d","pc3id","pc3d",
                              "res","bin","solc","ori","batch","omit","prob",
                              "score","llgS","llgW","chiS","chiW","cscore","cllgS","cllgW",
                              "cchiS","cchiW","fc","edc","cc","mf","rc","scr1",
                              "scr2","scr3"]

          self.master_fmts = ['i8','S16','f4','f4','f4','f4','f4','f4','f4',
                              'f4','f4','f4','f4','f4','f4','f4','f4',
                              'f4','f4','f4','f4','f4','f4','f4','i1',
                              'i1','i1','i1','i1','i1','i1','i1','i1','i1',
                              'f4','S16','f4','S16','f4','S16','f4',
                              'f2','i1','f2','S3','i2','?','f4',
                              'f4','f4','f4','f4','f4','f4','f4','f4',
                              'f4','f4','i1','i1','i1','i1','i1','f4',
                              'f4','f4']
          self.master_dtype = np.dtype(zip(self.master_cols,self.master_fmts))

          #initialize array all zeros and return
          master_data=np.zeros(num_peaks,dtype=self.master_dtype)
          return master_data


     def store_master(self,master_array,filename):
          self.master_csv_format = ['%12s','%8g','%8g','%8g','%8g','%8g','%8g','%8g',
                                    '%8g','%8g','%8g','%8g','%8g','%8g','%8g','%8g',
                                    '%8g','%8g','%8g','%8g','%8g','%8g','%8g','%2g',
                                    '%2g','%2g','%2g','%2g','%2g','%2g','%2g','%2g','%2g',
                                    '%4g','%12s','%4g','%12s','%4g','%12s','%4g','%10s',
                                    '%4g','%4g','%1d','%4g','%3s','%3d','%1d','%4g',
                                    '%8g','%8g','%8g','%8g','%8g','%8g','%8g','%8g',
                                    '%8g','%8g','%1d','%1d','%1d','%1d','%1d','%8g',
                                    '%8g','%8g']      
          print "STORING ALL DATA FOR %s PEAKS TO CSV FILE %s" % (master_array.shape[0],filename)
          np.savetxt(filename,master_array,fmt=",".join(self.master_csv_format))

     def read_master(self,filename):
          master_array = np.loadtxt(filename,delimiter=',',dtype=self.master_dtype)
          return master_array


     

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



     def read_features_csv(self,filename):
          data_array = np.loadtxt(filename,delimiter=',',dtype=self.np_raw_dtype)
          print "READ FEATURES FOR %s PEAKS FROM FILE %s" % (data_array.shape[0],filename)
          return data_array



     def model_rep(self,pdict):
          pdata=pdict['proc_data']
          mplist = pdict['mod_cont'] #actually a contact list
          mflag = pdict['mflag']
          report = ["MC=%s" % mflag,]
          if mflag in [0,4]:
               report.append("Unknown Solvent Model")
          if mflag == 1:
               report.append("Unmodeled")
          if mflag == 2:
               mp = mplist[0]
               report.append("Modeled as %s (%2.1fA away)" % (mp['resat'],mp['distance']))
          if mflag == 3:
               mp = mplist[0]
               report.append("Possibly modeled as %s, but %2.1fA away" % (mp['resat'],mp['distance']))
          if mflag == 5:
               report.append("Split model [%s]for single peak" % " ".join(mp['resat'] for mp in mplist))
          if mflag == 6:
               report.append("Shared model for multiple peaks [%s]" % " ".join(mp['resat'] for mp in mplist))
          if mflag == 7:
               report.append("Ambiguous model for multiple peaks [%s]" % " ".join(mp['resat'] for mp in mplist))
          return report

     def contact_rep(self,pdict):
          report = ["peak %s has status %s" % (pdict['db_id'],pdict['status']),]
          return report
          if fc == 1: #special
               pdict['warnings'].append("SPL")
               noself_med_cont = self.prune_cont(pdict['w_contacts'],omit_unrg=[pdict['unrg'],],omit_null=True,cutoff=2.5)
               noself_bad_cont = self.prune_cont(pdict['w_contacts'],omit_unrg=[pdict['unrg'],],omit_null=True,cutoff=1.7)
               report.append("At/Near Special -- Med: %s Bad: %s" % (len(noself_med_cont),len(noself_bad_cont)))

          if fc > 1 and fc < 6:
               close_cont = self.close_cont(pdict['mm_contacts'])
               if len(close_cont) > 0:
                    n_worst = 0
                    worst = close_cont[0]
                    for clist in close_cont:
                         n_close = len(clist)
                         if n_close > n_worst:
                              worst = clist
                         resname = clist[0]['resname']
                         chain = clist[0]['chain']
                         resid = clist[0]['resid']
                         cid = resname+"_"+chain+str(resid)
                         report.append("%s close contacts to %s" % (n_close,cid))
                    worst.sort(key = lambda x: x['distance'])
                    shortest_worst = worst[0]
               else:
                    shortest_worst = pdict['mm_contacts'][0]


          if fc == 2: #bad clashes
               pdict['warnings'].append("CONT1")
               #most model errors have ed score > 0:
               if pdata['score'] > 0 or pdata['cscore'] > 0:
                    if shortest_worst['name'] not in ['N','C','CA','O']:
                         report.append("--> ALT/ROT error at %s?" % shortest_worst['resat'])
                    else:
                         report.append("--> BACKBONE error/alt at %s?" % shortest_worst['resat'])
               else:
                    report.append("--> Spurious/Noise Peak?")

          if fc == 3 or fc == 4: #less bad contacts
               pdict['warnings'].append("CONT2")
               probs = pdict['prob_data']
               if probs[2][3] > 0.5: #metal kde/dir prob is > 50%
                    if pdata['edc'] < 7 and pdict['fofc_sigo'] > 2.0:
                         report.append("Likely Metal Ion")
                    else:
                         report.append("Likely Model Error at %s, possibly Metal Ion" % shortest_worst['resat'])
               elif pdata['score'] > 0:
                    if shortest_worst['name'] not in ['N','C','CA','O']:
                         report.append("--> ALT/ROT error at %s?" % shortest_worst['resat'])
                    else:
                         report.append("--> BACKBONE error/alt at %s?" % shortest_worst['resat'])
               else:
                    report.append("--> Spurious/Noise Peak?")

          if fc == 5: #one close contact
               pdict['warnings'].append("CONT3")
               probs = pdict['prob_data']
               if probs[2][3] > 0.5: 
                    if pdata['edc'] < 7:
                         report.append("Possible Metal Ion")
                    else:
                         report.append("Likely Model Error at %s" % shortest_worst['resat'] )
               elif probs[2][0] > 0.5:
                    if pdata['edc'] > 6:
                         report.append("--> Close Water")
                    else:
                         report.append("Possible Close Water")
               else:
                    report.append("Ambiguous, Inspect Manually?")
          if fc == 6: #weak
               pdict['warnings'].append("WEAK")
               report.append("Weak Density, likely noise peak")
          if fc == 7: #remote
               pdict['warnings'].append("REMOTE")
               if pdict['anchor']['model'] == 3:
                    report.append("REMOTE, connected to MM by %s" % pdict['anchor']['resat'])
               else:
                    report.append("REMOTE, no connection to MM")

          if edc == 0:
               report.append("Junk/Noise")
          return report

     def clust_rep(self,pdict):
          pdata=pdict['proc_data']
          pmf = pdata['mf']
          rep_str = []
          if pmf == 1:
               rep_str.append("None")
          if pmf == 2:
               rep_str.append("Large Cluster --> unmodelled ligand?")
               rep_str.append("List of associated peaks:")
               rep_str.append("   "+" ".join(pdict['sat_peaks']))
          if pmf == 3 or pmf == 4:
               rep_str.append("Cluster --> principal peak")
               rep_str.append("List of satellite peaks:")
               rep_str.append("   "+" ".join(pdict['sat_peaks']))
          if pmf == 5:
               rep_str.append("Satellite Peak:")
               rep_str.append("Possible associations:")
               rep_str.append("   "+" ".join(pdict['sat_peaks']))

          return rep_str

     def score_report(self,pdict,resid_names):
          preds = pdict['pred_data']
          probs = pdict['prob_data']
          pdata = pdict['proc_data']
          ambig = pdict['ambig']
          pick1 = pdict['pick']
          pick_name = pdict['pick_name']
          peakid = pdict['db_id']
          score_rep = {}
          lstr_outputs = []
          sstr_outputs = []
          p1 = " ".join(" %3.1f" % x for x in probs[0])
          p2 = " ".join(" %3.1f" % x for x in probs[1])
          p3 = " ".join(" %3.1f" % x for x in probs[2])
          lstr_outputs.append("Known_Classes:          %s" % "  ".join(resid_names))
          lstr_outputs.append("Class_prob_flat_prior   %s" % p1 )
          lstr_outputs.append("Class_prob_bias_prior   %s" % p2 )
          lstr_outputs.append("Class_prob_popc_prior   %s" % p3 )
          lstr_outputs.append("Peak Predicted to be: %s Ambig: %s" % (resid_names[pick1-1],ambig))
          sstr_outputs.append("P1: %s |P2: %s |P3: %s " % (p1,p2,p3))
          pdict['score_lstr'] = lstr_outputs
          pdict['score_sstr'] = sstr_outputs

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

     def peak_report(self,pdict):
          all_peak_db = pdict['peak_unal_db']
          if pdict['status'] in [6,]:
               print "PEAK %s Status: %s UNPROCESSED " % (pdict['db_id'],pdict['status'])
               return
          resid_names = pdict['resid_names']
          peakid = pdict['db_id']
          unal = pdict['unal']
          #if 'pred_data' not in pdict.keys() or pdict['status'] in [1,3,6,7]:
          #     return " %s NOPROC" % pdict['db_id']
          preds = pdict['pred_data']
          probs = pdict['prob_data']
          pdata = pdict['proc_data']
          #self.model_error(peakid,peak_db)
          self.score_report(pdict,resid_names)
          ambig = pdict['ambig']
          pick1 = pdict['pick']
          pick_name = pdict['pick_name']
          mflag = pdict['mflag']
          label = pdict['label']

          to_output = {"pscore":False,"mscore":False,'p_as_lab':False,"m_as_lab":False,
                       "clust_rep":False,"contact_rep":True,"model_rep":True}

          #first, status -- process error or sigma rejected
          if pdict['status'] not in [1,3,6,7]:
               to_output['pscore'] = True

          #next, look at contact situation:
          contact_report = self.contact_rep(pdict)
          if len(contact_report) == 0:
               to_output['contact_rep'] = False
          if pdata['fc'] == 2:
               to_output['model_rep'] = False

          #modeled solvent report
          model_report = self.model_rep(pdict)


          if mflag in [2,3,5,6,7]:
               to_output['p_as_lab'] = True
               probs_peak_as_label = " ".join("%3.2f" % x for x in probs[:,label-1])

          if pdata['mf'] in [2,3,4,5]:
               to_output['clust_rep'] = True
               clust_rep = self.clust_rep(pdict)
          
          #look at model peak
          if mflag in [2,3,5,6,7]:
               best_sol = all_peak_db[pdict['sol_mod'][0][0]]
               mpeak_legit = best_sol['status'] not in [1,3,4,6,7]
               if mpeak_legit:
                    self.score_report(best_sol,resid_names)
                    m_probs = best_sol['prob_data']
                    m_ambig = best_sol['ambig']
                    m_pick1 = best_sol['pick']
                    m_pick_name = best_sol['pick_name']
                    m_mflag = best_sol['mflag']
                    m_label = best_sol['label']
                    if m_label > 0:
                         to_output['m_as_lab'] = True
                         probs_model_as_label = " ".join("%3.2f" % x for x in m_probs[:,m_label-1])
                    to_output['mscore'] = True
               else:
                    best_sol = {}
                    best_sol['pick'] = 0
                    m_probs = np.zeros((3,4))
                    m_ambig = False
                    m_pick1 = 0
                    m_pick_name = "XXX"
                    probs_model_as_label = ""


          #DATA FOR OUTPUT
          fsig =  pdata['fofc_sigo']
          f2sig = pdata['2fofc_sigo']
          scr =  pdata['score']
          cscr = pdata['cscore']
          edc = pdata['edc']
          cc = pdata['cc']
          mf = pdata['mf']
          srchi = (pdata['chiS'] + pdata['cchiS'])/15.0
          wrchi = (pdata['chiW'] + pdata['cchiW'])/15.0
          peak_status = pdict['status']
          prob = pdata['prob']
          cdist = pdict['c1']
          ori = pdict['orires']
          p1 = " ".join(" %3.1f" % x for x in probs[0])
          p2 = " ".join(" %3.1f" % x for x in probs[1])
          p3 = " ".join(" %3.1f" % x for x in probs[2])
          status = pdict['status']
          anc_resat = pdict['anchor']['resat']
          adist = pdict['anchor']['distance']
          #contacts
          #original model
          clori = pdict['contacts'][0]
          #existing solvent
          clsol = pdict['sol_contacts'][0]
          #macromolecule
          clmac = pdict['mm_contacts'][0]
          #other peaks, 0 is self?
          clpeak = pdict['peak_contacts'][0]
          if clpeak['resname'] == "NUL":
               pcout = "None"
          else:
               pcout = "%2.1fA" % clpeak['distance']

          op = "NONE"

          if pdata['fc'] == 1:
               peak_spl = "Yes"
               pdict['warnings'].append("Spl_Pos")
               op = op+" (SPECIAL POSITION!)"
          elif pdata['mf'] == 6:
               peak_spl = "Maybe"
               pdict['warnings'].append("Near_Spl_Pos")
               op = op+" (NEAR SPECIAL POSITION!)"
          else:
               peak_spl = "No"


          if pdata['mf'] == 1:
               peak_clust = "No"
          elif pdata['mf'] == 6:
               peak_clust = "Unknown"
          else:
               peak_clust = "Yes"
          clust_sat_short = []
          for peak in pdict['sat_peaks']:
               if peak != peakid:
                    clust_sat_short.append(peak)
          num_sat = len(clust_sat_short)


          cls = "".join("%s" % x for x in [pdata['fc'],pdata['rc'],pdata['edc'],pdata['cc'],pdata['mf']])
          tally = "".join("%s" % x for x in [pdata['ol'],pdata['om'],pdata['oh'],pdata['wl'],pdata['wm'],pdata['wt'],
                                             pdata['sl'],pdata['sm'],pdata['st'],pdata['sp']])
          allflag = self.report_flags(pdata,resid_names[pick1-1],ambig)

          if len(pdict['warnings']) > 1:
               all_warn = "Warnings: %s" % " ".join(pdict['warnings'])
          else:
               all_warn = ""
          print "PEAK %s Status: %s Flags: %s %s %s %s" % (peakid,pdict['status'],cls,all_warn,str(pdata['prob'])[0:5],str(pdict['c1'])[0:5])
          print "   ED:  FoFc %5.2f 2FoFc %5.2f ED_score %4.1f ED_class %s" % (fsig,f2sig,scr,edc)
          print "   ENV: Contact_score %4.1f Contact_class %s" % (cscr,cc)
          print "        CONTACTS Closest: %2.1fA %s  MacMol: %2.1fA %s Other_Peaks: %s" % (clori['distance'],clori['resat'],clmac['distance'],clmac['resat'],pcout)
          print "        -- Clash: %s   Special: %s   Cluster: %s  FC: %s" % (pdata['wt'],peak_spl,peak_clust,pdata['fc'])
          if to_output['contact_rep']:
               print self.rep_print(contact_report,12)

          print "   PEAK ANALYSIS:"
          if to_output['model_rep']:
               print self.rep_print(model_report,12)
          if to_output['pscore']:
               print "      Scoring:"
               print "         Peak %s from PeakProbe:" % pdict['db_id']
               print self.rep_print(pdict['score_lstr'],12)
               if to_output['p_as_lab']:
                    print "               Probs for %s as %s: %s"% (peakid,resid_names[label-1],probs_peak_as_label)
          if to_output['mscore']:
               print "         Peak %s from Input Model (%2.1fA away):" % (clsol['resat'],clsol['distance'])
               print self.rep_print(best_sol['score_lstr'],12)
          if to_output['m_as_lab']:
               print "               Probs for %s as %s: %s" % (clsol['resat'],resid_names[m_label-1],probs_model_as_label)
          if to_output['clust_rep']:
               print "CLUSTER REPORT mf %d" % pdata['mf']
               print self.rep_print(clust_rep,12)
          print "   VERDICT: %s" % op

          flags = self.report_flags(pdata,pick_name,ambig=ambig)
          flagstr = " ".join(flags)
          fmtstr = ('{:>12} SCR {:5.1f} {:5.1f} P_nw {:3.2f} C1 {:3.2f}A {:>12} C{:5} STAT {:4d} || M/P {:>3} {:>3} ||'
                    'P1: {} |P2: {} |P3: {} || {}')
          
          outstr = fmtstr.format(peakid,scr,cscr,prob,adist,anc_resat,cls,status,ori,pick_name,p1,p2,p3,flagstr)
          print "SHORT",outstr
          return "DONE"



     def rep_print(self,str_list,indent):
          outst_list = []
          for string in str_list:
               outst_list.append(" "*indent+string)
          return "\n".join(outst_list)


     def report_flags(self,peak,pred,ambig=False):
          flags = []
          if peak['fc'] == 1:
               flags.append('SPL')
          if peak['fc'] == 2 or peak['fc'] == 3:
               flags.append('CLASH')
          if peak['fc'] == 3:
               flags.append('CLASH')
          if peak['fc'] == 4:
               flags.append('BUMP')
          if peak['fc'] == 5:
               flags.append('BUMP')
          if peak['fc'] == 6:
               flags.append('WEAK')
          if peak['fc'] == 7:
               flags.append('REMOTE')
          if peak['mf'] > 1 and peak['mf'] < 6:
               flags.append('CLUST')
          #if peak['mf'] == 6:
          #     flags.append('SELF')
          if pred == "HOH":
               if peak['edc'] < 5 or peak['cc'] < 5:
                    flags.append('BADW')
               if peak['rc'] < 7:
                    flags.append('CHIW')
          if pred == "SO4":
               if peak['edc'] > 4 or peak['cc'] > 4:
                    flags.append('BADS')
               if peak['rc'] > 3:
                    flags.append('CHIS')
          if ambig:
               flags.append('AMBIG')
          return flags
