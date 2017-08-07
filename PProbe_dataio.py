import sys,os,copy,math,ast
import numpy as np
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=1e6, edgeitems=1e6)
import sqlite3 as lite
from PProbe_selectors import Selectors
ppsel = Selectors(None)


class DataIO:
     def __init__(self):
          self.sql_sel_str = "so4_cc_fofc_out,wat_cc_fofc_out,so4_cc_2fofc_out,wat_cc_2fofc_out,"+\
                             "so4_cc_fofc_inv_in,so4_cc_fofc_inv_out,so4_cc_2fofc_inv_in,so4_cc_2fofc_inv_out,"+\
                             "so4_cc_fofc_inv_rev,so4_cc_2fofc_inv_rev,wat_cc_fofc_inv,wat_cc_2fofc_inv,"+\
                             "so4_fofc_mean_cc60,so4_fofc_stdev_cc60,so4_2fofc_mean_cc60,so4_2fofc_stdev_cc60,"+\
                             "orires,vol_fofc,vol_2fofc,charge,resolution,db_id,bin,batch,omit,"+\
                             "solc, fofc_sig_in, twofofc_sig_in, fofc_sig_out, twofofc_sig_out,dmove,cstr"
          self.np_raw_col_names = ppsel.alldata_input_col
          self.np_raw_col_formats = ppsel.alldata_input_formats
          self.np_raw_dtype = ppsel.alldata_input_dtype
          self.csv_format = ppsel.features_csv_format

          self.results_csv_format = ['%12s','%4g','%8g','%8g','%8g','%8g','%8g','%8g','%8g','%8g','%2g']
          self.results_dtype = [('id','S16'),('res','f4'),('score','f4'),('prob','f4'),
                                ('llgS','f4'),('llgW','f4'),('chiS','f4'),('chiW','f4'),
                                ('fchi','f4'),('kchi','f4'),('rc','i1')]

          #contact feature structured array
          self.cfeat_col =    ['id','ori','c1','c2','c3','ol','om','sl','sm','wl','wm','ot','st','wt',
                               'psfp','weak','special','remote','sadc','badc','close']
          self.cfeat_format = ['S16','S16',np.float32,np.float32,np.float32,
                               np.int16,np.int16,np.int16,np.int16,np.int16,np.int16,np.int16,np.int16,np.int16,
                               np.float32,np.bool_,np.bool_,np.bool_,np.bool_,np.bool_,np.bool_]
          self.cfeat_dtype = np.dtype(zip(self.cfeat_col,self.cfeat_format))


     #reads sql database with all columns labeled from PProbe output
     #returns a numpy structured array
     def read_sql(self,filename):
          print "OPENING DATABASE", filename
          con = lite.connect(filename)
          with con:
               con.row_factory = lite.Row
               cur = con.cursor()
               print "EXTRACTING COLUMNS"
               sql = "SELECT %s FROM Peaks" % self.sql_sel_str
               iterator = cur.execute(sql)
               data_array = np.fromiter((tuple(row) for row in cur),dtype=self.np_raw_dtype)
          con.close()
          print "ROWS IN:",data_array.shape[0]
          return data_array

     def extract_cfeat(self,features_list):
          cfeat_array = []
          for features in features_list:
               raw_cf = features['contact_feat']
               raw_fl = features['peak_flags']
               data = tuple([features['db_id'],features['orires'],raw_cf[0],raw_cf[1],raw_cf[2],
                             int(raw_cf[3]),int(raw_cf[4]),int(raw_cf[5]),int(raw_cf[6]),int(raw_cf[7]),
                             int(raw_cf[8]),int(raw_cf[9]),int(raw_cf[10]),int(raw_cf[11]),
                             features['fp_prob'],raw_fl['weak'],raw_fl['special'],raw_fl['remote'],raw_fl['sadc'],
                             raw_fl['badc'],raw_fl['close']])
               cfeat_array.append(data)
          output_array = np.fromiter((row for row in cfeat_array),dtype=self.cfeat_dtype)
          return output_array


     def extract_raw(self,features_list):
          raw_array = []
          for features in features_list:
               data = tuple([features['so4_cc_fofc_out'],features['wat_cc_fofc_out'], 
                             features['so4_cc_2fofc_out'], features['wat_cc_2fofc_out'], 
                             features['so4_cc_fofc_inv_in'], features['so4_cc_fofc_inv_out'], 
                             features['so4_cc_2fofc_inv_in'],features['so4_cc_2fofc_inv_out'],  
                             features['so4_cc_fofc_inv_rev'],features['so4_cc_2fofc_inv_rev'],  
                             features['wat_cc_fofc_inv'],  features['wat_cc_2fofc_inv'], 
                             features['so4_fofc_mean_cc60'],features['so4_fofc_stdev_cc60'],
                             features['so4_2fofc_mean_cc60'],features['so4_2fofc_stdev_cc60'],
                             features['orires'],features['vol_fofc'],features['vol_2fofc'],features['charge'],
                             features['resolution'],features['db_id'],features['bin'],np.random.randint(0,1000),features['omit'],
                             features['solc'],features['fofc_sig_in'],features['2fofc_sig_in'],
                             features['fofc_sig_out'],features['2fofc_sig_out'],
                             features['dmove'],features['cstr']])
               raw_array.append(data)
          data_array = np.fromiter((row for row in raw_array),dtype=self.np_raw_dtype) 
          return data_array

     def store_contacts(self,features):
          #run on every individual peak?
          db_id = features['db_id']
          contacts = features['contacts']
          strip_contacts = features['strip_contacts']
          s_contacts = features['s_contacts']
          w_contacts = features['w_contacts']
          p_contacts = features['peak_contacts']
          f = open(db_id+"_contacts.list",'w')
          for c_batch,c_list in  zip(("ORI","STR","SO4","WAT","PKI"),(contacts,strip_contacts,s_contacts,w_contacts,p_contacts)):
               for contact in c_list: #list of dictionaries
                    #simplify output for lookup later (huge amount of data)
                    dist='{:3.2f}'.format(contact['distance'])
                    resat = contact['resname']+"-"+contact['name']
                    chres = contact['chain']+contact['resid']
                    outstr = ','.join((db_id,c_batch,resat,chres,dist))
                    print >> f,outstr
          f.close()


     def store_features_csv(self,np_feature_array,filename):
          print "STORING FEATURES FOR %s PEAKS TO CSV FILE %s" % (np_feature_array.shape[0],filename)
          np.savetxt(filename,np_feature_array,fmt=",".join(self.csv_format))

     def read_features_csv(self,filename):
          data_array = np.loadtxt(filename,delimiter=',',dtype=self.np_raw_dtype)
          print "READ FEATURES FOR %s PEAKS FROM FILE %s" % (data_array.shape[0],filename)
          return data_array


     def store_features_sql(self,features,filename):
          if not "NaN" in features.itervalues():
               data = np.array((features['chainid'],features['resid'],
                                features['so4_cc_fofc_in'],features['so4_cc_fofc_out'], 
                                features['wat_cc_fofc_in'],features['wat_cc_fofc_out'], 
                                features['so4_cc_2fofc_in'],features['so4_cc_2fofc_out'], 
                                features['wat_cc_2fofc_in'],features['wat_cc_2fofc_out'], 
                                features['so4_cc_fofc_inv_in'],   features['so4_cc_fofc_inv_out'], 
                                features['so4_cc_2fofc_inv_in'],  features['so4_cc_2fofc_inv_out'],  
                                features['so4_cc_fofc_inv_rev'],  features['so4_cc_2fofc_inv_rev'],  
                                features['wat_cc_fofc_inv'],  features['wat_cc_2fofc_inv'], 
                                features['so4_fofc_mean_cc60'],features['so4_fofc_stdev_cc60'],
                                features['so4_2fofc_mean_cc60'],features['so4_2fofc_stdev_cc60'],
                                features['orires'],features['vol_fofc'],features['vol_2fofc'],features['charge'],
                                features['resolution'],features['db_id'],features['bin'],"0",features['omit'],
                                features['solc'],features['fofc_sig_in'],features['2fofc_sig_in'],
                                features['fofc_sig_out'],features['2fofc_sig_out'],
                                features['dmove'],features['cstr']),dtype=np.str_)
                  #batch is currently set to 0, used elsewhere for cross_validation, etc.

          con = lite.connect(filename)
          with con:
               cur = con.cursor()
               cur.execute("CREATE TABLE IF NOT EXISTS Peaks(chainid TEXT, resid TEXT,"+ 
                           "so4_cc_fofc_in REAL, so4_cc_fofc_out REAL,"+
                           "wat_cc_fofc_in REAL, wat_cc_fofc_out REAL,"+ 
                           "so4_cc_2fofc_in REAL, so4_cc_2fofc_out REAL,"+ 
                           "wat_cc_2fofc_in REAL, wat_cc_2fofc_out REAL,"+ 
                           "so4_cc_fofc_inv_in REAL, so4_cc_fofc_inv_out REAL,"+
                           "so4_cc_2fofc_inv_in REAL, so4_cc_2fofc_inv_out REAL,"+
                           "so4_cc_fofc_inv_rev REAL, so4_cc_2fofc_inv_rev REAL,"+
                           "wat_cc_fofc_inv REAL, wat_cc_2fofc_inv REAL,"+
                           "so4_fofc_mean_cc60 REAL, so4_fofc_stdev_cc60 REAL,"+
                           "so4_2fofc_mean_cc60 REAL, so4_2fofc_stdev_cc60 REAL,"+
                           "orires TEXT, vol_fofc INT, vol_2fofc INT,"+ 
                           "charge REAL, resolution REAL, db_id TEXT, bin INT, batch INT, omit INT,"+
                           "solc REAL, fofc_sig_in REAL, twofofc_sig_in REAL, fofc_sig_out REAL, twofofc_sig_out REAL, dmove REAL,cstr TEXT)")
               cur.execute("INSERT INTO Peaks VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", data.transpose())
      
          con.commit()
          con.close()

     def store_results_csv(self,results,filename):
          print "STORING RESULTS FOR %s PEAKS TO CSV FILE %s" % (results.shape[0],filename)
          np.savetxt(filename,results,fmt=",".join(self.results_csv_format))

     def read_results_csv(self,filename):
          results_array = np.loadtxt(filename,delimiter=',',dtype=self.results_dtype)
          print "READ RESULTS FOR %s PEAKS FROM FILE %s" % (results_array.shape[0],filename)
          return results_array
