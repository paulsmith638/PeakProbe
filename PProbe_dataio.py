import sys,os,copy,math,ast
import numpy as np
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=1e6, edgeitems=1e6)
import sqlite3 as lite


class DataIO:
     def __init__(self):
          self.sql_sel_str = "so4_cc_fofc_out,wat_cc_fofc_out,so4_cc_2fofc_out,wat_cc_2fofc_out,"+\
                             "so4_cc_fofc_inv_in,so4_cc_fofc_inv_out,so4_cc_2fofc_inv_in,so4_cc_2fofc_inv_out,"+\
                             "so4_cc_fofc_inv_rev,so4_cc_2fofc_inv_rev,wat_cc_fofc_inv,wat_cc_2fofc_inv,"+\
                             "so4_fofc_mean_cc60,so4_fofc_stdev_cc60,so4_2fofc_mean_cc60,so4_2fofc_stdev_cc60,"+\
                             "orires,vol_fofc,vol_2fofc,charge,resolution,db_id,bin,batch,omit,"+\
                             "solc, fofc_sig_in, twofofc_sig_in, fofc_sig_out, twofofc_sig_out,dmove,score"
          self.np_raw_col_names = ('ccSf','ccWf','ccS2','ccW2','ccSifi','ccSifo','ccSi2i','ccSi2o',
                                   'ccSifr','ccSi2r','ccWif','ccWi2','ccSf60','sdSf60','ccS260','sdS260',
                                   'ori','vf','v2','charge','res','id','bin','batch','omit','solc',
                                   'fofc_sigi','2fofc_sigi','fofc_sigo','2fofc_sigo','dmove','score')
          self.np_raw_col_formats = (np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, 
                                     np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64,
                                     'S16',np.float64,np.float64,np.float64,np.float64,'S16', np.int16,np.int16,np.bool,np.float64, 
                                     np.float64, np.float64, np.float64, np.float64, np.float64, np.float64)
          self.np_raw_dtype = np.dtype(zip(self.np_raw_col_names,self.np_raw_col_formats))
          self.csv_format = ['%8g','%8g','%8g','%8g','%8g','%8g','%8g','%8g',
                               '%8g','%8g','%8g','%8g','%8g','%8g','%8g','%8g',
                               '%3s','%4g','%4g','%8g','%4g','%12s','%1d','%3d','%1d','%4g',
                               '%4g','%4g','%4g','%4g','%4g','%8g']

          self.results_csv_format = ['%12s','%4g','%8g','%8g','%8g','%8g','%8g','%8g','%8g','%8g','%2g']
          self.results_dtype = [('id','S16'),('res','f4'),('score','f4'),('prob','f4'),
                                ('llgS','f4'),('llgW','f4'),('chiS','f4'),('chiW','f4'),
                                ('fchi','f4'),('kchi','f4')]



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
                             features['dmove'],features['score'][0]])
               raw_array.append(data)

          data_array = np.fromiter((row for row in raw_array),dtype=self.np_raw_dtype) 
          return data_array

     def store_features_csv(self,np_feature_array,filename):
          print "STORING FEATURES FOR %s PEAKS TO CSV FILE %s" % (np_feature_array.shape[0],filename)
          np.savetxt(filename,np_feature_array,fmt=",".join(self.csv_format))

     def read_features_csv(self,filename):
          data_array = np.loadtxt(filename,delimiter=',',dtype=self.np_raw_dtype)
          print "READ FEATURES FOR %s PEAKS FROM FILE %s" % (data_array.shape[0],filename)


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
                                features['dmove'],features['score'][0]),dtype=np.str_)
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
                           "solc REAL, fofc_sig_in REAL, twofofc_sig_in REAL, fofc_sig_out REAL, twofofc_sig_out REAL, dmove REAL,score REAL)")
               cur.execute("INSERT INTO Peaks VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", data.transpose())
      
          con.commit()
          con.close()

     def store_results_csv(self,results,filename):
          print "STORING RESULTS FOR %s PEAKS TO CSV FILE %s" % (results.shape[0],filename)
          np.savetxt(filename,results,fmt=",".join(self.results_csv_format))

     def read_results_csv(self,filename):
          results_array = np.loadtxt(filename,delimiter=',',dtype=self.results_dtype)
          print "READ RESULTS FOR %s PEAKS FROM FILE %s" % (results.shape[0],filename)
