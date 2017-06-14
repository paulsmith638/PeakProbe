#cannot be run through phenix python, as scipy conflicts
#must use database output and run in regular python2.7
import sys,os,copy,math,ast
import numpy as np
#from numpy import linalg as LA
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=1e6, edgeitems=1e6)
import sqlite3 as lite
from PProbe_classify import ClassifierFunctions as ppclas


class DataIO:
     def __init__(self):
          pass

     #reads sql database with all columns labeled from PProbe output
     #returns a numpy structured array
     def read_sql(self,filename):
          #names of columns in std sql database
          sel_str = "so4_cc_fofc_out,wat_cc_fofc_out,so4_cc_2fofc_out,wat_cc_2fofc_out,"+\
                    "so4_cc_fofc_inv_in,so4_cc_fofc_inv_out,so4_cc_2fofc_inv_in,so4_cc_2fofc_inv_out,"+\
                    "so4_cc_fofc_inv_rev,so4_cc_2fofc_inv_rev,wat_cc_fofc_inv,wat_cc_2fofc_inv,"+\
                    "so4_fofc_mean_cc60,so4_fofc_stdev_cc60,so4_2fofc_mean_cc60,so4_2fofc_stdev_cc60,"+\
                    "orires,vol_fofc,vol_2fofc,charge,resolution,db_id,bin,batch,omit,"+\
                    "solc, fofc_sig_in, twofofc_sig_in, fofc_sig_out, twofofc_sig_out,dmove,score"
          #column names for structured array, irrelevant database columns ignored
          col_names = ('ccSf','ccWf','ccS2','ccW2',
                       'ccSifi','ccSifo','ccSi2i','ccSi2o','ccSifr','ccSi2r','ccWif','ccWi2',
                       'ccSf60','sdSf60','ccS260','sdS260','ori','vf','v2','charge','res','id','bin','batch','omit',
                       'solc','fofc_sigi','2fofc_sigi','fofc_sigo','2fofc_sigo','dmove','score')
          #numpy data type formats
          col_formats = (np.float64, np.float64, np.float64, np.float64,
                         np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64,
                         np.float64, np.float64, np.float64, np.float64,'S16',np.float64,np.float64,np.float64,np.float64,'S16',
                         np.int16,np.int16,'S32',np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64)
          col_dtype = np.dtype(zip(col_names,col_formats))
          print "OPENING DATABASE", filename
          con = lite.connect(sys.argv[1])
          data_array=[]
          with con:
               con.row_factory = lite.Row
               cur = con.cursor()
               print "EXTRACTING COLUMNS"
               #sql = "SELECT %s FROM Peaks WHERE resolution>='%s' AND resolution<'%s'" % (sel_str,0.5,9.99)
               sql = "SELECT %s FROM Peaks" % sel_str
               iterator = cur.execute(sql)
               data_array = np.fromiter((tuple(row) for row in cur),dtype=col_dtype)
          con.close()
          print "ROWS IN:",data_array.shape[0]
          return data_array

     def print_raw_data(self,raw_data):
          #select just data columns and original ID string
          col_names = ['ccSf','ccWf','ccS2','ccW2',
                       'ccSifi','ccSifo','ccSi2i','ccSi2o','ccSifr','ccSi2r','ccWif','ccWi2',
                       'ccSf60','sdSf60','ccS260','sdS260','charge','vf','v2','res','id','ori']
          sel_array = raw_data[col_names]
          for row in sel_array:
               print ",".join(list((str(x) for x in row)))

