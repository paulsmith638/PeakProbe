from __future__ import division
#generic imports
import sys,math,ast,copy
import numpy as np
#cctbx imports
from scitbx.array_family import flex
from iotbx.ccp4_map import write_ccp4_map
import iotbx.pdb
from scitbx import matrix
#others
import sqlite3 as lite

class Util:
      def __init__(self):
            pass


      def original_residue(self,peak_object):
      #utility function to output the residue identity of contacts close to peak
      #the peak atom itselv is given chainid ZZ and resid 9999
      #contacts are sorted dictionary, so can return upon a match
            for contact in peak_object.contacts:
                  resname=contact['resname']
                  dist=contact['distance']
                  chainid = contact['chain']
                  resid = contact['resid']
                  if not (chainid == 'ZZ' and resid == '9999'):
                        if resname == "HOH" and dist < 1.6:
                              return "HOH"
                        elif resname == "SO4" and dist < 1.6:
                              return "SO4"
                        elif resname == "PO4" and dist < 1.6:
                              return "PO4"
                        else:
                              return "XXX"
            return "XXX"
            


      def charge(self,peak_object):
      #no longer "charge", but rather a probabilistic evaluation of the local environment
      #all contacts within 5A of a S/P of a sulfate phosphate or a water with 5sig fofc density
      #were analyzed by atom type (e.g. ALA-O).  The ratio of number of contacts per so4/po4 to
      #to water, take ln(ratio) to give "logodds" ratios below (all empirical)
      #summed to give a pseudo-probability, not weighted or normalized to number of contacts
      #however, gives ~80% F1 score upon logistic regression alone
            logodds = ast.literal_eval('{"TRP-O":-2.3824 ,"MET-O":-2.3648 ,"TYR-O":-2.3621 ,"PRO-O":-2.2968 ,"GLN-O":-2.1017 ,"PRO-N":-1.9836 ,"PHE-O":-1.9821 ,"LEU-O":-1.9613 ,"ILE-N":-1.9404 ,"VAL-O":-1.9377 ,"MET-SD":-1.9294 ,"ILE-O":-1.9175 ,"ALA-O":-1.8735 ,"TYR-N":-1.8496 ,"VAL-N":-1.8293 ,"ASP-O":-1.7759 ,"ASN-O":-1.7619 ,"PHE-N":-1.7246 ,"SER-O":-1.6582 ,"ARG-O":-1.6056 ,"LEU-N":-1.5904 ,"MET-N":-1.5608 ,"GLY-O":-1.5567 ,"TRP-N":-1.5534 ,"GLU-O":-1.5348 ,"LYS-O":-1.5013 ,"CYS-N":-1.4648 ,"THR-O":-1.4505 ,"CYS-O":-1.4334 ,"ALA-N":-1.3181 ,"HIS-O":-1.2472 ,"ASN-N":-1.2010 ,"ASP-OD1":-1.1974 ,"GLN-N":-1.1047 ,"ASP-N":-1.0909 ,"CYS-SG":-1.0854 ,"GLU-OE1":-1.0400 ,"ASP-OD2":-0.9923 ,"GLU-OE2":-0.9691 ,"ASN-OD1":-0.8993 ,"GLU-N":-0.8775 ,"GLN-OE1":-0.7805 ,"HIS-N":-0.7076 ,"ARG-N":-0.6993 ,"THR-N":-0.6934 ,"SER-N":-0.6783 ,"TRP-NE1":-0.6253 ,"GLY-N":-0.5769 ,"LYS-N":-0.5465 ,"TYR-OH":-0.4579 ,"GLN-NE2":-0.3702 ,"THR-OG1":-0.3541 ,"MG-MG":-0.3360 ,"ASN-ND2":-0.2997 ,"SER-OG":-0.1467 ,"HIS-ND1":0.1188 ,"HIS-NE2":0.2007 ,"ARG-NE":0.4366 ,"ZN-ZN":0.5237 ,"ARG-NH1":0.8532 ,"LYS-NZ":1.0653 ,"ARG-NH2":1.1350}')
      # no consideration given to altloc or distance scaling
            running_prob = 0.0
            for contact in peak_object.contacts:
                  name = contact['name']
                  resn = contact['resname']
                  dist = contact['distance']
                  cont_id = resn+"-"+name
                  if logodds.has_key(cont_id) and float(dist) < 5.0:
                        running_prob = running_prob + logodds[cont_id]
            return running_prob
	

      def print_results(self,results):
      #utility function to print results from a dictionary direct to stdout
            if not "NaN" in results.itervalues():
                  print ("DATA: %2s %4s SO4_FOFC %5.4f %5.4f HOH_FOFC %5.4f %5.4f SO4_2FOFC %5.4f %5.4f HOH_2FOFC %5.4f %5.4f " + 
                        "INV(fin,fout,2fin,2fout,fR,2fR,watf,wat2f) %5.4f %5.4f %5.4f %5.4f %5.4f %5.4f %5.4f %5.4f " + 
                        "CC60 %5.4f %5.4f %5.4f %5.4f ORI %3s VOLS(F,2F) %5d %5d %-.2f %s %s") % \
                        (results['chainid'],results['resid'],
                         results['so4_cc_fofc_in'],results['so4_cc_fofc_out'], 
                         results['wat_cc_fofc_in'],results['wat_cc_fofc_out'], 
                         results['so4_cc_2fofc_in'],results['so4_cc_2fofc_out'], 
                         results['wat_cc_2fofc_in'],results['wat_cc_2fofc_out'], 
                         results['so4_cc_fofc_inv_in'],   results['so4_cc_fofc_inv_out'], 
                         results['so4_cc_2fofc_inv_in'],  results['so4_cc_2fofc_inv_out'],  
                         results['so4_cc_fofc_inv_rev'],  results['so4_cc_2fofc_inv_rev'],  
                         results['wat_cc_fofc_inv'],  results['wat_cc_2fofc_inv'], 
                         results['so4_fofc_mean_cc60'],results['so4_fofc_stdev_cc60'],
                         results['so4_2fofc_mean_cc60'],results['so4_2fofc_stdev_cc60'],
                         results['orires'],results['vol_fofc'],results['vol_2fofc'],results['charge'],
                         results['resolution'],results['db_id'])
            else:
                  print "DATA: %2s %4s FAILED!" % (str(results['chainid']),str(results['resid']))


      def gen_db_id(self,pdb_code,chainid,resid):
      #string concatenation of the pdb_code,chainid, and padded resid/peak id number
      # xxxx_n_yyyyy xxx3=pdb_code, n=chainid, yyyyy = zero padded resid
            return pdb_code+"_"+chainid+"_"+str(resid.zfill(5))

      def assign_bin(self,resolution):
            #assigns a bin number based on resolution (log based bins)
            bins = {'bin1':[0.00,1.31],'bin2':[1.31,1.50],'bin3':[1.50,1.72], 
                    'bin4':[1.72,1.97],'bin5':[1.97,2.25],'bin6':[2.25,2.58], 
                    'bin7':[2.58,2.95],'bin8':[2.95,3.37],'bin9':[3.37,np.inf]}

            res = float(resolution)
            for binname,resrange in bins.iteritems():
                  if float(res) >= resrange[0] and res < resrange[1]:
                        return int(binname[3])
            return np.nan

      def assign_resolution(self,pdb_id,input_resolution=999.99):
            #currently queries a database
            import sqlite3 as lite
            con = lite.connect('/home/paul/projects/phenix/pdb_data/xray_structures.sql')
            cur = con.cursor()
            record = cur.execute("SELECT resolution FROM Structures WHERE pdbid=?",(pdb_id,))
            try:
                  #fetch mathing pdb resolution
                  reso = float(record.fetchone()[0])
            except:
                  reso = input_resolution
            return reso



      def store_results(self,results,db_name):
      #store results in an sqlite db
            if not "NaN" in results.itervalues():
                  data = np.array((results['chainid'],results['resid'],
                                   results['so4_cc_fofc_in'],results['so4_cc_fofc_out'], 
                                   results['wat_cc_fofc_in'],results['wat_cc_fofc_out'], 
                                   results['so4_cc_2fofc_in'],results['so4_cc_2fofc_out'], 
                                   results['wat_cc_2fofc_in'],results['wat_cc_2fofc_out'], 
                                   results['so4_cc_fofc_inv_in'],   results['so4_cc_fofc_inv_out'], 
                                   results['so4_cc_2fofc_inv_in'],  results['so4_cc_2fofc_inv_out'],  
                                   results['so4_cc_fofc_inv_rev'],  results['so4_cc_2fofc_inv_rev'],  
                                   results['wat_cc_fofc_inv'],  results['wat_cc_2fofc_inv'], 
                                   results['so4_fofc_mean_cc60'],results['so4_fofc_stdev_cc60'],
                                   results['so4_2fofc_mean_cc60'],results['so4_2fofc_stdev_cc60'],
                                   results['orires'],results['vol_fofc'],results['vol_2fofc'],results['charge'],
                                   results['resolution'],results['db_id'],results['bin'],"0",results['omit'],
                                   results['solc'],results['fofc_sig_in'],results['2fofc_sig_in'],
                                   results['fofc_sig_out'],results['2fofc_sig_out'],
                                   results['dmove'],results['score'][0]),dtype=np.str_)
                  #batch is currently set to 0, used elsewhere for cross_validation, etc.

    	    con = lite.connect(db_name)
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

      def training_rejects(self,data_array,all_data_columns):
            #takes data array with all data (raw, pca scaled, info) and rejects based on dictionary of cutoffs
            #returns boolean array 1=reject
            reject_mask = np.zeros(data_array.shape[0],dtype=np.bool)
            reject_cutoffs={'ccSf':(0.0,1.0),'ccWf':(0.0,1.0),'ccS2':(-1.0,1.0),'ccW2':(0.0,1.0),
                            'ccSifi':(-0.25,0.25),'ccSifo':(-0.25,0.25),'ccSi2i':(-0.25,0.25),'ccSi2o':(-0.3,0.3),
                            'ccSifr':(-0.2,1.0),'ccSi2r':(-0.2,1.0),'ccWif':(-0.25,0.25),'ccWi2':(-0.25,0.25),
                            'ccSf60':(-0.25,1.0),'sdSf60':(0.0,0.1),'ccS260':(-0.2,1.0),'sdS260':(0.0,0.1),
                            'vf':(0.0,300.0),'v2':(0.0,300.0),'charge':(-30,10),
                            'RX0':(-6.0,6.0),'RX1':(-6.0,6.0),'RX2':(-6.0,6.0),'RX3':(-6.0,6.0),'RX4':(-6.0,6.0),
                            'RX5':(-6.0,6.0),'RX6':(-6.0,6.0),'RX7':(-6.0,6.0),'RX8':(-6.0,6.0),
                            'RX9':(-6.0,6.0),'RX10':(-6.0,6.0),'RX11':(-6.0,6.0),'RX12':(-10.0,10.0),'RX13':(-6.0,6.0),
                            'RX14':(-6.0,6.0),'RX15':(-6.0,6.0),'RX16':(-6.0,6.0),'RX17':(-6.0,6.0),'RX18':(-6.0,6.0),
                            'res':(0.7,4.5),'fofc_sigi':(1.0,30.0),'2fofc_sigi':(0.5,12.0),'fofc_sigo':(1.0,30.0),
                            '2fofc_sigo':(0.5,12.0),'dmove':(0.0,1.1)}
            for column in all_data_columns:
                  if column in reject_cutoffs:
                        lowc,highc = reject_cutoffs[column]
                        reject_sel = np.logical_or(np.array(data_array[column],dtype=np.float64) < lowc, 
                                                   np.array(data_array[column],dtype=np.float64) > highc)
                        print "REJECTED %s PEAKS %s" % (np.count_nonzero(reject_sel),column)
                        reject_mask = np.logical_or(reject_sel,reject_mask)
            total_rej = np.count_nonzero(reject_mask)
            print "TOTAL REJECTS",total_rej,float(total_rej)/data_array.shape[0]
            return reject_mask


      def store_final_results(self,raw_data,xform_data,db_name,train_rej=False):
      #store results in an sqlite db
            raw_data_columns = ['ccSf','ccWf','ccS2','ccW2','ccSifi','ccSifo','ccSi2i','ccSi2o','ccSifr','ccSi2r',
                                'ccWif','ccWi2','ccSf60','sdSf60','ccS260','sdS260','vf','v2','charge']
            xform_data_columns = ['RX0','RX1','RX2','RX3','RX4','RX5',
                                 'RX6','RX7','RX8','RX9','RX10','RX11',
                                 'RX12','RX13','RX14','RX15','RX16','RX17','RX18']
            info_data_columns = ['ori','res','id','bin','batch','omit',
                                 'solc','fofc_sigi','2fofc_sigi','fofc_sigo',
                                 '2fofc_sigo','dmove','score']
            raw_data_in = raw_data[raw_data_columns]
            xform_data_in = xform_data[xform_data_columns]
            info_data_in = raw_data[info_data_columns]
            #score is in pca array
            info_data_in['score'] = xform_data['score']
            array_size = info_data_in.shape[0]
            all_data_columns = raw_data_columns + xform_data_columns + info_data_columns
            all_data_formats = [(np.str_,32),(np.str_,32),(np.str_,32),(np.str_,32),(np.str_,32),(np.str_,32),(np.str_,32),(np.str_,32),
                                (np.str_,32),(np.str_,32),(np.str_,32),(np.str_,32),(np.str_,32),(np.str_,32),(np.str_,32),(np.str_,32),
                                (np.str_,32),(np.str_,32),(np.str_,32),(np.str_,32),(np.str_,32),(np.str_,32),(np.str_,32),(np.str_,32),
                                (np.str_,32),(np.str_,32),(np.str_,32),(np.str_,32),(np.str_,32),(np.str_,32),(np.str_,32),(np.str_,32),
                                (np.str_,32),(np.str_,32),(np.str_,32),(np.str_,32),(np.str_,32),(np.str_,32),(np.str_,32),(np.str_,32),
                                (np.str_,32),(np.str_,32),(np.str_,32),(np.str_,32),(np.str_,32),(np.str_,32),(np.str_,32),(np.str_,32),
                                (np.str_,32),(np.str_,32),(np.str_,32)]#need string conversion to play nice with sql
            all_data_dtype = np.dtype(zip(all_data_columns,all_data_formats))
            all_array = np.fromiter((tuple(list(raw_data_in[i])+list(xform_data_in[i])+list(info_data_in[i])) for i in np.arange(array_size)),
                                    dtype=all_data_dtype,count=array_size)
            if train_rej:
                  reject_mask = self.training_rejects(all_array,all_data_columns)
                  join_rejects = np.logical_or(reject_mask,all_array['omit'] == 'True')
                  mask_as_str = join_rejects.astype(np.dtype((np.str_,32)))
                  all_array['omit'] = mask_as_str
            array_to_sql = all_array
    	    con = lite.connect(db_name)
    	    with con:
                  cur = con.cursor()
                  cur.execute("CREATE TABLE IF NOT EXISTS Peaks(so4_cc_fofc_out REAL,wat_cc_fofc_out REAL,so4_cc_2fofc_out REAL,"+ 
                              "wat_cc_2fofc_out REAL,so4_cc_fofc_inv_in REAL, so4_cc_fofc_inv_out REAL,so4_cc_2fofc_inv_in REAL,"+ 
                              "so4_cc_2fofc_inv_out REAL,so4_cc_fofc_inv_rev REAL, so4_cc_2fofc_inv_rev REAL,wat_cc_fofc_inv REAL,"+ 
                              "wat_cc_2fofc_inv REAL,so4_fofc_mean_cc60 REAL, so4_fofc_stdev_cc60 REAL,so4_2fofc_mean_cc60 REAL,"+ 
                              "so4_2fofc_stdev_cc60 REAL,vol_fofc INT, vol_2fofc INT,charge REAL,"+
                              "rx0 REAL,rx1 REAL,rx2 REAL,rx3 REAL,rx4 REAL,rx5 REAL,rx6 REAL,rx7 REAL,rx8 REAL,rx9 REAL,rx10 REAL,rx11 REAL,"+
                              "rx12 REAL,rx13 REAL,rx14 REAL,rx15 REAL,rx16 REAL,rx17 REAL,rx18 REAL,"+
                              "orires TEXT,resolution REAL, db_id TEXT, bin INT,batch INT,omit TEXT,solc REAL,fofc_sig_in REAL,"+
                              "twofofc_sig_in REAL,fofc_sig_out REAL,twofofc_sig_out REAL,dmove REAL,score REAL)")
                  for peak in array_to_sql:
                        cur.execute("INSERT INTO Peaks VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,"+
                                    "?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", peak.transpose())
                  #cur.execute("CREATE INDEX by_db_id ON PEAKS(db_id)")
            con.commit()
    	    con.close()

      def extract_raw(self,results_list):
          col_names = ('ccSf','ccWf','ccS2','ccW2',
                       'ccSifi','ccSifo','ccSi2i','ccSi2o','ccSifr','ccSi2r','ccWif','ccWi2',
                       'ccSf60','sdSf60','ccS260','sdS260','ori','vf','v2','charge','res','id','bin','batch','omit')
          #numpy data type formats
          col_formats = (np.float64, np.float64, np.float64, np.float64,
                         np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64,
                         np.float64, np.float64, np.float64, np.float64,'S16',np.float64,np.float64,np.float64,np.float64,'S16',
                         np.int16,np.int16,np.bool)
          col_dtype = np.dtype(zip(col_names,col_formats))
          
          raw_array = []
          for results in results_list:
               data = tuple([results['so4_cc_fofc_out'], results['wat_cc_fofc_out'], results['so4_cc_2fofc_out'], results['wat_cc_2fofc_out'], 
                             results['so4_cc_fofc_inv_in'],   results['so4_cc_fofc_inv_out'],results['so4_cc_2fofc_inv_in'],  results['so4_cc_2fofc_inv_out'],  
                             results['so4_cc_fofc_inv_rev'],  results['so4_cc_2fofc_inv_rev'],results['wat_cc_fofc_inv'],  results['wat_cc_2fofc_inv'], 
                             results['so4_fofc_mean_cc60'],results['so4_fofc_stdev_cc60'],results['so4_2fofc_mean_cc60'],results['so4_2fofc_stdev_cc60'],
                             results['orires'],results['vol_fofc'],results['vol_2fofc'],results['charge'],results['resolution'],results['db_id'],results['bin'],
                             "0",results['omit']])
               raw_array.append(data)

          data_array = np.fromiter((row for row in raw_array),dtype=col_dtype) 
          return data_array
            
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

   
