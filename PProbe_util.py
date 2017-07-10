from __future__ import division
#generic imports
import sys,math,ast,copy
import numpy as np
#cctbx imports
#from scitbx.array_family import flex
#from iotbx.ccp4_map import write_ccp4_map
#import iotbx.pdb
#from scitbx import matrix
#others
try:
      import sqlite3 as lite
except: #skip and hope user doesn't call sqlite
      pass

class Util:
      def __init__(self):
            pass


      def original_residue(self,peak_object,cutoff=1.6):
      #utility function to output the residue identity of contacts close to peak
      #the peak atom itselv is given chainid ZZ and resid 9999
      #contacts are sorted dictionary, so can return upon a match
            for contact in peak_object.contacts:
                  resname=contact['resname']
                  dist=contact['distance']
                  chainid = contact['chain']
                  resid = contact['resid']
                  if not (chainid == 'ZZ' and resid == '9999'):
                        if dist < cutoff:
                              return resname[0:3]
            #if peak was nothing (nothing close by, return XXX
            return "XXX"
            


      def charge(self,peak_object,cutoff=5.0):
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
                  if logodds.has_key(cont_id) and float(dist) < cutoff:
                        running_prob = running_prob + logodds[cont_id]
            return running_prob
	

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
            #dev option to assign resolutions directly from pdb database data
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

   
