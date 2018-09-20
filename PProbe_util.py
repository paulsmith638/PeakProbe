from __future__ import division
#generic imports
import sys,math,ast,copy
import numpy as np

#utility class of agnostic functions that should work
#under all python versions with/without scipy


class Util:
      def __init__(self):
            pass

      def gen_db_id(self,pdb_code,chainid,resid):
      #string concatenation of the pdb_code,chainid, and padded resid/peak id number
      # xxxx_n_yyyyy xxx3=pdb_code, n=chainid, yyyyy = zero padded resid
            return pdb_code+"_"+chainid+"_"+str(resid.zfill(5))

      def gen_unids(self,awl,model=None):
            at_record = awl.format_atom_record()
            serial_to_coord = at_record[11:54]
            unid_atom = hash(serial_to_coord)
            if model is None:
                  model = awl.model_id.strip()
            stc_m = serial_to_coord+str(model)
            unid_all = hash(stc_m)
            resgroup = awl.parent().parent()
            rg_id = resgroup.id_str()
            resname=awl.resname.strip()
            r_n_m = rg_id+resname+str(model)
            unid_rg = hash(r_n_m)
            return unid_atom,unid_all,unid_rg

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

   
      def write_atom(self,serial,name,alt,resname,chain,resid,ins,x,y,z,occ,temp,element,charge):
            pdb_fmt_str ="{:<6s}{:5d} {:^4s}{:1s}{:3s}{:>2s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n" 
            record = pdb_fmt_str.format("ATOM",serial,name,alt,resname,chain,resid,ins,x,y,z,occ,temp,element,charge)
            return record


      def index_by_pdb(self,data_array):
            #grab a single pdb a large dataset (useful for cluster analysis)
            #uses the 1st 4 characters of dbid.
            #returns a dictionary with lists of indices (in case not sorted)
            pdbid_hash = {}
            #first pdbid
            prevpdb = data_array['id'][0][0:4]
            #pdbid_col = np.zeros(data_array.shape[0],dtype='|S4')
            for index,peak in enumerate(data_array):
                  pdbid = peak['id'][0:4]
                  if pdbid in pdbid_hash:
                        pdbid_hash[pdbid].append(index)
                  else:
                        pdbid_hash[pdbid] = [index,]
                        if pdbid != prevpdb:
                              prevpdb = pdbid                        
            return pdbid_hash



      def batch_data_equal(self,data_array,max_size):
            #all data are assigned a random int from 0 - 999
            #for equal size batches, 1000 % batches must be zero (1000 batches numbered 0-999)
            #returns an integer array with grouped batches to give max_size target
            num_target = 1
            data_size = data_array.shape[0]
            for num_groups in (1,2,4,5,8,10,20,25,40,50,100,125,200,250,500,1000):
                  exp_batch_size = data_size/num_groups
                  if exp_batch_size < max_size:
                        num_target = num_groups
                        break #stop when a suitable grouping is found
                  if num_groups == 1000:
                        num_target = 1000 #last resort
            int_mask = data_array['batch'] % num_groups
            return int_mask

      def scale_density(self,sigma,solc):
            return sigma/np.sqrt(1.0-solc)

      def pick_from_prob(self,kde_probs):
            preds = np.argsort(kde_probs,axis=1)[:,::-1]+1
            tally = np.bincount(preds[:,0],minlength=5)
            pick1 = np.argmax(tally)
            if tally[pick1] == 1: #each prior picks a different pick
                  best_scores = np.sort(kde_probs,axis=1)[:,::-1]
                  best_prior = np.argmax(best_scores[:,0])
                  pick1 = np.argmax(kde_probs[best_prior,:])+1
                  #print "PRIOR AMBIG",kde_probs[0],kde_probs[1],kde_probs[2],best_prior,pick1
            return pick1
