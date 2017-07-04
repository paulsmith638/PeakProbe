import sys,os,copy,math,ast
import numpy as np


class PCA:
    def __init__(self):
        pass


    def modal_matrix(self,data_as_rows,verbose=False):
        
        #cov_data = np.cov(data_as_rows.T)
        #explicitly use correlation rather than covariance
        #as large and small bins of data are compared later on
        
        cov_data = np.corrcoef(data_as_rows.T)
        
        
        #store as modal matrix (columns are eigenvectors, sorted by eigenval high to low
        #using Hermetian eigenvalue decomposition
        #eigenval,eigenvect = np.linalg.eigh(cov_data)
        #eigenval_argsort = np.argsort(eigenval)[::-1]
        #returns modal matrix with eigenvectors as columns
        #modal = eigenvect.T[eigenval_argsort].T
        
        #using singular value decomposition (u=v.T), for square symmetric, etc
        # u is essentially a modal matrix, check that det=1
        u,l,v = np.linalg.svd(cov_data)
        modal = u.copy()
        detu = np.linalg.det(u)
        if detu < 0:#swap last vector
            modal[:,-1] = -1.0*modal[:,-1]
        if verbose:
            #print "INPUT COV/CORR"
            #print cov_data
            print "MODAL DETERMINANT", np.linalg.det(modal) #should be 1.0
        return modal

    def pca_by_bin(self,input_data,res,bin_mask,num_bins,window=5,plot=False):
        #input is numerical data columns to be analyzed by covariance
        #bin_mask = integer column that breaks down data by appropriate bins
        #num_bins = index of last bin (actually +1 numbers of bins)


        #initialize empty lists
        eig_data,binres = [],[]
        #generate binw a minimum number of sulfate and a target minimum resolution width
        for resbin in np.arange(num_bins + 1):
            #overlap bins to give "rolling window" PCA calculations
            ressel = np.logical_and(bin_mask >= resbin,bin_mask <= resbin+window)
            databin = input_data[ressel]
            res_block=res[ressel]
            print "    CALC EIGv for BIN %2s RES %4.2f - %4.2f [%4.2f] %6s" % \
                (resbin,np.amin(res_block),np.amax(res_block),np.nanmean(res_block),res_block.shape[0])
            bin_modal = self.modal_matrix(databin,verbose=True)
            eig_data.append(bin_modal)
            #get resolutions for each databin
            resmean = np.nanmean(res_block)
            binres.append(resmean)

        binres_vals = np.array(binres)#1d array of resolutions for fitting
        #creates a 3d array, x is resolution slice, y and z are binwise eigenvectors matrix
        pca_array = np.array(eig_data)
        return pca_array,binres_vals


    def eig_sort(self,ref_modal,target_modal,verbose=False):
        #takes a reference modal matrix and a matrix to be arranged so 
        #that eigen vectors align with the reference
        #gets a bit messy with rows, columns, etc.

        #initialize empty modal matrix
        new_modal = np.zeros(target_modal.shape)

        #pool of vectors from which to pick "best" eigenvector by matching to reference pca set
        vector_pool = target_modal.T #use as rows
        ref_as_rows = ref_modal.T

        #boolean mask array for the vector pool
        selbool = np.ones(vector_pool.shape[0],dtype=np.bool_)
        #iterate through eigenvectors of reference, find best fit in target (absolute)
        for index in np.arange(ref_modal.shape[0]):
            ref_vect = ref_as_rows[index]
            dotvect = np.zeros(vector_pool.shape[0])
            #get dot product for each each eigenvector and the reference vector (abs)
            for eigind,eigv in enumerate(vector_pool):
                #absolute value gives best co-linearity in either orientation
                dotvect[eigind] = np.abs(np.dot(ref_vect,eigv.T))
                #find the largest dot product (sort plus/minus later)
            best_ind = np.argmax(dotvect)
            new_modal[:,index] = vector_pool[best_ind]
            #eliminate the used vector from the pool
            selbool[best_ind] = False
            vector_pool = vector_pool[selbool]
            selbool = selbool[selbool]
                
            #"flip" eigenvectors that are reversed w.r.t the reference vector
            dotprod = np.dot(ref_vect,new_modal[:,index])
            if dotprod < 0:
                new_modal[:,index] = np.multiply(new_modal[:,index],-1.0)
            #avoid spurrious "reflections" in transformation
            #if determinant is negative, flip last eigenvector, which
            #should correspond to smallest eigenvector
            if np.linalg.det(new_modal)< 0:
                new_modal[:,-1] = new_modal[:,index]*-1.0

        if verbose:
            print "   REF MODAL DET:",np.linalg.det(ref_modal)
            print "   OUT MODAL DET:",np.linalg.det(new_modal)
            print "   REF-OUT NORM:",np.linalg.norm(np.subtract(ref_modal,target_modal))
        return new_modal






        

