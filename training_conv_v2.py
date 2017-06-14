import sys,os,copy,math,ast
import numpy as np
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=1e6, edgeitems=1e6)
from PProbe_train import TrainingFunctions as pptrain
from PProbe_selectors import Selectors as ppsel
from PProbe_dataio import DataIO as ppio

def run(args):
     #instantiate all needed classes
     iofunc = ppio()
     trafunc = pptrain()

     raw_total_array = iofunc.read_sql(args[0])
     #sort by resolution, then peak id
     raw_data = np.sort(raw_total_array,order=['res','id'])
     
     #resolution dependent scaling by mean/sigma
     trafunc.calculate_res_scales(raw_data,plot=True)     
     norm_data = trafunc.standardize_data(raw_data,post_pca=False)

     #calculate resolution dependent PCA transformation matrix coefficients
     trafunc.calc_res_pca(norm_data,plot=True)

     #do actual PCA transformation
     pca_data = trafunc.pca_xform_data(norm_data,plot=True)

     #calculates resolution dependent scales for PCA transformed data
     trafunc.calculate_res_scales(pca_data,post_pca=True,plot=True)

     #scale PCA transformed data
     pca_scaled_data = trafunc.standardize_data(pca_data,post_pca=True)

     #calculate distribution coefficients, write out
     trafunc.calc_jsu_coeff(pca_scaled_data,plot=True)
     trafunc.sn_plot()

     #analyze data by discriminant analysis
     trafunc.discriminant_analysis(pca_scaled_data)

if (__name__ == "__main__"):
     run(sys.argv[1:])

