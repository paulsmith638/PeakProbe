import sys,os,copy,math,ast
import numpy as np
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=1e6, edgeitems=1e6)
from PProbe_selectors import Selectors as ppsel
from PProbe_dataio import DataIO as ppio
from PProbe_classify import ClassifierFunctions as ppclas
from PProbe_util import Util as pputil


def run(args):
     #instantiate all needed classes
     iofunc = ppio()
     ppcf = ppclas()
     pput = pputil()
     
     input_db = args[0]
     raw_total_array = iofunc.read_sql(input_db)
     #sort by resolution, then peak id
     raw_data = np.sort(raw_total_array,order=['res','id'])
     selectors = ppsel(raw_data)
     raw_data = raw_data[selectors.included_data_bool] # remove omitted data

     #resolution dependent scaling by mean/sigma
     norm_data = ppcf.standardize_data(raw_data,post_pca=False)

     #do actual PCA transformation
     pca_data = ppcf.pca_xform_data(norm_data,plot=True)

     #scale PCA transformed data
     pca_scaled_data = ppcf.standardize_data(pca_data,post_pca=True)

     #analyze data by discriminant analysis
     ppcf.discriminant_analysis(pca_scaled_data,plot=True)
     ppcf.score_stats(pca_scaled_data,plot=False,write_res=True)
     
     #store data
     #pput.store_final_results(raw_data,pca_scaled_data,"all_data_output.db",train_rej=False)

if (__name__ == "__main__"):
     run(sys.argv[1:])

