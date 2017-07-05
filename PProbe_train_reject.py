#cannot be run through phenix python, as scipy conflicts
#must use database output and run in regular python2.7
import sys,os,copy,math,ast
import numpy as np
import scipy as sp
import scipy.optimize as op
import scipy.integrate as spi
import scipy.spatial.distance as spsd
from sklearn.neighbors import NearestNeighbors as nn
from sklearn import preprocessing as skpp
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=1e6, edgeitems=1e6)
import sqlite3 as lite
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from PProbe_classify import ClassifierFunctions as ppclas

#constant for spline function fitting
RESMIN = 0.7
RESMAX = 4.2

#functions used in resolution dependent scaling schemes
class RejectFunctions:
     def __init__(self,verbose=False):
          cfunc = ppclas(verbose=True)
          #function from classifier needed here
          #pass references to generate unbound methods
          self.johnsonsu_stats=cfunc.johnsonsu_stats
          self.johnsonsu_pdf=cfunc.johnsonsu_pdf
          self.get_stats=cfunc.get_stats
          self.get_res_scales=cfunc.get_res_scales
          self.get_post_pca_scales = cfunc.get_post_pca_scales
          self.get_jsu_coeffs = cfunc.get_jsu_coeffs
          self.gen_xform_mat = cfunc.gen_xform_mat
          self.xform_data = cfunc.xform_data
          self.standardize_data = cfunc.standardize_data
          self.pca_xform_data = cfunc.pca_xform_data
          self.discriminant_analysis = cfunc.discriminant_analysis
          self.score_stats = cfunc.score_stats
          self.score_breakdown = cfunc.score_breakdown
          self.ppsel = cfunc.ppsel #inherit imported class?

    
     #CHISQ DISTRIBUTION FITTING
     #used for rejection during training
     def chisq_pdf(self,k,xval): #conventional chisq
     #allow non-discrete dof
          gamma_int = lambda x,z: (x**(z-1))*np.exp(-x)
          gamma_k2 = spi.quad(gamma_int,0,np.inf,args=(k/2.0,))[0]
          chisq = 1/(2**(k/2.0)*(gamma_k2))*xval**((k/2.0)-1)*np.exp(-xval/2.0)
          return chisq     

     def chisq_cdf(self,xval,k):
          #hacked quick integration, quad.spi doesn't do well here, so wing it
          numbins = 10000
          chibins = np.linspace(0,200,numbins)
          binwidth = float(np.amax(chibins))/numbins
          chidensity = self.chisq_pdf(k,chibins)*binwidth
          values = []
          for x in xval:
               tosumbin = chibins <= x #boolean mask
               cdfsum = np.nansum(chidensity[tosumbin])
               values.append(cdfsum)
          return np.clip(np.array(values),0.0,1.0)
          """
          #repeat pdf here as fitting and integrating require different input order
          gamma_int = lambda x,z: (x**(z-1))*np.exp(-x)
          gamma_k2 = spi.quad(gamma_int,0,np.inf,args=(k/2.0,))[0]
          chisq = lambda x,k: 1/(2**(k/2.0)*(gamma_k2))*x**((k/2.0)-1)*np.exp(-x/2.0)
          cumulant = spi.quad(chisq_pdf,0,xval,args=(k,))
          return cumulant
          """
          
     def chisq_target(self,param,xval,yval):
          #target=1.0/((param)*(param)) + 5.0*np.nansum((yval - self.chisq_pdf(param,xval))**2)
          target=np.nansum(yval*(yval - self.chisq_pdf(param,xval))**2)
          return target

     def chisq_fit(self,xval,yval):
          initial = np.array((xval[np.argmax(yval)],))#initial 
          constraints = np.array([(0.95,99.0),])
          L = op.fmin_l_bfgs_b(self.chisq_target,initial,args=(xval,yval),bounds=constraints,approx_grad=True,disp=False)
          #bad integration hack to check that resulting curve is a probability distribution and calculate mean
          xtest = np.linspace(0,1000,1000)
          curve_sum  = np.nansum(self.chisq_pdf(L[0],xtest))
          curve_mean = np.nansum(np.multiply(self.chisq_pdf(L[0],xtest),xtest)) 
          print "          CHISQ SUM %4.2f MEAN %4.2f" % (curve_sum,curve_mean),L[0]
          return (L[0],curve_mean)

     #GAMMA DISTRIBUTION FITTING
     def gamma_pdf(self,param,xval):
          k,shape = param
          gamma_int = lambda x,z: (x**(z-1))*np.exp(-x)
          gamma_k = spi.quad(gamma_int,0,np.inf,args=(k,))[0]
          gamma = 1/(gamma_k*shape**k)*xval**((k)-1)*np.exp(-xval/shape)

     def gamma_cdf(self,xval,k,shape):
          numbins = 10000
          gammabins = np.linspace(0,200,numbins)
          binwidth = float(np.amax(gammabins))/numbins
          gammadensity = self.gamma_pdf(k,gammabins)*binwidth
          values = []
          for x in xval:
               tosumbin = gammabins <= x #boolean mask
               cdfsum = np.nansum(gammadensity[tosumbin])
               values.append(cdfsum)
          return np.clip(np.array(values),0.0,1.0)

     def gamma_target(self,param,xval,yval):
          target = np.nansum((yval - self.gamma_pdf(param,xval))**2)
          return target


     def gamma_fit(self,xval,yval):
          initial = np.array((10.0,1.0))#initial 
          constraints = np.array([(0.95,50.0),(-10.0,10,0)])
          L = op.fmin_l_bfgs_b(self.gamma_target,initial,args=(xval,yval),bounds=constraints,approx_grad=True,disp=False)
          return L[0]
          

          


     def training_rejects(self,data_array,data_columns):
          print "REJECTING INPUT PEAKS BY PRE-DEFINED CUTOFFS"
          #takes data array with all data (raw, pca scaled, info) and rejects based on dictionary of cutoffs
          #returns boolean array 1=reject
          reject_mask = np.zeros(data_array.shape[0],dtype=np.bool)
          reject_cutoffs={'ccSf':(-999.9,999.9),'ccWf':(-999.9,999.9),'ccS2':(-999.9,999.9),'ccW2':(-999.9,999.9),
                          'ccSifi':(-999.9,999.9),'ccSifo':(-999.9,999.9),'ccSi2i':(-999.9,999.9),'ccSi2o':(-999.9,999.9),
                          'ccSifr':(-999.9,999.9),'ccSi2r':(-999.9,999.9),'ccWif':(-999.9,999.9),'ccWi2':(-999.9,999.9),
                          'ccSf60':(-999.9,999.9),'sdSf60':(-999.9,999.9),'ccS260':(-999.9,999.9),'sdS260':(-999.9,999.9),
                          'vf':(0.0,300.0),'v2':(0.0,300.0),'charge':(-30,30),
                          'RX0':(-10.0,10.0),'RX1':(-10.0,10.0),'RX2':(-10.0,10.0),'RX3':(-10.0,10.0),'RX4':(-10.0,10.0),
                          'RX5':(-10.0,10.0),'RX6':(-10.0,10.0),'RX7':(-10.0,10.0),'RX8':(-10.0,10.0),
                          'RX9':(-10.0,10.0),'RX10':(-10.0,10.0),'RX11':(-10.0,10.0),'RX12':(-10.0,10.0),'RX13':(-10.0,10.0),
                          'RX14':(-10.0,10.0),'RX15':(-10.0,10.0),'RX16':(-10.0,10.0),'RX17':(-10.0,10.0),'RX18':(-10.0,10.0),
                          'res':(0.5,5.2),'fofc_sigi':(-999.9,999.9),'2fofc_sigi':(-999.9,999.9),'fofc_sigo':(1.0,999.9),
                          '2fofc_sigo':(0.2,999.9),'dmove':(0.0,3.5)}
          for column in data_columns:
               if column in reject_cutoffs:
                    lowc,highc = reject_cutoffs[column]
                    reject_sel = np.logical_or(np.array(data_array[column],dtype=np.float64) < lowc, 
                                               np.array(data_array[column],dtype=np.float64) > highc)
                    reject_count = np.count_nonzero(reject_sel)
                    if reject_count > 0:
                         print "     REJECTED %6s PEAKS WITH %10s CUTOFF %4.2f %4.2f" % (reject_count,column,lowc,highc)
                    #combine_rejects
                    reject_mask = np.logical_or(reject_sel,reject_mask)
          total_rej = np.count_nonzero(reject_mask)
          print "TOTAL REJECTS %10s %4.3f" % (total_rej,float(total_rej)/data_array.shape[0])
          return reject_mask



     def results_chisq(self,data_array,results_array,plot=False,plot_str='out'):
          cfunc = ppclas(verbose=True)
          if plot:
               gridplot = plt.figure(figsize=(64,24),tight_layout=True)
          self.score_breakdown(data_array,results_array)
          results_class = results_array['rc']
          tpsel = np.logical_and(results_class > 0,results_class <=4)
          tnsel = np.logical_and(results_class > 4,results_class <=8)
          fpsel = np.logical_and(results_class > 8,results_class <=12)
          fnsel = np.logical_and(results_class > 12,results_class <=16)

          stat_list = ['score','llgS','llgW','chiS','chiW','fchi']
          pop_list = ["All","labS","labW","obsS","obsW","TP","TN","FP","FN","pTP","pTN","pFP","pFN"]
          plt_count = 1
          
          for sindex,stat in enumerate(stat_list):
               for index,rpop in enumerate(pop_list):
                    if rpop == "All":
                         selector = np.ones(data_array.shape[0],dtype=np.bool_)
                    if rpop == "labS":
                         selector = np.logical_or(tpsel,fnsel)
                    if rpop == "labW":
                         selector = np.logical_or(tnsel,fpsel)
                    if rpop == "obsS":
                         selector = np.logical_or(tpsel,fpsel)
                    if rpop == "obsW":
                         selector = np.logical_or(tnsel,fnsel)
                    if rpop == "TP" or rpop == "pTP":
                         selector = tpsel
                    if rpop == "TN" or rpop == "pTN":
                         selector = tnsel
                    if rpop == "FP" or rpop == "pFP":
                         selector = fpsel
                    if rpop == "FN" or rpop == "pFN":
                         selector = fnsel
                    statcol = results_array[stat][selector]
                    sub_class = results_class[selector]

                    if statcol.shape[0] > 10:
                         bins = 100
                         xmax = np.clip(np.percentile(statcol,99.99),10,np.inf)
                         plot_fit = False
                         if (stat != 'llgS') and (stat != 'llgW') and (stat != 'score'):
                              chis_hist,chis_bins = np.histogram(statcol,bins=bins,density=True,range=(0,xmax))
                              fit = self.chisq_fit(((chis_bins[:-1] + chis_bins[1:]) / 2.0),chis_hist)
                              plot_fit = True
                         if plot:
                              sub = gridplot.add_subplot(len(stat_list),len(pop_list),plt_count)
                              #sub.set_xticklabels(size=7)
                              #sub.set_yticklabels(size=7)
                              plot_xmax = np.clip(np.percentile(statcol,99.9),10,300)
                              plot_xmin = np.clip(np.percentile(statcol,0.01),-50,0)
                              if stat == stat_list[0]:
                                   sub.set_title(rpop+" "+str(statcol.shape[0]))
                              if rpop == pop_list[0]:
                                   sub.set_ylabel(stat)
                              xdat = np.linspace(plot_xmin,plot_xmax,bins)
                              if rpop[0] != 'p':
                                   sub.hist(statcol,normed=True, bins=bins,range=(plot_xmin,plot_xmax),alpha=0.5,color='r')
                              else:
                                   for subselect,color in zip((1,2,3,4),('r','b','g','y')):
                                        subdata = statcol[(sub_class-1) % 4 == subselect-1]
                                        if subdata.shape[0] > 0:
                                             sub.hist(subdata,normed=False, stacked=False,bins=bins,range=(plot_xmin,plot_xmax),alpha=0.5,color=color)
                                             sub.text(0.3+0.1*subselect,0.9,"C%1d" % subselect,verticalalignment='bottom',horizontalalignment='right',
                                                      transform=sub.transAxes,fontsize=12,color=color)
                              if plot_fit:
                                   ydat = self.chisq_pdf(fit[0],xdat)
                                   sub.plot(xdat,ydat,color='b')
                                   sub.text(0.9,0.9,"%4.2f" % fit[0],verticalalignment='bottom',horizontalalignment='right',transform=sub.transAxes,fontsize=12)
                              sub.text(0.9,0.8,"%4.2f" % np.nanmean(statcol),verticalalignment='bottom',horizontalalignment='right',transform=sub.transAxes,fontsize=12)
                    plt_count = plt_count + 1
                    
          if plot:
               plt.savefig("CHISQ_FITS_"+plot_str+".png")
               plt.close()


     def feature_chisq(self,data_array):
          """
          For a data batch, features are normalized, and chisq stat calculated
          Conforms to chisq(k=19).  Requires a sufficient batch of data for accurately
          estimating mean/std.
          """
          selectors = self.ppsel(data_array)
          #select only feature columns
          feature_chisq = np.zeros(data_array.shape[0])
          for selector_mask,name in zip((selectors.inc_obss_bool,selectors.inc_obsw_bool),("S","W")):
               population = data_array[selector_mask]
               sel_ind = np.argwhere(selector_mask == True)
               #new_data_copy
               data_in = population[selectors.pca_view_col].view(selectors.raw_dtype)
               #calculate means
               colmeans = np.nanmean(data_in,axis=0)
               colstd = np.nanstd(data_in,axis=0)
               #standardize on the fly
               #treats population as normally distributed, unknown consequences . . . 
               data_in = (data_in - colmeans[None,:])/colstd[None,:]
               feature_chisq[sel_ind] = np.nansum(np.multiply(data_in,data_in),axis=1)
          return feature_chisq


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


     def knn_analysis(self,data_array,plot=False,idstr=''):
          """
          knn scales as nlogn, so very large data chunks are impractical
          knn on a subset is suspect, but if we assume somewhat uniformly distributed
          features, batchwise make be valid.  If a particular peak has no "near" 
          neighbor wrt population averages, even in a subset, than it may be suspect
          """
          print "KNN ANALYSIS %s PEAKS" % data_array.shape[0]
          selectors = self.ppsel(data_array)
          knn_chisq = np.zeros(data_array.shape[0]) - 1.0 #negative values errors, ignore later
          for selector_mask,name in zip((selectors.inc_obss_bool,selectors.inc_obsw_bool),("S","W")):
               #first select S/W
               pop_ind = np.argwhere(selector_mask)
               population = data_array[selector_mask]
               data_in = population[selectors.pca_view_col].view(selectors.raw_dtype)
               pop_result = np.zeros(population.shape[0])
               #break down into batches, max size = 50000
               int_mask = self.batch_data_equal(population,50000)
               num_groups = np.amax(int_mask) + 1
               print "     ANALSIS POPULATION %s IN %s GROUPS" % (name,str(num_groups))
               for groupb in np.arange(num_groups):
                    gmask = int_mask == groupb
                    sel_ind = np.argwhere(gmask)
                    input_data = data_in[gmask]
                    knn_result = self.knn_dist(input_data,numk=11)
                    pop_result[sel_ind] = knn_result #recombine output to population results
               knn_chisq[pop_ind] = pop_result #recombine populations to full array
          return knn_chisq


     def reject_peaks(self,results_array,so4_mask):
          #hard cutoffs for rejections
          print "REJECTING PEAKS BASED ON HARD CUTOFFS FOR CHISQ"
          reject_cutoffs={'chiS':(0.0,44.0),'chiW':(0.0,44.0),'fchi':(0.0,44.0),'kchi':(0.0,29.6)}
          #array for reject flags, 4 columns
          total_rejects = np.zeros((results_array.shape[0],4),dtype=np.bool)
          for index,column in enumerate(['chiS','chiW','fchi','kchi']):
               if column in reject_cutoffs:
                    lowc,highc = reject_cutoffs[column]
                    toolow_mask = results_array[column] < lowc

                    toohigh_mask = results_array[column] > highc
                    reject = np.logical_or(toolow_mask,toohigh_mask)
                    if column == 'chiS':
                         #only reject sulfate based on sulfate chisq
                         reject = np.logical_and(reject,so4_mask) 
                    if column == 'chiW':
                         reject = np.logical_and(reject,np.invert(so4_mask))
                    count = np.count_nonzero(reject)
                    print "     CHIREJ %7s PEAKS BY %5s CUTOFFS %3.1f %3.1f" % (count,column,lowc,highc)
                    total_rejects[:,index] = reject
          reject_counts = np.nansum(total_rejects,axis=1)
          for i in range(5):
               count = np.count_nonzero(reject_counts == i)
               print "     PEAKS REJECTED %1s X %8s" % (str(i),count)
          return total_rejects

     def normalize_columns(self,data_array): 
          #probably exists in numpy somewhere
          #only numerical numpy data
          colmeans = np.nanmean(data_array,axis=0)
          colstd = np.nanstd(data_array,axis=0)
          return (data_array - colmeans[None,:])/colstd[None,:]



     def knn_dist(self,data_array,numk=11): #numk = neighbors, includes initial zero for self
          """
          Nearest Neighbor search finds numk - 1 nearest neighbors by Euclidean distance
          in feature space (samples the leading edge of the histogram of the entire distance matrix).
          These distances are normalized and the chi_sq stastic is calculated.  Peaks with high chisq
          values would be suspected outliers.  Analysis requires a population of peaks large enough
          to represent a realistic distribution of distances, which is automatically the case in training,
          but may not be true for single structures or small batches of peaks

          For selecting 10nn, output dist^2 measurements follow chisq(k=10), even for random data.
          """
          #data should be broken down by S/W population first
          #data array only numeric view, no info, id etc.
          data_in = data_array
          data_in = self.normalize_columns(data_in)
          nbrs = nn(n_neighbors=numk,algorithm='ball_tree',metric='l2',n_jobs=10).fit(data_in)
          dist = nbrs.kneighbors(data_in)[0] #output only distances
          distmean,diststd = np.nanmean(dist[:,1::]),np.nanstd(dist[:,1::])#ignore the zero for self-neighbor
          zdist = np.divide(dist[:,1::]-distmean,diststd)#normalize neighbor distances to entire population
          zdistsq_sum = np.nansum(np.multiply(zdist,zdist),axis=1) #chisq stat
          print "     KNNCHISQ %s PEAKS MEAN=%s" % (str(data_array.shape[0]),str(np.nanmean(zdistsq_sum)))
          return zdistsq_sum
