import sys,os,copy,math,ast,time
import numpy as np
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=1e6, edgeitems=1e6)
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PProbe_dataio import DataIO
from PProbe_selectors import Selectors
from PProbe_stats import StatFunc
from PProbe_util import Util
from matplotlib import cm
mpl.rcParams.update({'font.size': 36})

class KDE:
     def __init__(self,master_dictionary,train=False,verbose=False):
          self.verbose = verbose
          self.ppio = DataIO(phenix_python=False)
          self.pput = Util()
          mdict = master_dictionary
          if train == False:
               self.kdedict = mdict['kde']
               self.populations = self.kdedict['populations']
               self.features = self.kdedict['features']
          else:
               self.populations = ['HOH','SO4','OTH','ML1']
               self.features = ['score','cscore']
          self.flat_prior = np.ones(len(self.populations),dtype=np.float32)/len(self.populations)
          #expected value of MLE of alpha coefficients of dirichlet distribution
          #trained against probabilities from +1 smoothed counts from entire PDB
          self.dir_prior = np.array([ 0.8516044 ,  0.04814615,  0.05036076,  0.04988869])
          self.chi_prior1 = np.array([0.97,0.01,0.01,0.01])

     def joint_likelihood(self,feat1,feat2,prior):
          #generates joint lh from 1d vectors (independent features)
          #returns 3d array, x,y are joints, z is for a population
          #each population is normalized, last pop is "all"
          xcenters = self.kdedict[feat1]['xcent']
          xbins = len(xcenters)
          ycenters = self.kdedict[feat2]['xcent']
          ybins = len(ycenters)
          pops = self.kdedict['populations'][:]
          all_joint = np.zeros((xbins,ybins,len(pops)+1)) #include all
          for k in range(len(pops)):
               ihist = np.array(self.kdedict[feat1][pops[k]]['hist'])
               jhist = np.array(self.kdedict[feat2][pops[k]]['hist'])
               all_joint[:,:,k] = np.outer(jhist,ihist).T*prior[k]
          all_joint[:,:,-1] = np.nansum(all_joint[:,:,0:-1],axis=2)
          return all_joint

     def kde_label(self,data_array):
          #label dataset for training / scoring
          #current class labels are
          #1 = water
          #2 = so4/po4
          #3 = other -- anions, polymers, common reagents, etc.
          #4 = metals of any stripe
          selectors = Selectors(data_array)
          labs = selectors.inc_obss_bool
          labw = selectors.inc_obsw_bool
          oth_labels = ['DTT','MAL','EOH','SUC','SCN','P6G','GSH','CO3','CIT','BOG',
                        'NO3','IMD','BME','ACY','PGE','PG4','TRS','MPD','DMS','PEG',
                        'ACT','EDO','GOL',' CL',' BR']
          met_labels = [' MG',' CA',' ZN',' MN',' NI',' CD',' CO',' FE']
          laboth = np.zeros(data_array.shape[0],dtype=np.bool_)
          for residue in oth_labels:
               laboth = np.logical_or(laboth,(data_array['ori'] == residue))
          labmet = np.zeros(data_array.shape[0],dtype=np.bool_)
          for residue in met_labels:
               labmet = np.logical_or(labmet,(data_array['ori'] == residue))
          orin = np.zeros(data_array.shape[0],dtype=np.int16)
          orin[labw] = 1
          orin[labs] = 2
          orin[laboth] = 3
          orin[labmet] = 4
          return orin


     def train_kde(self,data_array,master_dictionary,cols=None,bins=50):
          """
          train vectors of histogram values for two independent scoring
          criteria (x_stat,y_stat) -- names correspond to column names
          in master array data structure.  bins=histogram bins
          residue populations are currently hard-coded below
          """
          #populations
          #HOH = water #1
          #SO4 = sulfate or phosphate #2
          #OTH = chloride, acetate, glycerol, PEG #3
          #ML1 = metals -- trained are MG,CA,MN,ZN #4

          print "TRAINING KDE VECTORS on %s PEAKS:" % data_array.shape[0]
          resid_names = ['HOH','SO4','OTH','ML1']

          if cols is None:
               stat_cols = ['score','cscore']
          else:
               stat_cols = cols

          #data cleanup / selection
          for x_stat in stat_cols:
               cull = np.isnan(data_array[x_stat])
               print "  CULLING %s DATA POINTS" % np.count_nonzero(cull)
               data_array = data_array[np.invert(cull)]
          labels = self.kde_label(data_array)

          kdedict = {}
          kdedict['populations'] = resid_names
          kdedict['features'] = stat_cols

          for x_stat in stat_cols:
               plotx = data_array[x_stat]
               xmean = np.nanmean(plotx)
               xmin,xmax = np.amin(plotx),np.amax(plotx)
               xstd = np.nanstd(plotx)
               xrange = xmax-xmin
               plotxmin = xmin+0.001*xrange
               plotxmax = xmax-0.001*xrange


               p1x = plotx[labels==1]
               p2x = plotx[labels==2]
               p3x = plotx[labels==3]
               p4x = plotx[labels==4]
               

               #linear KDE, all data
               kdestatdict = {}               
               xshist,xsed = np.histogram(plotx,bins=bins,range=(plotxmin,plotxmax),normed=True)
               xcenters = xsed[:-1] + 0.5*(xsed[1:] - xsed[:-1])
               kdestatdict['xcent'] = list(xcenters)

               for dset in [(p1x,"HOH"),(p2x,"SO4"),(p3x,"OTH"),(p4x,"ML1"),(plotx,"ALL")]:
                    kdepopdict = {}
                    xdat = dset[0]
                    print "     TRAIN stat,pop,count,mean,std,min,max,sc_mean,sc_max: ",x_stat,dset[1],xdat.shape[0],xmean,xstd,xmin,xmax,plotxmin,plotxmax
                    kdepopdict['count'] = dset[0].shape[0]
                    xshist,xsed = np.histogram(xdat,bins=xsed,normed=True) #normed as pdf, not pmf
                    nz_min = np.amin(xshist[xshist > 0.0]) #smallest non_zero value
                    t_hist = np.clip(xshist,nz_min,1.0) #fudge a bit, no zero values
                    n_hist = t_hist/np.nansum(t_hist)
                    kdepopdict['hist'] = list(n_hist)
                    kdestatdict[dset[1]] = kdepopdict

               kdedict[x_stat] = kdestatdict
          popcounts = []
          for pop in resid_names:
               popcounts.append(kdedict[stat_cols[0]][pop]['count'])
          
          popfrac = np.clip(popcounts,1,1E20)/np.nansum(popcounts).astype(np.float64) #no zero counts
          #used for prior estimation
          kdedict['scales'] = list(popfrac)
          #new dictionary of dictionaries added to master
          return kdedict



     def kde_bin(self,column,xdat):
          # bins data to histogram grid
          kdedict = self.kdedict
          xcenters = kdedict[column]['xcent']
          xbins = len(xcenters)
          idx_a = np.zeros((xdat.shape[0],xbins))
          for column in range(xbins):
               idx_a[:,column] = np.abs(xcenters[column] - xdat)
          kdebin = np.argmin(idx_a,axis=1)
          return kdebin


     def kde_bprob(self,x1,x2,prior):
          #calculate probabilities from Bayes' Law
          #and given prior (vectorized)
          #assumes independent conditionals
          if len(prior.shape) == 1: # global prior
               all_prior = prior[None,:]
          elif len(prior.shape) == 2: #individual_prior
               all_prior = prior
          joint = np.multiply(x1,x2)
          jprior = np.multiply(joint,all_prior)
          #norm = np.nansum(jprior,axis=1)
          #pclass = np.divide(jprior,norm[:,None])
          #return pclass
          return jprior

     def kde_grid_prob(self,column1,column2,x1,x2,prior):
          kdedict = self.kdedict
          xcenters = kdedict[column1]['xcent']
          xbins = len(xcenters)
          ycenters = kdedict[column2]['xcent']
          ybins = len(ycenters)
          pops = self.populations
          all_joint = self.joint_likelihood(column1,column2,prior)
          #bin data
          peak_ind = np.zeros((x1.shape[0],2),dtype=np.int64)
          peak_ind[:,0] = self.kde_bin(column1,x1)
          peak_ind[:,1] = self.kde_bin(column2,x2)

          #get all likelihoods returns vector of pop lh for each
          llA = np.zeros((x1.shape[0],all_joint.shape[2]))

          # score is log-likelihood gain of pick/grid
          for pind in range(len(pops)):
               lA = all_joint[peak_ind[:,0],peak_ind[:,1],pind]
               llXY =np.log(all_joint[peak_ind[:,0],peak_ind[:,1],-1] - lA)
               llA[:,pind] = np.log(lA) - llXY
          return llA[:,0:4]




     def cmatrix(self,label,pred):
          #confusion matrix for scoring
          #expects integer >= 1 for label/pred
          max_lind = np.amax(label)
          max_pind = np.amax(pred)
          max_all = max(max_lind,max_pind)+1
          cmatrix = np.zeros((max_all,max_all))
          for i in np.arange(1,max_all,1):
               for j in np.arange(1,max_all,1):
                    cmatrix[i,j] = np.count_nonzero(np.logical_and(label==i,pred==j))
          cmatrix[:,0] = np.nansum(cmatrix,axis=1)
          cmatrix[0,:] = np.nansum(cmatrix,axis=0)
          return cmatrix

     def predict_kde(self,prob_array):
          #i=1 numbering for each in populations
          pred_arr = np.zeros(prob_array.shape,dtype=np.int8)
          for index in range(pred_arr.shape[1]):
               pred_arr[:,index,:] = np.argsort(prob_array[:,index,:])[:,::-1]+1 
          return pred_arr

     def kde_score(self,data_array,input_dfile=None):
          if self.verbose:
               print "Generating Histogram Likelihoods for %s Peaks" % data_array.shape[0]
          kdedict = self.kdedict
          resid_names = kdedict['populations']
          features = kdedict['features']
          prior_list = [self.flat_prior,self.dir_prior,self.flat_prior] #flat used twice, updated
          hack_prob = data_array['prob'] - 0.5
          prob_arr = np.zeros((data_array.shape[0],len(prior_list),len(resid_names)))
          for priorind in range(len(prior_list)):
               prior = prior_list[priorind]
               prob_arr[:,priorind,:] = self.kde_grid_prob('score','cscore',data_array['score'],data_array['cscore']+hack_prob,prior)


          #assign structure specific probabilities based on empirical prior
          #start from flat prior, update counts (3 iterations is well converged)
          pput = self.pput
          pdbid_hash = pput.index_by_pdb(data_array)
          for pdbid,pdbind in pdbid_hash.iteritems():
               datain = data_array[pdbind]
               for i in range(3):
                    #update prior based on counts within a structure
                    #starting from flat prior
                    probin = prob_arr[pdbind,2,:]
                    predin = np.argmax(probin,axis=1)+1 # top prediction
                    #print "PREDS",list(np.count_nonzero(predin==x) for x in range(6))
                    schi = np.add(datain['chiS'],datain['cchiS'])
                    wchi = np.add(datain['chiW'],datain['cchiW'])
                    #set stringent criteria for observing in population
                    all_good = np.logical_and(datain['fc'] == 0,datain['edc'] > 0)
                    not_w = datain['prob'] > 0.8
                    good_s = np.logical_and(datain['llgS'] > 0,not_w)
                    good_w = np.logical_and(datain['llgW'] > 0,np.invert(not_w))
                    mismatch = np.invert(np.logical_or(good_s,good_w))
                    valid_w = np.logical_and(predin == 1,np.logical_and(good_w,wchi < 35.0))
                    valid_s = np.logical_and(predin == 2,np.logical_and(good_s,schi < 35.0))
                    valid_o = np.logical_and(predin == 3,np.logical_and(mismatch,wchi > 25.0))
                    valid_m = np.logical_and(predin == 4,np.logical_and(mismatch,schi < 25.0))
                    vgw = np.logical_and(all_good,valid_w)
                    vgs = np.logical_and(all_good,valid_s)
                    vgo = np.logical_and(all_good,valid_o)
                    vgm = np.logical_and(all_good,valid_m)
                    #print "PCOUNT",list(np.count_nonzero(x) for x in [vgw,vgs,vgo,vgm])
                    valid = np.logical_or(vgw,np.logical_or(vgs,np.logical_or(vgo,vgm)))
                    counts = []
                    for predlab in range(len(resid_names)):
                         count_mask = np.logical_and(predin==predlab+1,valid)
                         count_ids = set(datain['id'][count_mask])
                         counts.append(len(count_ids))

                    #counts of each population + 1 (additive smoothing / dirichlet with alpha=1)
                    countin = np.array(counts,dtype=np.float32)
                    alpha= (countin+1.0)/(np.nansum(countin)+len(resid_names))
                    prob_arr[pdbind,2,:] = self.kde_grid_prob('score','cscore',datain['score'],datain['cscore'],alpha)
                    if i == 2:
                         outlist = [pdbid,i+1,len(pdbind),resid_names[0],counts[0],resid_names[1],counts[1],
                                    resid_names[2],counts[2],resid_names[3],counts[3]]
                         #print "     PDB %4s Cycle %1d POPCOUNT (%5d): %3s %5d %3s %5d %3s %5d %3s %5d" % tuple(outlist)
          #cprob = np.clip(data_array['prob'],1E-5,(1.0-1E-5))
          #llgw = np.log(1.0 - cprob) - np.log(cprob)
          #prob_arr[:,3,0] = llgw #np.add(prob_arr[:,2,0],llgw)
          #prob_arr[:,3,1] = np.add(prob_arr[:,2,1],-llgw)
          #prob_arr[:,3,2] = np.add(prob_arr[:,2,2],-llgw)
          #prob_arr[:,3,3] = np.add(prob_arr[:,2,3],-llgw)


          #for pind,peak in enumerate(data_array):
          #     print "NEWB",peak['id'],peak['score'],peak['cscore'],peak['prob'],peak['ori'],prob_arr[pind,0,:],prob_arr[pind,1,:],prob_arr[pind,2,:],prob_arr[pind,3,:]
          return prob_arr



     def gen_plot_hist(self,features=None,prior=None):
          if prior is None:
               prior = self.dir_prior
          kdedict = self.kdedict
          if features is None:
               features = ('score','cscore')
          x_stat = features[0]
          y_stat = features[1]
          probgrid = self.joint_likelihood(x_stat,y_stat,prior)
          all_lh_range = np.amax(probgrid[:,:,-1]) - np.amin(probgrid[:,:,-1]) 
          bad_cut = np.amin(probgrid[:,:,-1]) + all_lh_range*2E-4
          bad2d = probgrid[:,:,-1] < bad_cut
          pick2d = np.argmax(probgrid[:,:,0:4],axis=2)+1
          pick2d[bad2d] = np.amax(pick2d)+1
          return probgrid,pick2d

     def pop_lasso(self,cutoff=0.5):
          prob2d,pick2d = self.gen_plot_hist(prior=self.dir_prior)
          new_grid = np.zeros(pick2d.shape,dtype=np.int16)
          for popind in range(len(self.populations)):
               datain = prob2d[:,:,popind].copy()
               datain = datain/np.nansum(datain)
               pop_mask = pick2d == popind+1
               pop_ind = np.argwhere(pop_mask)
               grid_list = []
               for x,y in pop_ind:
                    grid_list.append((x,y,datain[x,y]))
               grid_list.sort(key = lambda x: x[2])
               gmax = grid_list[-1]
               include = np.zeros(new_grid.shape,dtype=np.bool_)
               include[gmax[0],gmax[1]] = True
               #banned_xy = np.invert(strong)
               cumulant = np.nansum(datain[np.argwhere(include)])
               while cumulant < cutoff:
                    #print cumulant
                    allowed_xy = include.copy()
                    for cx,cy in np.argwhere(include):
                         for x_adj in [-1,0,1]:
                              for y_adj in [-1,0,1]:
                                   ni,nj = x_adj+cx,y_adj+cy
                                   shift = abs(x_adj)+abs(y_adj)
                                   if ni < 0 or ni > (new_grid.shape[0]-1) or nj < 0 or nj > (new_grid.shape[1]-1):
                                        continue
                                   if pop_mask[ni,nj] == True and shift < 2:
                                        allowed_xy[ni,nj] = True
                    allowed_and_new = np.logical_and(allowed_xy,np.invert(include))
                    candidates = list((pt[0],pt[1],datain[pt[0],pt[1]]) for pt in np.argwhere(allowed_and_new))
                    if len(candidates) == 0:
                         #print "Giving Up",popind,cumulant
                         break
                    else:
                         candidates.sort(key = lambda x: x[2]/(((abs(x[0]-gmax[0]))**2)+(abs(x[1]-gmax[1]))**2))
                         include[candidates[-1][0],candidates[-1][1]] = True
                         cumulant = np.nansum(datain[include])

               i_include = list((pt[0],pt[1]) for pt in np.argwhere(include))
               for i in range(new_grid.shape[0]):
                    for j in range(new_grid.shape[1]):
                         if (i,j) in i_include:
                              new_grid[i,j] = popind+1
          return new_grid


     def make_plot(self,prob2d,pick2d,prior=None):
          kdedict = self.kdedict
          resid_names = kdedict['populations']
          features = ('score','cscore')
          numfeat = len(features)
          if prior is None: #default flat
               prior = self.flat_prior
          x_stat = features[0]
          y_stat = features[1]
          xcenters = kdedict[x_stat]['xcent']
          ycenters = kdedict[y_stat]['xcent']
          xbins = len(xcenters)
          ybins = len(ycenters) 
          gridplot = plt.figure(figsize=(32,32))
          sub=gridplot.add_subplot(1,1,1)
          sub.set_title("PProbe_Scores")
          sub.set_xlabel('Electron Density Score')
          sub.set_ylabel('Contact Score')
          cmap = mpl.colors.ListedColormap(['gray','royalblue','orangered','g','yellow','gray'])
          bounds=[0,1,2,3,4,5,6]
          norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
          sub.pcolormesh(np.array(xcenters),np.array(ycenters),pick2d.T,cmap=cmap,edgecolors='None',norm=norm)
          grid= self.pop_lasso()
          mask = np.invert(np.logical_or(grid==1,grid==2))
          masked_in = np.ma.masked_where(mask,grid)
          cmap2 = mpl.colors.ListedColormap(['gray','blue','red','g','yellow','gray']) 
          norm2 = mpl.colors.BoundaryNorm(bounds, cmap2.N)
          sub.pcolormesh(np.array(xcenters),np.array(ycenters),masked_in.T,cmap=cmap2,edgecolors='None',norm=norm2)
          sub.axhline(lw=5,c='k')
          sub.axvline(lw=5,c='k')
          return sub



     def plot_kde(self,prior=None,datain=None,outstr=""):
          print "PLOTTING KDE GRID"
          if prior is None:
               prior = self.dir_prior
          prob2d,pick2d = self.gen_plot_hist(prior=prior)
          sub = self.make_plot(prob2d,pick2d)
          plt.savefig("KDE_grid2"+outstr+".png")
          plt.clf()
          plt.close()


     def plot_data_on_grid(self,score,cscore,batches,prior=None,outstr="test"):
          print "PLOTTING 2D DATA GRID"
          prob2d,pick2d = self.gen_plot_hist(prior=prior)
          sub = self.make_plot(prob2d,pick2d)
          """
          kdedict = self.kdedict
          resid_names = kdedict['populations']
          features = ('score','cscore')
          x_stat = features[0]
          y_stat = features[1]
          xcenters = kdedict[x_stat]['xcent']
          ycenters = kdedict[y_stat]['xcent']
          prob2d,pick2d = self.gen_plot_hist(prior=prior)
          gridplot = plt.figure(figsize=(32,32))
          sub=gridplot.add_subplot(1,1,1)
          sub.set_title("PProbe_Scores")
          sub.set_xlabel('Electron Density Score')
          sub.set_ylabel('Contact Score')
          sub.pcolormesh(np.array(xcenters),np.array(ycenters),pick1d.T,cmap=cm.Set1)
          sub.contour(np.array(xcenters),np.array(ycenters),new_grid.T,levels=[0],colors=['black',],linewidths=5.0,linestyles='dotted')
          """
          goodb = batches > 0
          batches = batches[goodb]
          score=score[goodb]
          cscore=cscore[goodb]
          batch_ind = list(set(batches))
          batch_ind.sort()
          normc = mpl.colors.Normalize(vmin=batch_ind[0],vmax=batch_ind[-1])
          cmapper = cm.ScalarMappable(norm=normc,cmap=cm.cool)
          pno = np.arange(0,score.shape[0],1)+1
          for bi in batch_ind:
               sel = batches == bi
               d1 = score[sel]
               d2 = cscore[sel]
               pno1 = pno[sel]
               sub.scatter(d1,d2,s=300,marker="$%s$" % bi ,cmap=cmapper)
          plt.savefig(outstr+"_hist_plot.png")
          plt.clf()
          plt.close()
