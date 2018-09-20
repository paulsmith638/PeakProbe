from __future__ import division
#generic imports
import sys,math,ast,copy
import numpy as np
from PProbe_util import Util
from PProbe_dataio import DataIO
from PProbe_graph import Graph
import cProfile

global_null_contact={"name":"null","chain":"X","element":"H","distance":6.01,
                     "coord":(999.9,999.9,999.9),"resname":"NUL","altloc":"","resid":0,
                     "model":5,"special":False,"unat":hash('XXX'),"unal":hash('XXX'),"unrg":hash('XXX'),
                     "s_name":"null","s_chain":"X","s_element":"H",
                     "s_coord":(999.9,999.9,999.9),"s_resname":"NUL","s_altloc":"","s_resid":0,
                     "s_model":5,"ctype":"null","s_unat":hash('XXX'),"s_unal":hash('XXX'),"s_unrg":hash('XXX'),
                     "s_resat":"None","resat":"None","sym":0}

class Contacts:
      def __init__(self,phenix_python=False):
            global iotbx,CctbxHelpers,flex
            self.null_contact=global_null_contact
            if phenix_python:
                  from PProbe_cctbx import CctbxHelpers
                  import iotbx.pdb
                  from scitbx.array_family import flex

      def do_cprofile(func):
            def profiled_func(*args, **kwargs):
                  profile = cProfile.Profile()
                  try:
                        profile.enable()
                        result = func(*args, **kwargs)
                        profile.disable()
                        return result
                  finally:
                        profile.print_stats()
            return profiled_func
      

      def charge(self,contact_list,cutoff=6.0):
      #no longer "charge", but rather a probabilistic evaluation of the local environment
      #all contacts within 5A of a S/P of a sulfate phosphate or a water with 5sig fofc density
      #were analyzed by atom type (e.g. ALA-O).  The ratio of number of contacts per so4/po4 to
      #to water, take ln(ratio) to give "logodds" ratios below (all empirical)
      #summed to give a pseudo-probability, not weighted or normalized to number of contacts
      #however, gives ~80% F1 score upon logistic regression alone
            logodds = ast.literal_eval("{'TYR-O': -0.67786107398924988, 'TYR-N': -0.71797717689991158, 'TYR-C': -0.79323045006300019, 'VAL-CG2': -0.68590616759993006, 'VAL-CG1': -0.83778689043800103, 'GLN-CD': 0.047138105334724546, 'MET-CA': -0.69989273447396327, 'MET-CB': -0.57795947868708553, 'MET-CE': -0.65496713394624895, 'SER-N': 0.19729240735328024, 'SER-O': 0.0079501788120935925, 'GLU-OE2': -0.052159869793746996, 'SER-C': 0.15376376901351138, 'LEU-CD1': -0.74903334841028824, 'LEU-CD2': -0.79243434488319309, 'TRP-NE1': 0.2132272393979914, 'GLY-CA': 0.27174669707255222, 'GLU-OE1': -0.066802325156775366, 'THR-OG1': 0.37524479109275677, 'ALA-N': -0.48889023896354522, 'ALA-O': -0.57386887567057632, 'PHE-O': -0.31170353820547592, 'PHE-N': -0.62064702755027967, 'PHE-C': -0.50721253316779713, 'THR-CG2': 0.20892985310636053, 'MG-MG': 1.6405756764223838, 'ALA-C': -0.62481838824558167, 'ASP-OD2': -0.27885938761433948, 'ASP-OD1': -0.38935888372627647, 'THR-CA': 0.19768579634556746, 'THR-CB': 0.26596174177565163, 'LYS-N': -0.21474961880645188, 'LYS-O': -0.26536141187364576, 'TYR-OH': 0.37705081115562161, 'TRP-CD1': 0.097747554216357932, 'TRP-CD2': -0.25621091956225978, 'PHE-CD2': -0.65206768783203295, 'ILE-O': -0.73515229294515116, 'TRP-N': -0.73440533685311082, 'TRP-O': -0.80112702902762156, 'MET-SD': -0.81682432757578294, 'ILE-CD1': -0.74115378611022209, 'TRP-C': -0.9669332954579487, 'GLU-CD': -0.32415004573912576, 'ASP-CB': -0.57895854331580421, 'GLU-CG': -0.37069132442173242, 'GLU-CA': -0.58901565066134343, 'GLU-CB': -0.45067065107488757, 'ILE-CG2': -0.89738553632150597, 'LYS-NZ': 0.99970552164074633, 'ILE-C': -0.93430368305205447, 'LYS-CD': 0.69768534970816221, 'LYS-CE': 0.90829134633992625, 'TRP-CE2': 0.11216134471741197, 'TRP-CE3': -0.70036219968347757, 'LYS-CA': -0.22104382240319378, 'MET-CG': -0.70250077792639876, 'ILE-N': -1.1115477566129921, 'HIS-CD2': 1.1202713087018494, 'ASP-O': -0.80657005548211491, 'TRP-CZ3': -0.35273671152751324, 'TRP-CZ2': 0.2797404061352804, 'HIS-NE2': 1.1898479607295638, 'VAL-C': -0.69601907051219081, 'GLN-CA': -0.5956334595900209, 'TYR-CD2': -0.3063547890194982, 'GLN-CB': -0.45724520582535016, 'VAL-O': -0.49996682586969549, 'VAL-N': -0.76384259171151969, 'GLN-CG': -0.27203022641737817, 'TYR-CD1': -0.27505456643217502, 'HIS-ND1': 1.0577503288935974, 'ILE-CB': -1.0443886106702198, 'ILE-CA': -1.1670314298923123, 'HIS-CE1': 1.1779809326949156, 'THR-C': 0.078093604428505881, 'THR-N': 0.1828826376805775, 'THR-O': -0.018082219537221472, 'GLU-C': -0.56187575175060045, 'HIS-CB': 0.68284138158363772, 'HIS-CA': 0.58349128716501975, 'ALA-CB': -0.3692315094525318, 'HIS-CG': 1.0072663545262404, 'GLU-N': -0.50597624337167135, 'GLU-O': -0.6261974013460424, 'GOL-C1': 0.78196405613490683, 'GOL-C2': 0.84370866966207214, 'GOL-C3': 0.75168391873634322, 'SER-OG': 0.47417333475918189, 'ASN-OD1': 0.11585627152217609, 'LYS-CG': 0.32597106626927669, 'PRO-CD': -0.68390239418189869, 'GLN-NE2': 0.11252978530411321, 'PRO-CA': -0.5010224365772834, 'PRO-CB': -0.36744500449693923, 'PRO-O': -0.47186131944288767, 'PRO-N': -0.66707766544470259, 'ASN-CG': 0.11765125262563975, 'ARG-NE': 1.2587494508525419, 'ASN-CA': -0.35898177444522039, 'ASN-CB': -0.14815872567598201, 'ILE-CG1': -0.81175480293100366, 'LYS-CB': 0.041097944074319499, 'CYS-SG': 0.045419153471840494, 'ASN-ND2': 0.22926641662671951, 'TYR-CA': -0.64970281727339541, 'TYR-CB': -0.57557968857327413, 'TYR-CG': -0.66830534502847649, 'ARG-NH2': 1.3876550285150921, 'ARG-NH1': 1.359234698076476, 'GOL-O1': 0.70312671904088264, 'TYR-CZ': 0.1380472952247066, 'ASN-C': -0.40729669478513003, 'GLN-OE1': 0.048225158638201322, 'ASN-N': -0.43046637583747571, 'ASN-O': -0.3746731554832301, 'ARG-C': -0.13465324883366792, 'TRP-CH2': -0.015796566220871971, 'ARG-N': -0.083192115133338615, 'ARG-O': -0.11298290374293037, 'LEU-C': -0.84890815132610176, 'ASP-C': -0.76489454626484299, 'LEU-N': -0.80830072083145255, 'LEU-O': -0.71124885285537465, 'VAL-CA': -1.0814528272796329, 'CYS-CB': -0.34829083373417619, 'CYS-CA': -0.34567826518344363, 'ASP-CA': -0.79247355567929811, 'ASP-N': -0.67375587417269489, 'ASP-CG': -0.54702169817233592, 'PRO-C': -0.46355847784782356, 'HIS-N': 0.4325689819955118, 'LEU-CG': -0.84839677087186149, 'LEU-CA': -0.91355093297246814, 'LEU-CB': -0.86612137274775625, 'GLN-N': -0.42294722306114485, 'ALA-CA': -0.51288154817664755, 'SER-CA': 0.34189492055482895, 'SER-CB': 0.41471113489629052, 'LYS-C': -0.2348287415533476, 'PHE-CE1': -0.44645603452412769, 'PHE-CE2': -0.52718582541488523, 'PRO-CG': -0.33589791538620911, 'PHE-CZ': -0.50120619821795198, 'ARG-CZ': 1.3511232057791813, 'PHE-CA': -0.52346663441039432, 'ARG-CG': 0.72404553418061657, 'ARG-CD': 0.74650739348774808, 'PHE-CB': -0.39590728813637643, 'ARG-CB': 0.37071164417332486, 'PHE-CG': -0.50417340301981428, 'ARG-CA': -0.00736515291506026, 'CYS-O': 0.18265893927651713, 'CYS-N': -0.29322964492029857, 'CYS-C': 0.24430359619340775, 'TYR-CE1': 0.10957199631572699, 'TYR-CE2': 0.15265371986839499, 'GLY-C': 0.26313854604738274, 'GLY-N': 0.26977223228743358, 'GLY-O': 0.20252296951838289, 'MET-N': -0.5973950487959564, 'GOL-O2': 0.82176271404386669, 'GOL-O3': 0.70015596174853434, 'GLN-C': -0.74115278584064936, 'HIS-C': 0.42131056172470238, 'MET-O': -0.72454593503855991, 'GLN-O': -0.70251790369125588, 'HIS-O': 0.47964770720964939, 'PHE-CD1': -0.59937384558017659, 'MET-C': -0.63804642280220303, 'TRP-CA': -0.68608279575318176, 'TRP-CB': -0.49827510364447897, 'VAL-CB': -0.86303105079860087, 'TRP-CG': -0.088296478881204732}")
            running_prob = 0.0
            counted_unal = []
            for contact in contact_list:
                  if contact['unal'] in counted_unal or contact['model'] > 2:
                        continue
                  name = contact['name'].strip()
                  resn = contact['resname'].strip()
                  dist = contact['distance']
                  cont_id = resn+"-"+name
                  counted_unal.append(contact['unal']) # in case duplicated
                  if logodds.has_key(cont_id) and float(dist) < cutoff:
                        running_prob = running_prob + logodds[cont_id]
            return running_prob
	
      def fmt_cont(self,contact):
            if contact['resname'] != "NUL":
                  return contact['resname']+"_"+contact['chain']+str(contact['resid'])+"_"+contact['name']
            else:
                  return "None"


      def cluster_modsol(self,all_peak_db):
            #finds all unal of the same residue group for input solvent and assigns to clusters
            all_sol = list(pdict['unal'] for pdict in all_peak_db.values() if pdict['model'] == 3)
            sol_clust_db = {}
            sol_unrg = set(all_peak_db[unal]['unrg'] for unal in all_sol)
            for unrg in sol_unrg:
                  unrg_mem = sol_clust_db.get(unrg,[])
                  unrg_list = list(unal for unal in all_sol if all_peak_db[unal]['unrg'] == unrg)
                  unrg_mem.extend(unrg_list)
                  sol_clust_db[unrg] = unrg_mem
            for unal in all_sol:
                  pdict = all_peak_db[unal]
                  unrg = pdict['unrg']
                  clust_mem = sol_clust_db[unrg]
                  if len(clust_mem) > 1:
                        pdict['clust_mem'] = clust_mem
                        pdict['clust_rank'] = -1
                        pdict['clust_score'] = -1

                  

      #@do_cprofile
      def cluster_analysis(self,all_peak_db):
            print "-"*79
            print "CLUSTER ANALYSIS:"
            #filter peaks, then feed to clustering routine
            #clustering is slow, so prefilter peaks to reduce input size?
            all_peak = list(pdict['unal'] for pdict in all_peak_db.values() if pdict['model'] == 4)
            #pnw_cut = 0.05
            #map_2fofc_cut = 0.3
            #map_fofc_cut = 3.0
            excluded_status = [-1,1,3,7]
            excluded_fc = [-1,2,3,4,5,7]
            exclude = []
            ex_count = [0,0,0,0]
            for unal in all_peak:
                  pdict = all_peak_db[unal]
                  status = pdict.get('status',-1)
                  fc = pdict.get('fc',-1)
                  if status in excluded_status:
                        exclude.append(unal)
                        ex_count[0] = ex_count[0] + 1
                        continue
                  if any([pdict.get('fofc_sigo_scaled',None),pdict.get('2fofc_sigo_scaled',None)]) is None:
                        exclude.append(unal)
                        ex_count[1] = ex_count[1] + 1
                        continue
                  if fc in excluded_fc:
                        exclude.append(unal)
                        ex_count[2] = ex_count[2] + 1
                        continue
                  clp_dist = pdict.get('peak_contacts',[{"distance":6.01},])[0]['distance']
                  if clp_dist > 3.8:
                        exclude.append(unal)
                        ex_count[3] = ex_count[3] + 1
                        continue
            exclude = list(set(exclude))
            print "  --> analyzing %s peaks (%s excluded)" % (len(all_peak)-len(exclude),len(exclude))
            #get actual phylogeny
            n_clust = self.peak_phylo(all_peak_db,omit_unal=exclude)
            #also cluster input solvent
            self.cluster_modsol(all_peak_db)
            # assign mf flag accordingly
            assigned_mf = []
            for unal,pdict in all_peak_db.iteritems():
                  if 'clust_mem' not in pdict:
                        pdict['mf'] = 0 #N/A 
                        continue
                  if pdict['clust_rank'] < 0:
                        if pdict['model'] == 4:
                              pdict['mf'] = 0 
                        else: #for existing model solvent
                              pdict['mf'] = np.clip(len(pdict['clust_mem']),0,8).astype(np.int)*-1
                        continue
                        
                  mfclus,mfrank,mfscore = 0,0,0
                  mfclust = np.clip(len(pdict['clust_mem']),0,9).astype(np.int)
                  if mfclust != 0:
                        mfrank = np.clip(pdict['clust_rank'],-1,8).astype(np.int)+1
                        mfscore = np.clip(int(pdict['clust_score']) / 5, 0,9).astype(np.int)
                  mf = int(100*mfclust+10*mfrank+mfscore)
                  if mfrank == 1:
                        assigned_mf.append(mf)
                  pdict['mf'] = mf
            clust_counts = []
            count_labels = []
            for i in np.arange(2,10,1):
                  clust_counts.append(list(int(mf/100) for mf in assigned_mf).count(i))
                  count_labels.append(str(i))
            count_labels[-1] = count_labels[-1]+"+"
            tally_out = " ".join(list(count_labels[i]+"="+str(clust_counts[i]) for i in range(len(clust_counts))))
            print "FOUND %d possible clusters --> breakdown [size=count]: %s" %(n_clust, tally_out)
            print "-"*79
            
                  
            

      def peak_accum(self,score_mat,path_mat,peak_phylo,sind,cind,cutoff=1.3,cut_ratio=0.75):
            #takes a pair of peaks, finds smallest common cluster from divisive tree
            #and poaches members from this cluster if above cutoffs
            #cutoff is absolute score value (1.3 used for dwt_sim)
            #cut_ratio is fraction of score of input pair to include (any, single linkage)
            conn_w = score_mat[sind,cind]
            if conn_w < cutoff:
                  return []
            phylo1 = peak_phylo[sind]
            phylo2 = peak_phylo[cind]
            conn_cut = np.amax((cut_ratio*conn_w,cutoff))
            #find smallest common subcluster containing parent(sind) and child (cind)
            common = sorted(list(set(phylo1) & set(phylo2)))[::-1]
            gc_cluster = common[0]
            gc_members = path_mat[gc_cluster] == 1
            # add members if above cutoff (single-linkage poaching)
            gc_cut = np.logical_or(np.logical_and(score_mat[sind] > conn_cut,gc_members),np.logical_and(score_mat[cind] > conn_cut,gc_members))
            gc_cut[[sind,cind]] = 0
            gc_cull = np.nonzero(gc_cut)[0]
            clust_accum = [sind,cind]
            for member in gc_cull:
                  if member not in clust_accum:
                        clust_accum.append(member)
            return clust_accum
 
      #@do_cprofile
      def peak_phylo(self,all_peak_db,omit_unal=[],clust_cut=1.3):
            #clust cut is cutoff for similarity in agglomeration
            ppgraph = Graph()
            #collect peaks for analysis
            feat_list = list(pdict  for pdict in all_peak_db.values() if (pdict['model'] == 4 and pdict['unal'] not in omit_unal))
            unal_list = list(pdict['unal'] for pdict in feat_list)
            if len(feat_list) > 0:
                  cont_db = feat_list[0]['cont_db']
            else:
                  return 0

            #generate multi_layer adjacency tensor and flattened supra-adjacency matrix
            cont_mlat,idict,sam = ppgraph.get_mlat(cont_db,unal_list)
            no_peaks = cont_mlat.shape[0]
            no_layer = cont_mlat.shape[2]
            all_feat_list = list(all_peak_db[idict[i]] for i in range(no_peaks))
            #setup adjacency tensors/matrices
            mlat_adj = cont_mlat < 6.01
            full_dmask = mlat_adj == 0

            for h in range(no_layer):#no self loops
                  np.fill_diagonal(full_dmask[:,:,h,h],0)
            mlat_adj[full_dmask] = 0
            sam_adj = sam < 6.01
            np.fill_diagonal(sam_adj,0)

            #print hop_sam[0:10,0:10]
            red_dist = np.zeros((no_peaks,no_peaks))+1E20
            red_hops = np.zeros((no_peaks,no_peaks))+1E20
            #uses Dikjstra's algorithm, once with distances, once with adjacency
            t_dist,t_path,t_hops = ppgraph.graph_dsp(sam,list(range(no_peaks)))
            #fold all distances and hopes back into ASU, omit self 0.0 distance (but allow self by sim)
            for i in range(no_peaks):
                  for j in range(no_peaks):
                        distances = list(t_dist[i].get(k,np.inf) for k in range(sam.shape[0]) if (k % no_peaks == j and k != i))
                        distances.extend(list(t_dist[j].get(k,np.inf) for k in range(sam.shape[0]) if (k % no_peaks == i and k != j)))
                        #print i,j,distances
                        hops = list(t_hops[i].get(k,np.inf) for k in range(sam.shape[0]) if ( k % no_peaks == j and k != i))
                        hops.extend(list(t_hops[j].get(k,np.inf) for k in range(sam.shape[0]) if ( k % no_peaks == i and k != j)))
                        #print i,j,hops
                        red_dist[i,j] = np.amin(distances)
                        red_hops[i,j] = np.amin(hops)

            #special positions also turn up here
            special = np.diag(red_dist) < 1.3 #bool array
            peak_cont_sort = np.argsort(red_dist,axis=1)
            min_cont_d = np.amin(red_dist,axis=1)
            #reduced adjacency matrix should take into account all found sym-mates
            red_adj_mat = np.logical_and(red_hops > 0.5,red_hops < 1.5) #avoid float roundoff
            red_dmask = np.invert(red_adj_mat)

            #data for distance metrics for clustering, density score and level
            all_score = list(pdict['score'] for pdict in all_feat_list)
            all_densig = list(pdict['2fofc_sigo_scaled'] for pdict in all_feat_list)
            #setup scoring matrices
            trunc_mdist = red_dist.copy()
            invalid = np.isinf(trunc_mdist) #remove invalid distances (disjoint peaks)
            trunc_mdist[invalid] = 999.0
            #pseudo inverse distance used for harmonic centrality
            inv_dist = 1.0/(np.clip(trunc_mdist,1.3,999.0)) #clipped to avoid special position jump
            inv_dist[red_dmask] = 0.0 #non-adjacent set to zero
            np.fill_diagonal(inv_dist,0.0) #self also zero

            min_dsq_mat = np.square(trunc_mdist) # squared distance matrix
            #squared distances binned, similarity assigned
            #rbf function didn't capture the point, so coarsely binned
            d_bin1 = min_dsq_mat < 4.1
            d_bin2 = np.logical_and(min_dsq_mat >=4.1,min_dsq_mat<6.0)
            d_bin3 = np.logical_and(min_dsq_mat >=6.0,min_dsq_mat<12.0)
            d_bin4 = np.logical_and(min_dsq_mat >=12.0,min_dsq_mat<36.0)
            d_bin5 = min_dsq_mat >=36.0
            dsq_sim = np.zeros(min_dsq_mat.shape)
            dsq_sim[d_bin1] = 1.0
            dsq_sim[d_bin2] = 0.8
            dsq_sim[d_bin3] = 0.4
            dsq_sim[d_bin4] = 0.2
            #feature similarities assigned by radial basis functions
            score_sim = ppgraph.simsq_rbf(all_score,red_dmask,calc_sim=True,norm=True)
            densig_sim = ppgraph.simsq_rbf(all_densig,red_dmask,calc_sim=True,norm=True)
            #pairwise edge weights
            #weight dsq by average density 
            den_weight = np.add(np.array(all_densig)[:,None],np.array(all_densig)[None,:])/2.0
            dwt_sim = np.multiply(np.multiply(dsq_sim,densig_sim),den_weight)

            #multilayer degree centrality
            #dcv = np.einsum("abgd,gd,b",mlat_adj,np.ones(((no_layer,no_layer))),np.ones(no_peaks))
            #not used, only helpful with peaks for every solvent atom present
            #local eigencentrality, also not very helpful
            #peak_cent = ppgraph.local_eigcent(red_adj_mat,peak_cont_sort)
            

            #peak hierarchy by divisive clustering (paths are really subgraphs)
            n_nodes = dwt_sim.shape[0]
            n_edges = np.nansum(dwt_sim > 0.0)
            if_complete = float((n_nodes*(n_nodes-1))/2)
            print "      DivClust on %s nodes, av degree = %5.4f" % (dwt_sim.shape[0],n_edges/if_complete)
            paths = ppgraph.div_cluster(dwt_sim)
            path_matrix = np.zeros((len(paths),no_peaks),dtype=np.int16)
            pm_phylo = {}
            #populate path matrix by strided index from list of lists
            for pind,path in enumerate(paths):
                  path_matrix[pind,path] = 1
            for pind in range(no_peaks):
                  pm_phylo[pind] = list(np.nonzero(path_matrix[:,pind] == 1)[0])
            path_count = np.nansum(path_matrix,axis=1)# number of members at each cluster level

            #rank by pairwise score, numpy output is convoluted, convert to culled list
            cluster_rank = np.dstack(np.unravel_index(np.argsort(dwt_sim,axis=None)[::-1],dwt_sim.shape))
            #accumulate clusters, start with most strongly linked peak pairs, initial cutoff = 1.3
            cluster_rank = list([i,j] for i,j in cluster_rank[0] if (i < j and dwt_sim[i,j] > clust_cut))
            clusters = []
            #quick function for calculating mean edge weights from density weighted similarity
            clust_mean = ppgraph.clust_mean
            #sind and cind are "source index" and "contact index", values are matrix indices
            #grow clusters from strongly linked pairs using hierarchy 
            #not true agglomerative, more accumulative as vertices are "poached" from other clusters
            #as clusters are assembled
            for sind,cind in cluster_rank:
                  new_clust = self.peak_accum(dwt_sim,path_matrix,pm_phylo,sind,cind,cutoff=1.3)
                  if len(new_clust) > 1:
                        grew = True
                  else:
                        grew = False
                  while grew:
                        start = len(new_clust)
                        pair_list = ppgraph.pairind_gen(new_clust)
                        for sind2,cind2 in pair_list:
                              test_clust = self.peak_accum(dwt_sim,path_matrix,pm_phylo,sind2,cind2,cutoff=1.3)
                              for peak in test_clust:
                                    if peak not in new_clust:
                                          new_clust.append(peak)
                        if len(new_clust) == start:
                              grew = False
                  if len(new_clust) > 0:
                        clusters.append(sorted(list(set(new_clust))))
            #next, peaks in better clusters are removed from worse, score is total dwt_sim edges
            #empirically, logistic regression puts midpoint at 6.5 with 92% ppv for 
            #separating water clusters from non water clusters
            if len(clusters) > 1:
                  clusters.sort(key=lambda clust: len(clust)*clust_mean(clust,dwt_sim),reverse=True)
                  for i in range(len(clusters)):
                        for peak in clusters[i]:
                              for cluster in clusters[i+1::]:
                                    if peak in cluster:
                                          cluster.remove(peak)
                        clusters.sort(key=lambda clust: len(clust)*clust_mean(clust,dwt_sim),reverse=True)
                  clusters.sort(key=lambda clust: len(clust)*clust_mean(clust,dwt_sim),reverse=True)
            clusters = list(clust for clust in clusters if len(clust) > 1)
            #cluster matrix is i=peak, j=cluster number, i,j=centrality
            #cluster centrality is degree centrality from dwt_sim for only cluster members
            clust_mat = np.zeros((no_peaks,len(clusters)))
            for clustind,clust in enumerate(clusters):
                  for peak in clust:
                        clust_mat[peak,clustind]=np.nansum(dwt_sim[peak,clust])
            #lastly, store scores and cluster/pair members in main dictionary
            for i in range(no_peaks):
                  unal = idict[i]
                  peak_dwt_max = np.amax(dwt_sim[i,:])
                  peak_dwt_un = all_peak_db[idict[np.argmax(dwt_sim[i,:])]]['unal']
                  all_peak_db[unal]['clust_pair'] = [peak_dwt_un,peak_dwt_max]
                  clust_ind = np.nonzero(clust_mat[i] > 0)[0]
                  red_clust = np.nonzero(clust_mat[:,clust_ind])[0]
                  if len(red_clust) > 1:
                        clust_ind = clust_ind[0] #why does np add dimensions?
                        clust_score = len(red_clust)*clust_mean(red_clust,dwt_sim)
                        clust_cent = np.nansum(dwt_sim[i,red_clust])
                        clust_rank = np.nonzero(np.argsort(clust_mat[:,clust_ind])[::-1] == i)[0]
                        if len(clust_rank) > 0:
                              clust_rank = clust_rank[0]
                        else:
                              clust_rank = -1
                  else:
                        clust_score = 0.0
                        clust_cent = 0.0
                        clust_rank = -1
                  all_peak_db[unal]['clust_mem'] = sorted(list(idict[x] for x in red_clust))
                  all_peak_db[unal]['clust_score'] = clust_score
                  all_peak_db[unal]['clust_cent'] = clust_cent
                  all_peak_db[unal]['clust_rank'] = clust_rank
            #debugging
            #for peak,db in all_peak_db.iteritems():
            #      if peak in unal_list:
            #            if 'clust_mem' in db:
            #                  print "CLUST",db['resat'],all_peak_db[db['clust_pair'][0]]['resat'],len(db['clust_mem']),db['clust_score'],db['clust_cent'],db['clust_rank']
            return clust_mat.shape[1]


      def bipartite_search(self,all_peak_db,omit_unal=[]):
            #find peak to macromolecule bridges by shortest path approach
            #working, but not finished or used at the moment
            ppgraph = Graph()
            #collect peaks for analysis
            peak_list = list(pdict for pdict in all_peak_db.values() if (pdict['model'] == 4 and pdict['unal'] not in omit_unal))
            sol_list = []
            if len(peak_list) > 0:
                  cont_db = peak_list[0]['cont_db']
                  for pind,peak in enumerate(peak_list):
                        sol_conts = peak['sol_contacts']
                        for scont in sol_conts:
                              sol_list.append(scont['unal'])

            else:
                  return 0
            #uniquify and collect dictionaries
            sol_list = list(all_peak_db[unal] for unal in list(set(sol_list)))
            total_ulist = list(pdict['unal'] for pdict in peak_list) + list(pdict['unal'] for pdict in sol_list)
            total_ulist = list(set(total_ulist))
            pi_dict,ip_dict = {},{}
            for pind,peak in enumerate(total_ulist):
                  pi_dict[peak] = pind + 1
                  ip_dict[pind + 1] = peak
            

            # bipartite graph with extra r/c for "mm" (pseudotripartite?)
            s2p_db = {}
            dmat = np.ones((len(total_ulist)+1,len(total_ulist)+1)) * np.inf
            for unal in total_ulist:
                  peak = all_peak_db[unal]
                  mm_conts = peak['mm_contacts']
                  if peak['model'] == 3:
                        in_conts = peak.get('peak_contacts',[])
                        if len(mm_conts)>0:
                              dmat[pi_dict[unal],0] = mm_conts[0]['distance']
                  if peak['model'] == 4:
                        in_conts = peak.get('sol_contacts',[])
                        if len(mm_conts)>0:
                              dmat[0,pi_dict[unal]] = mm_conts[0]['distance']

                  for cont in in_conts:
                        s_mod = cont['s_model']
                        sunal = cont['s_unal']
                        c_mod = cont['model']
                        cunal = cont['unal']
                        if s_mod == 3 and c_mod == 4:
                              cpair = (sunal,cunal)
                        elif s_mod == 4 and c_mod == 3:
                              cpair = (cunal,sunal)
                        else:
                              continue
                        pair_dist = s2p_db.get(cpair,[])
                        pair_dist.append(cont['distance'])
                        s2p_db[cpair] = pair_dist
            for pair,dlist in s2p_db.iteritems():
                  unal1,unal2 = pair
                  if unal1 not in pi_dict or unal2 not in pi_dict:
                        continue
                  if unal1 != unal2 and len(dlist) > 0:
                        i = pi_dict[unal1]
                        j = pi_dict[unal2]
                        d1 = np.amin(dlist)
                        d2 = dmat[i,j]
                        dmat[i,j] = np.amin((d1,d2))
                        dmat[j,i] = np.amin((d1,d2))
            
                  
            dmat[0,0] = 0
            dmat_check = np.zeros((dmat.shape[0],dmat.shape[1],2))
            dmat_check[:,:,0] = dmat
            dmat_check[:,:,1] = dmat.T
            dmat = np.amin(dmat_check,axis=2)
            #iteratively cut edges and find paths to macromolecule
            for pind in np.arange(1,dmat.shape[0],1):
                  origd = dmat[0,pind]
                  bridged = True
                  one_hops = []
                  to_remove = [pind,]
                  while bridged:
                        hop_mat = dmat.copy()
                        hop_mat[dmat == 0.0] = 10.0
                        hop_mat[0,:] = 0.0
                        hop_mat[:,0] = 0.0
                        hop_mat[0,to_remove] = np.inf
                        hop_mat[to_remove,0] = np.inf
                  
                        t_hops,t_hpth = ppgraph.graph_dsp(hop_mat,[0],adj_cut=4.5)
                        bridge_i = t_hpth[0].get(pind,None)
                        if bridge_i is not None:
                              bridge_d = t_hops[0].get(bridge_i,99.99)
                              if bridge_d > 6.0:
                                    bridged=False
                              if bridge_i not in to_remove:
                                    if hop_mat[bridge_i,0] == 0.0:
                                          one_hops.append(bridge_i)
                                    to_remove.append(bridge_i)
                              else:
                                    bridged = False
                  #for i in one_hops:
                  #      print "ONEHOP",i,all_peak_db[ip_dict[pind]]['resat'],all_peak_db[ip_dict[i]]['resat']
                        
                        


      @classmethod
      def prune_cont(cont_obj,cont_list,omit_unat=[],omit_unal=[],omit_models=[],omit_unrg=[],cutoff=6.0,uniquify=False,unires=False,omit_null=False):
            #taskes a contact list of dictionaries
            #returns pruned list to by specified identifiers
            # uniquify adds each contact only once (may be multiple by symmetry)
            # unires gives only closest contact per residue
            cont_list.sort(key = lambda x: x['distance'])
            if len(cont_list) == 0:
                  return cont_list
            if cont_list[0]['ctype'] == "null":
                  if omit_null:
                        return []
                  else:
                        return cont_list
            omit_list= []
            pruned = []
            added_unat = []
            for index,cont in enumerate(cont_list):
                  if int(cont['model']) in omit_models:
                        omit_list.append(index)
                        continue
                  if cont['unat'] in omit_unat:
                        omit_list.append(index)
                        continue 
                  if cont['unal'] in omit_unal:
                        omit_list.append(index)
                        continue 
                  if cont['unrg'] in omit_unrg:
                        omit_list.append(index)
                        continue 
                  if cont['distance'] > cutoff:
                        omit_list.append(index)
                        continue
                  if cont['resname'] == "NUL" and omit_null:
                        omit_list.append(index)
                        continue 

            for i in range(len(cont_list)):
                  if i not in omit_list:
                        if uniquify and cont_list[i]['unat'] not in added_unat:
                              pruned.append(cont_list[i])
                              #sorted, only shortest will be added
                              added_unat.append(cont_list[i]['unat'])
                        else:
                              pruned.append(cont_list[i])

            if len(pruned) == 0 and not omit_null:
                  pruned.append(global_null_contact)


            pruned.sort(key = lambda x: x['distance'])

            if unires:
                  omit_rg = []
                  to_omit = []
                  for index,cont in enumerate(pruned):
                        if cont['unrg'] not in omit_rg:
                              omit_rg.append(cont['unrg'])
                        else:
                              to_omit.append(index)
                  res_pruned = []
                  for index,cont in enumerate(pruned):
                        if index not in to_omit:
                              res_pruned.append(cont)
                  return res_pruned
            return pruned

      def unique_chresid(self,pdb):
            reslist = []
            for model in pdb.models():
                  for chain in model.chains():
                        chainid = chain.id.strip()
                        for resgroups in chain.residue_groups():
                              for atomgroups in resgroups.atom_groups():
                                    for atom in atomgroups.atoms():
                                          resid = resgroups.resseq.strip()
                                          reslist.append((chainid,resid))
            return set(reslist)


      def omited_peaks(self,orig_pdb_hier,strip_pdb_hier):
            #generates hier of all atoms in orig but not in strip
            orig_resid = self.unique_chresid(orig_pdb_hier)
            strip_resid = self.unique_chresid(strip_pdb_hier)
            omits = orig_resid - strip_resid
            selstrs = []
            if len(omits) > 0:
                  atom_selection_manager = orig_pdb_hier.atom_selection_cache()
                  for omit in omits:
                        selstrs.append("(chain %2s and resid %4s)" % (omit[0],omit[1]))
                  sel_str = " or ".join(selstrs).format(['"{:25s}"'*len(selstrs)])
                  omit_selection = atom_selection_manager.selection(string = sel_str)
                  omit_hier = orig_pdb_hier.select(omit_selection)
                  return omit_hier
            else:
                  omit_hier = orig_pdb_hier.deep_copy()
                  for model in omit_hier.models():
                        omit_hier.remove_model(model)
                  return omit_hier #hack to return empty hier

      def merge_hier(self,hier_list,symmetry):
            ppcctbx = CctbxHelpers()
            allhier = ppcctbx.merge_hier(hier_list,symmetry)
            return allhier

      def peak_allcont(self,merge_hier,symm):
            ppcctbx = CctbxHelpers()
            contact_db = ppcctbx.contacts_to_all(merge_hier,symm)
            for unal,clist in contact_db.iteritems():
                  if len(clist) == 0:
                        clist.append(self.null_contact)
            return contact_db

      def solvent_peaks(self,orig_pdb_hier):
            #identifies molecules of common solvent in original structure (modelled)
            ppio = DataIO()
            atom_selection_manager = orig_pdb_hier.atom_selection_cache()
            selstrs = []
            for resname in ppio.common_solvent + ppio.common_elem:
                  selstrs.append("(resname '%3s')" % resname)
            sel_str = " or ".join(selstrs).format(['"{:s}"'*len(selstrs)])
            sol_selection = atom_selection_manager.selection(string = sel_str)
            sol_hier = orig_pdb_hier.select(sol_selection)
            print "   Found %s solvent atoms in original structure" % sol_hier.atoms().size()
            return sol_hier

      def parse_peak_contacts(self,peak_contacts,unid):
            local=[]
            strip=[]
            peakc=[]
            pruned = self.prune_cont(peak_contacts,unid)
            for contact in pruned:
                  model = int(contact['model'])
                  s_model = int(contact['s_model'])
                  if model < 3:
                        local.append(contact)
                  if model == 1:
                        strip.append(contact)
                  if model == 4:
                        peakc.append(contact)
            return local,strip,peakc

      def process_contacts(self,pdict,write_contacts=False):
            ppio = DataIO()
            p_unal = pdict['unal']
            p_unat = pdict['unat']
            p_unrg = pdict['unrg']

            #collect, sort, prune contacts
            #if derived from merged hierarchy, no self peaks, will contain null contact (min length = 1)
            #ori = all contacts from a given peak to the original model (models 1 and 2)
            #peak = only contacts to other peaks (model 3)
            #sol = contacts to common solvent (resnames in PPcont), redundant with ori
            #closest_solres is uniquified by residue group 
            #mm_contacts is ori contacts pruned of all common solvent 
            cont_list = pdict['cont_db'][p_unal]
            all_sol_cont  = self.prune_cont(cont_list,omit_models=[1,2,4],uniquify=False)
            all_peak_cont = self.prune_cont(cont_list,omit_models=[1,2,3],uniquify=False)
            all_ori_cont  = self.prune_cont(cont_list,omit_models=[3,4],uniquify=False)
            all_omit_cont = self.prune_cont(all_ori_cont,omit_models=[1,3,4],uniquify=False)
            closest_solres = self.prune_cont(all_sol_cont,omit_models=[1,2,4],uniquify=True,unires=True) 
            #for c1 (essential for cc scoring), get closest macromolecular contact by deleting solvent
            sol_del = []
            for cind,cont in enumerate(all_sol_cont):
                  if cont['distance'] < 6.0:
                        sol_del.append(cont['unat'])
            sol_del.append(p_unat) #remove any solvent peaks to get only protein/mm
            mm_contacts = self.prune_cont(all_ori_cont,omit_unat=sol_del)

            #identify the original model type (for label comparison in scoring)

            if pdict['model'] == 4: # is an inputpeak, unknown, find something close
                  pdict['label'] = 0
                  if all_ori_cont[0]['distance'] < 1.6: #fixed cutoff?
                        pdict['orires'] = all_ori_cont[0]['resname']
                  else:
                        pdict['orires'] = 'XXX' #nothing close
            elif pdict['model'] == 3:
                  pdict['orires'] = pdict['resat'].split("_")[0]
                  if pdict['orires'] == 'HOH':
                        pdict['label'] = 1
                  elif pdict['orires'] in ['SO4','PO4']:
                        pdict['label'] = 2
                  elif pdict['orires'] in ppio.common_oth:
                        pdict['label'] = 3
                  elif pdict['orires'] in ppio.common_met:
                        pdict['label'] = 4
                  else:
                        pdict['label'] = 0
            else:
                  pdict['orires'] = pdict['resat'].split("_")[0]
                  pdict['label'] = 0
            #here, set c1 , might be updated later with new "anchor" contact
            pdict['anchor'] = mm_contacts[0]
            pdict['c1'] = mm_contacts[0]['distance']
            #next, get local probability environment "charge"
            pdict['charge'] = self.charge(all_ori_cont) #scoring includes some common solvent

            pdict['contacts'] = all_ori_cont
            pdict['strip_contacts'] = self.prune_cont(all_ori_cont,omit_models=[2,3,4])
            pdict['peak_contacts'] = all_peak_cont
            pdict['mm_contacts'] = mm_contacts
            pdict['sol_contacts'] = all_sol_cont
            pdict['omit_contacts'] = self.prune_cont(all_ori_cont,omit_models=[1,3,4])
            # initialize pdict values
            pdict['anc_for'] = []
            pdict['mod_for'] = []
            pdict['sol_mod'] = []
            pdict['mflag'] = 0
            if pdict['c1'] < 2.5:
                  close_cont = self.close_cont(mm_contacts,cutoff=2.5)
                  s_cc = sorted(close_cont,key = lambda x: len(x))
                  worst = s_cc[-1]
                  worst.sort(key = lambda x: x['distance'])
                  shortest_worst = worst[0]
            else:
                  shortest_worst = pdict['mm_contacts'][0]
            pdict['worst_mm'] = shortest_worst
            self.find_modsol_anchor(pdict)

      def find_modsol_anchor(self,pdict,omit_unal=[]):
            pput = Util()
            p_unal = pdict['unal']
            cont_db = pdict['cont_db']
            all_cont = pdict['cont_db'][p_unal]
            mm_conts = pdict.get('mm_contacts',None)
            if mm_conts is None:
                  mm1_dict=6.01
            else:
                  mm_cont1 = mm_conts[0] #closest mm contact
                  mm1_dist = mm_cont1['distance']

            input_model = pdict['model']
            all_peak_db = pdict['peak_unal_db']

            # first, get all close solvent and input peaks, unique by residue, within mm contact distance + 0.1
            possible_models_anchors = self.prune_cont(all_cont,omit_unal=omit_unal,omit_models=[1,2],
                                                      uniquify=True,unires=True,cutoff=mm1_dist+0.1,omit_null=True)

            # next, sort by distance (> 1.7 is anchor, otherwise model), models must be cross (peak to sol or sol to peak)

            omit_from_mod = []
            omit_from_anc = []

            for cont in possible_models_anchors:
                  if cont['distance'] >= 1.7:
                        omit_from_mod.append(cont['unal'])
                  else:
                        omit_from_anc.append(cont['unal'])
                  if cont['model'] == input_model:
                        omit_from_mod.append(cont['unal']) #no self-model models
                        
            possible_models = self.prune_cont(possible_models_anchors,omit_unal=omit_from_mod,uniquify=True,unires=True,omit_null=True)
            possible_anchors = self.prune_cont(possible_models_anchors,omit_unal=omit_from_anc,uniquify=True,unires=True,omit_null=True)
            pdict['mod_cont'] = possible_models
            pdict['anc_cont'] = possible_anchors

            #debugging output
            #for cont in possible_models:
            #      cpdict = all_peak_db[cont['unal']]
            #      print "POSMOD",pdict['db_id'],pdict['model'],cont['resat'],cont['distance'],cpdict['model']
            #for cont in possible_anchors:
            #      cpdict = all_peak_db[cont['unal']]
            #      print "POSANC",pdict['db_id'],pdict['model'],cont['resat'],cont['distance'],cpdict['model']
         
      def associate_models(self,all_peak_db,peak_list,sol_list):
            #for input of all peaks, associate model 3 (solvent) with model 4 (peaks)
            #by auctions and associate based on lowest distances


            #start with setting up peaks with pruned solvent contacts, skip those with none
            for unal in peak_list:
                  pdict = all_peak_db[unal]
                  possible_models = self.prune_cont(pdict['sol_contacts'],cutoff=pdict['c1'],omit_null=True)
                  pdict['mod_cont'] = possible_models
                  if len(possible_models) == 0:
                        pdict['mflag'] = 1
                        pdict['mod_for'] = []
                        pdict['sol_mod'] = []
                        continue
            #2-way auction, each stores (peak,solvent,distance) --> important for hashing
            solvent_bids = {}
            peak_bids = {}
            pair_bids = {}
            for unal in peak_list: #first peaks
                  pdict = all_peak_db[unal] 
                  if pdict['mflag'] == 1:
                        continue
                  possible_models = pdict['mod_cont']
                  #find all contacts between peak and solvent
                  #each peak "bids" for a solvent
                  for cont in possible_models:
                        sol_u = cont['unal']
                        sol_d = cont['distance']
                        bids = solvent_bids.get(sol_u,[])
                        pair = pair_bids.get((unal,sol_u),[])
                        bids.append((unal,sol_u,sol_d))
                        pair.append(sol_d)
                        solvent_bids[sol_u] = sorted(list(set(bids)),key = lambda x: x[2])
                        pair_bids[(unal,sol_u)] = sorted(pair)
            for s_unal in sol_list: #next, solvent with fixed cutoff
                  pdict = all_peak_db[s_unal] 
                  possible_models = self.prune_cont(pdict['peak_contacts'],cutoff=3.0,omit_null=True)
                  #find all contacts between peak and solvent
                  #each peak "bids" for a solvent (reversed here)
                  for cont in possible_models:
                        peak_u = cont['unal']
                        peak_d = cont['distance']
                        bids = peak_bids.get(peak_u,[])
                        pair = pair_bids.get((peak_u,s_unal),[])
                        bids.append((peak_u,s_unal,peak_d))
                        pair.append(peak_d)
                        peak_bids[peak_u] = sorted(list(set(bids)),key = lambda x: x[2])
                        pair_bids[(peak_u,s_unal)] = sorted(list(set(pair)))
            #debugging output
            u2r = lambda x: all_peak_db.get(x,{'resat':'Null'})['resat']
            #for k,v in solvent_bids.iteritems():
            #      print "SB"," ".join(list("%s %s %s" % (u2r(p),u2r(s),d) for p,s,d in v))
            #for k,v in peak_bids.iteritems():
            #      print "PB"," ".join(list("%s %s %s" % (u2r(p),u2r(s),d) for p,s,d in v))
            #for k,v in pair_bids.iteritems():
            #      print "PAIR",u2r(k[0]),u2r(k[1])," ".join("%3.2f" % d for d in v)

            #lowest bids are assigned first, then removed from consideration
            cycles = 0
            claimed_sol = []
            while len(pair_bids) > 0 and cycles < 10:
                  cycles = cycles + 1 #just in case
                  best_bids = []
                  for u2,d in pair_bids.iteritems():
                        best_bids.append((u2[0],u2[1],d[0]))
                  best_bids.sort(key = lambda x: x[2])
                  closed_pairs = []
                  #claimed_sol = []
                  for bid in best_bids:
                        p_unal,s_unal,dist = bid
                        p_dict = all_peak_db[p_unal]
                        s_dict = all_peak_db[s_unal]
                        toadd = True
                        if s_unal in claimed_sol and len(p_dict['sol_mod']) > 0: #claimed, already associat.
                              toadd = False
                        if s_unal in claimed_sol and len(p_dict['sol_mod']) == 0: #claimed, no association
                              dist = dist*-1.0 #already claimed, mark negative
                        if toadd:
                              p_dict['sol_mod'].append((s_unal,dist))
                              s_dict['mod_for'].append((p_unal,abs(dist)))
                        del pair_bids[(p_unal,s_unal)][0]
                        if len(pair_bids[(p_unal,s_unal)]) == 0:
                              closed_pairs.append((p_unal,s_unal))
                        if dist > 0:
                              claimed_sol.append(s_unal)

                  for u2 in set(closed_pairs):
                        del pair_bids[(u2[0],u2[1])]
            #lastly, prune s.t. sol_mod distances are not more than 1.5A more than closest
            for p_unal in peak_list:
                 pdict = all_peak_db[p_unal]
                 sol_mod = pdict['sol_mod']
                 if len(sol_mod) > 1:
                       cutoff = sol_mod[0][1] + 1.5
                       trunc = list(ud for ud in pdict['sol_mod'] if ud[1] < cutoff)
                       pdict['sol_mod'] = trunc

            #for p_unal in peak_list:
            #      pdict = all_peak_db[p_unal]
            #      print "MOD",pdict['resat']," ".join(list("%s %s" % (u2r(u),d) for u,d in pdict['sol_mod']))
            #for s_unal in sol_list:
            #      pdict = all_peak_db[s_unal]
            #      print "SOL",pdict['resat']," ".join(list("%s %s" % (u2r(u),d) for u,d in pdict['mod_for']))
            return






            claimed_solvent = {}
            peak_owners = {}
            finished_sol = []
            unresolved_peaks = {}
            #start with lowest bids, go from there
            cycles = 0
            while len(solvent_bids) > 0 and cycles < 10:
                  finished_cycle = []
                  cycles = cycles + 1
                  for sol_u,bids in solvent_bids.iteritems():
                        if len(bids) > 0: 
                              peak_u,peak_d = bids[0]
                              claims = claimed_solvent.get(sol_u,[])
                              claims.append((peak_u,peak_d))
                              claimed_solvent[sol_u] = sorted(list(set(claims)), key = lambda x: x[1])
                              del bids[0]
                        if len(bids) == 0:
                              finished_cycle.append(sol_u)
                  for sol_u in finished_cycle:
                        del solvent_bids[sol_u]
                        finished_sol.append(sol_u)            
            #mark owners after initial 1:1 matches
            for sol_u,claims in claimed_solvent.iteritems():
                  for claim in claims:
                        peak_u = claim[0]
                        peak_d = claim[1]
                        owner = peak_owners.get(peak_u,[])
                        owner.append((sol_u,peak_d))
                        peak_owners[peak_u] = sorted(list(set(owner)),key = lambda x: x[1])
            peak_with_mod = []
            sol_with_peak = []
            for p_unal,sol_list in peak_owners.iteritems():
                  pdict = all_peak_db[p_unal]
                  #allow only matches within range
                  trunc_list = list(ud for ud in sol_list if ud[1] < sol_list[0][1] + 1.5)
                  pdict['sol_mod'] = trunc_list
                  if len(trunc_list) > 0:
                        peak_with_mod.append(p_unal)
            for s_unal,p_list in claimed_solvent.iteritems():
                  sdict = all_peak_db[s_unal]
                  trunc_list = list(ud for ud in p_list if ud[1] < p_list[0][1] + 1.5)
                  sdict['mod_for'] = trunc_list
                  if len(trunc_list) > 0:
                        sol_with_peak.append(s_unal)
            pwm =len(set(peak_with_mod)) 
            tot_sol = len(list(pdict for pdict in all_peak_db.values() if pdict['model'] == 3))
            swp = len(set(sol_with_peak))
            print "PEAK-->MODEL ASSOCIATIONS: %s/%s peaks w/ model, %s/%s sol atoms w/ peak" % (pwm,len(peak_list),swp,tot_sol)

      def close_cont(self,contacts,cutoff=2.5):
            # generates a list of clists, each list is a list of contacts to a particular residue_group
            cl_cont = self.prune_cont(contacts,cutoff=cutoff)
            cl_uniq = self.prune_cont(cl_cont,unires=True,omit_null=True)
            cl_rg = list(set(list(p['unrg'] for p in cl_uniq)))
            cl_list = []
            for rg in cl_rg:
                  clist = []
                  for cont in cl_cont:
                        if cont['unrg'] == rg:
                              clist.append(cont)
                  cl_list.append(clist)
            return cl_list


      def find_near_best(self,all_peak_db,unal,model,pick,cutoff=2.0,output_unscored=False,include_self=False):
            pdict = all_peak_db[unal]
            cont_db = pdict['cont_db']
            all_cont = cont_db[unal]
            models_to_omit = [1,2,3,4]
            models_to_omit.remove(model)
            possible = self.prune_cont(all_cont,omit_models=models_to_omit,cutoff=cutoff,omit_null=True)
            if include_self and 'prob_data' in pdict.keys():
                  matches = [(unal,pdict['prob_data'][0,pick-1],0.0),]
            else:
                  matches = []
            for cont in possible:
                  c_unal = cont['unal']
                  cp_dict=all_peak_db[c_unal]
                  if "pick" in cp_dict.keys():
                        if cp_dict['pick'] == pick:
                              pick_prob = cp_dict['prob_data'][0,pick-1]
                              #print "PPROB",pick_prob
                              matches.append((c_unal,pick_prob,cont['distance']))
                  elif output_unscored:
                        matches.append((c_unal,-1.0,cont['distance']))
            matches.sort(key = lambda match: match[1],reverse=True)
            return matches

      def quick_cluster(self,unal,cont_db,cdist=5.0):
            pcont_list = self.prune_cont(cont_db[unal],omit_models=[1,2,3],omit_unal=[unal,],cutoff=cdist,omit_null=True)
            unal_list = list(cont['unal'] for cont in pcont_list)
            return unal_list
       
      def assign_group(self,g1,g2,data):
            norms = np.zeros((2,data.shape[0]))
            g1diff = np.subtract(data,g1[None,:])
            g2diff = np.subtract(data,g2[None,:])
            norms[0,:] = np.linalg.norm(g1diff,axis=1)
            norms[1,:] = np.linalg.norm(g2diff,axis=1)
            group = norms[0,:] < norms[1,:]
            return group

      def hier_from_contacts(self,contact_list,resid,symm):
            #generates a dummy pdb of water coordinates from a contact list
            #useful for finding clashes, does not regenerate all residues, only non-H contacts
            #unique by shortest, pair_generator will apply symmetry
            pput = Util()
            pdb_atoms = ""
            added_unal = []
            unique_cont = self.prune_cont(contact_list,uniquify=True)
            for cind,contact in enumerate(unique_cont):
                  if contact['unal'] in added_unal:
                        continue
                  sym=contact['sym']
                  element = contact['element']
                  coord = contact['coord']
                  resid = int(contact['resid'])
                  name = contact['name']
                  resname = contact['resname']
                  altloc = contact['altloc']
                  chain = contact['chain']

                  if element == 'H':
                        continue
                  if (resid == 9999 and chain == 'ZZ'):
                        #remove original peak contact
                        continue
                  pdb_atoms = pdb_atoms+pput.write_atom(cind,name,altloc,resname,chain,resid,"",coord[0],coord[1],coord[2],1.0,35.0,element,"")
                  added_unal.append(contact['unal'])
            dummy_pdb = iotbx.pdb.input(source_info=None, lines=flex.split_lines(pdb_atoms))
            dummy_hier = dummy_pdb.construct_hierarchy()
            #dummy_hier.write_pdb_file(contact_list[0]['s_resat']+"_cont.pdb",append_end=True,crystal_symmetry=symm)
            return dummy_hier
          
      def cflag_pass1(self,features):
            #first pass look for bad contacts (one less than 1.6, three or more less than 2.2)
            clash = False
            culled = self.prune_cont(features['mm_contacts'],omit_unat=[features['unat'],],cutoff=2.2,unires=False,omit_null=True)
            if len(culled) > 0:
                  if culled[0]['distance']< 1.6:
                        clash = True
                  elif len(culled) > 2:
                        clash = True
            return clash
                                           
      def parse_contacts(self,features):
            """
            Tally Contacts and to filter out bad peaks
            Peaks refined too close to protein atoms will have many contacts less than 1A to atoms in the strip hierarchy
            """
            pput = Util()
            unal = features['unal']
            contact_tally = {}
            dist_cutoffs = {"low":(0.00,1.1),"mid":(1.1,1.7),"high":(1.7,4.5)}
            special = np.count_nonzero(list(cont['special'] == True for cont in features['w_contacts']))
            contact_tally['sp'] = special
            for cta,cdict in (('o','contacts'),('t','strip_contacts'),('s','s_contacts'),('w','w_contacts'),('p','peak_contacts')):
                  if len(features[cdict]) > 0:
                        for ctb,cutoffs in (('l','low'),('m','mid'),('h','high')):
                              lcut = dist_cutoffs[cutoffs][0]
                              hcut = dist_cutoffs[cutoffs][1]
                              contact_dict = features[cdict]
                              cut_contacts =np.count_nonzero(list((cont['distance'] >= lcut and cont['distance'] < hcut) for cont in contact_dict)) 
                              cut_contacts = np.clip(cut_contacts,0,9)
                              contact_tally[cta+ctb] = cut_contacts
                  else:
                        contact_tally[cta+'l'],contact_tally[cta+'m'],contact_tally[cta+'h'] = 0,0,0
            for feature in ('ol','om','oh','wl','wm','sl','sm','sp'):
                  features[feature] = contact_tally[feature]
            features['wt'] = features['wl'] + features['wm']
            features['st'] = features['sl'] + features['sm']
            



            
            
