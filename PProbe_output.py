import sys,os,copy,math,ast,time
import numpy as np
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=1e6, edgeitems=1e6)
from PProbe_selectors import Selectors
from PProbe_util import Util
from PProbe_dataio import DataIO as PPio
from PProbe_stats import StatFunc
import iotbx.pdb
from scitbx.array_family import flex

class Output:
    def __init__(self,mdict_file=None,outstem=None):
        self.ppio = PPio(phenix_python=False)
        self.pput = Util()
        self.ppstat = StatFunc()
        self.master_dict = self.ppio.read_master_dict(input_dfile=mdict_file)
        self.initialized = False
        if outstem is None:
            self.outstem = "test_"
        else:
            self.outstem=outstem
        self.istat_dict = self.get_istat_dict()

    def tab_iprint(self,i,pref=""):
        restab = self.restab
        dat = restab[i]
        resat = self.u2r.get(dat['unal'],"Unk")
        if dat['mod'] == 0:
            if dat['model'] == 3:
                if int(dat['moc']/2) % 2 == 1: #claimed
                    sm = "[%s]" % self.u2r.get(restab['unal'][dat['peaki']],"Unk")
                else:
                    sm = "---unclaimed"
            else:
                sm = "None"
        else:
            sm = self.u2r.get(restab['unal'][dat['modi']],"None")
  
        fmtstr = ('{:<14} TYPE {:1} GRP {:4} {:2} PROC {:1} PICK {:1} {:1} QUAL {:1} MOD {:1} SM {:<14} MOC {:1} ' 
                  'MQUAL {:1} MPICK {:1} {:1} MLAB {:1} CMP {:2} FC {:1} MF {:1}{:1}{:1} STAT {:4} NP {:1} GV {:3} {:1}')
        outstr = fmtstr.format(resat,dat['model'],dat['group'],dat['grank'],dat['proc'],dat['pick'],dat['cpick'],
                               dat['qual'],dat['mod'],sm,dat['moc'],dat['mqual'],dat['mpick'],dat['mcpick'],
                               dat['mlab'],dat['mcomp'],dat['fc'],dat['mfr'],dat['mfq'],dat['mfl'],
                               dat['status'],dat['npick'],dat['istat'],dat['fmk'])
        print pref,outstr
    


    def initialize_lists(self,all_peak_db,valsol=False):

        # start with null null peak
        # included so refs to index 0 point to null
        null_peak = all_peak_db[-8861610501908601326]

        self.all_unal = [null_peak['unal'],]  
        self.all_unat = [null_peak['unat'],]
        self.all_unal.extend(list(pdict['unal'] for pdict in all_peak_db.values() if pdict['model'] in [3,4]))
        self.all_unat.extend(list(pdict['unat'] for pdict in all_peak_db.values() if pdict['model'] in [3,4]))
        self.omit_unat = list(pdict['unat'] for pdict in all_peak_db.values() if (pdict['model'] == 2 or( pdict['model'] == 3 and valsol)))
        omitted_sol_at = set(self.all_unat) & set(self.omit_unat)
        self.omitted_solu = list(unal for unal in self.all_unal if all_peak_db[unal]['unat'] in omitted_sol_at)
        ptot = len(self.all_unal)
        columns = ('unal','group','grank','model','proc','pick','cpick','qual','mod','modi','peaki','moc','mqual','mpick','mlab','mcpick','mcomp','fc','mfr','mfq','mfl','status','npick','istat','fmk')
        fmts =    ('i8'  ,'i8'   , 'i2'  , 'i2'  ,  'i2', 'i2' ,'i2'   ,'i2'  ,'i4' ,'i2'  ,'i2'   ,'i2' ,'i2'   ,'i2'   ,'i2'  ,'i2'    ,'i2'   ,'i2','i2' ,'i2' ,'i2' ,'i2'    ,'i2'   ,'i2'   ,'i2')
        rt_dtype = np.dtype(zip(columns,fmts))
        self.restab = np.zeros(ptot,dtype=rt_dtype)
        #lookup hashes
        self.u2i = {}
        self.i2u = {}
        self.u2r = {}
        self.i2r = lambda i: self.u2r[self.i2u[i]]
        for uind,unal in enumerate(self.all_unal):
            self.restab['unal'][uind] = unal 
            self.u2i[unal] = uind
            self.i2u[uind] = unal
            self.u2r[unal] = all_peak_db[unal]['resat']


        self.pm_mat = self.get_pm_mat(all_peak_db)

        for uind,unal in enumerate(self.all_unal):
            urow = self.restab[uind]
            pdict = all_peak_db[unal]
            for col in ('model','fc','status'):
                urow[col] = pdict.get(col,-1)
            if urow['status'] > 999 and urow['status'] < 5000:
                urow['pick'] = urow['status']/1000
                urow['qual'] = (urow['status'] % 1000)/100
            else:
                urow['pick'] = pdict.get('pick',0)
                urow['qual'] = 0
            if urow['model'] == 4:
                urow['mfr'] = (pdict['mf'] % 100)/10
                urow['mfq'] = pdict['mf'] % 10
                urow['proc'] = pdict['status'] not in [3,7]
                match_soli = list(np.nonzero(self.pm_mat[uind]>=0.0)[0])
                if len(match_soli)>0:
                    mod_soli = list(si for si in match_soli if self.pm_mat[uind][si] < 1.65 and si != uind)
                    mod_soli.sort(key = lambda si: self.pm_mat[uind,si])
                    n_mod = len(mod_soli)
                    if n_mod > 0:
                        urow['mod'] = n_mod
                        urow['modi'] = mod_soli[0]
                if valsol: #models are self
                    urow['mod'] = 1
                    urow['mlab'] = pdict['label']
            elif urow['model'] == 3:
                urow['proc'] = pdict['status'] not in [1,3,6,7]
                claimed_by_i = list(np.nonzero(self.pm_mat[uind]>=0.0)[0])
                claimed = 0
                if len(claimed_by_i) > 0:
                    mod_peaki = list(si for si in claimed_by_i if self.pm_mat[uind][si] < 1.65 and si != uind)
                    mod_peaki.sort(key = lambda pi: self.pm_mat[uind,pi]) #gets closest
                    n_mod = len(mod_peaki)
                    if n_mod > 0 and mod_peaki[0] != 0: #null doesnt count
                        claimed = 1
                        urow['peaki'] = mod_peaki[0]
                omit = unal in self.omitted_solu
                urow['moc'] = omit+2*claimed+4*urow['proc']
                urow['group'] = pdict['unrg']#changed later
                urow['mlab'] = pdict['label']
                if urow['fc'] == -1:
                    urow['fc'] = 0

        self.link_mat = self.get_link_mat(all_peak_db)
        #group by linkage, peak/model pairs
        tmp_groups = []
        for punal in self.all_unal:
            pi = self.u2i[punal]
            tmp_groups.append(tuple(sorted(list(np.nonzero(self.link_mat[pi] > 0)[0]))))
        tmp_groups = list(set(tmp_groups)) #uniquify
        tmp_groups.sort(key = lambda ilist: len(ilist), reverse=True)
        tmp_groups = self.uniq_groups(None,explicit_lists=tmp_groups)
        group_dict = {}
        for gi,group in enumerate(tmp_groups):
            if len(group) > 1:
                group_type = self.restab['model'][group[0]]
                group_unal = list(self.i2u[si] for si in group)
                elect_pick = self.clust_elect(all_peak_db,group_unal)
                elect_rank = self.group_rank(all_peak_db,group_unal,elect_pick)
            else:
                elect_pick = 0
                elect_rank = [0,]
            for sind,si in enumerate(group):
                group_dict[si] = [gi+1,elect_pick,elect_rank[sind]]

        #assign groups, group selections, group ranks
        for pi,peak in enumerate(self.restab):
            peak['mfl'] = np.clip(np.count_nonzero(self.link_mat[pi]),0,9)
            group_dat = group_dict.get(pi,[0,0,0])
            peak['group'] = group_dat[0]
            peak['cpick'] = group_dat[1]
            peak['grank'] = group_dat[2]
        grouped_unproc = self.merge_masks([self.restab['mfl']>1,self.restab['proc']==0])
        self.get_group_mats()
        self.restab['grank'][grouped_unproc] = 9
        if not valsol:
            self.match_modsol()
        #self.propagate_modparam()
        self.initialized = True



    def match_modsol(self):
        #propagate model info to peaks
        for pind,peak in enumerate(self.restab):
            if peak['model'] == 4 and peak['mod'] > 0:
                solmod = self.restab[peak['modi']]
                peak['mlab'] = solmod['mlab']
                peak['moc'] = solmod['moc']
                if solmod['proc']:
                    peak['mqual'] = solmod['qual']
                    peak['mpick'] = solmod['pick']
                    peak['mcpick'] = solmod['cpick']


    def propagate_modparam(self):
        #watch out for shallow copies?
        for sind,solpeak in enumerate(self.restab):
            if solpeak['model'] == 3:
                s_group = solpeak['group']
                gseli = np.nonzero(self.restab['group'] == s_group)[0]
                s_moc = solpeak['moc']
                for si in gseli:
                    if s_moc > self.restab['moc'][si]:
                        self.restab['moc'][si] = s_moc
     

                    
    def tab_rsel(self,column,tlist,match=False,union=None,intersect=None):
        #returns boolean selectors
        if type(tlist) != type([]):#in case called with bad arg
            tlist = [tlist,]
        sel = None
        if len(tlist) == 1:
            sel = self.restab[column] == tlist[0]
        elif len(tlist) == 2 and not match: #take as low/high inclusive
            s1 = self.restab[column] >= tlist[0]
            s2 = self.restab[column] <= tlist[1]
            sel = np.logical_and(s1,s2)
        elif len(tlist) > 2 or match: #accumulate matches
            sel = np.zeros(self.restab.shape[0],dtype=np.bool_)
            for val in tlist:
                toadd = self.restab[column] == val
                sel = np.logical_or(sel,toadd)
        if union is not None:
            sel = np.logical_or(union,sel)
        if intersect is not None:
            sel = np.logical_and(intersect,sel)
        if sel is None: #failsafe
            sel = np.zeros(self.restab.shape[0],dtype=np.bool_)
        #print "SELECTED",np.count_nonzero(sel)
        return sel

    def merge_masks(self,mask_list,opp=""):
        #be sure input masks are not overwritten
        if type(mask_list) != type([]) or len(mask_list) == 0:
            return 0
        if len(mask_list) == 1:
            return mask_list[0]
        combined_mask = np.zeros(mask_list[0].shape,dtype=np.bool_)
        combined_mask[:] = mask_list[0][:]
        if len(opp) == 0:
            opp = "i"*(len(mask_list)-1)
        if len(opp) < (len(mask_list)-1):
            opp=opp+opp[-1]*(len(mask_list) - len(opp)-1)
        for index,mask in enumerate(mask_list[1::]):
            if opp[index] == 'i' or opp[index] == 'I': #intersect
                combined_mask = np.logical_and(combined_mask,mask)
            if opp[index] == 'u' or opp[index] == 'U': #union
                combined_mask = np.logical_or(combined_mask,mask)
        return combined_mask

    def selprint(self,sel,pref=""):
        for i in np.nonzero(sel)[0]:
            self.tab_iprint(i,pref=pref)



    def explicit_rank(self,all_peak_db,ilist,forpick=None):
        #givin index list, returns ranked (best to worst) as
        #tuple (proc,i,pick,score)
        #if forpick is not given, score is score for pick
        #if forpick, scores for a given pop are used
        if len(ilist) == 1:
            return [(0,ilist[0],0,0.0),]
        rt = self.restab
        all_list = list(set(ilist))
        proc_list = list(pi for pi in all_list if rt['proc'][pi])
        if len(proc_list) < 1:
            #return null data
            return list((0,i,0,-1.0*ind) for ind,i in enumerate(ilist))
        all_proc_list = []
        for pi in proc_list:
            pdict = all_peak_db[self.i2u[pi]]
            if forpick is None:
                p_pick = rt['pick'][pi]
                prior = 1
            else:
                p_pick = forpick
                prior = 0
                
            p_score = pdict['prob_data'][prior,p_pick-1]
            all_proc_list.append((1,pi,p_pick,p_score))
        all_proc_list.sort(key = lambda ipqs: ipqs[3],reverse=True)
        nproc_list = list(set(all_list) - set(proc_list))
        for pind,pi in enumerate(nproc_list):
            all_proc_list.append((0,pi,0,all_proc_list[-1][3]-1.0*pind))
        return all_proc_list
                

    def compare_p2m(self,all_peak_db,punal,sol_filt=None):
        #compares histogram likelihoods peaks and model
        #only 1 to 1, single peak vs single mod
        rt = self.restab
        if sol_filt is None: #all sol allowed
            sol_filt = np.ones(rt.shape[0],dtype=np.bool_)

        peak = rt[self.u2i[punal]]
        pi = self.u2i[punal]
        pdict = all_peak_db[punal]
        p_pick = peak['pick']
        p_group = peak['group']
        mi = peak['modi']
        solmod = rt[mi]
        m_pick = solmod['pick']
        m_group = solmod['group']
        p_sel = self.ingroup_mat[p_group]
        m_sel = self.ingroup_mat[m_group]
        munal = solmod['unal']
        if (peak['mod'] == 0 or peak['moc'] != 7  or peak['pick'] == 0):
            #no model, model is not processed
            rt['mcomp'][p_sel] = 0
            rt['mcomp'][m_sel] = 0
            return
        if not sol_filt[mi]:
            rt['mcomp'][p_sel] = 0
            rt['mcomp'][m_sel] = 0
            return

        mdict = all_peak_db[munal]
        peak_scores = pdict['prob_data'][:,p_pick-1]
        mod_scores = mdict['prob_data'][:,m_pick-1]
        count = np.count_nonzero(np.greater(peak_scores,mod_scores))
        
        if p_pick == m_pick:
            if count > 1: #peak is better
                rt['mcomp'][p_sel] = 1
                rt['mcomp'][m_sel] = 2
            else:#model has better scores
                rt['mcomp'][p_sel] = 2
                rt['mcomp'][m_sel] = 1
        else: #disagreement
            if count > 1: 
                rt['mcomp'][p_sel] = 3
                rt['mcomp'][m_sel] = 4
            else:#model has better scores
                rt['mcomp'][p_sel] = 4
                rt['mcomp'][m_sel] = 3

    def expand_to_rg(self,all_peak_db,unal):
        pdict = all_peak_db[unal]
        p_unrg = pdict['unrg']
        p_mod = pdict['model']
        match_unal = [unal,]
        for t_unal in list(unal for unal in self.all_unal if all_peak_db[unal]['model'] == p_mod):
            t_pdict = all_peak_db[t_unal]
            t_unrg = t_pdict['unrg']
            if t_unrg == p_unrg:
                match_unal.append(t_unal)
        return list(set(match_unal))


    def get_group_mats(self):
        #useful matrices for creating boolean selectors
        #ingroup_mat is row = group, true for peaks in group
        #system_mat is row = index, 3/4 for peak/mod association
        #fully expanded and uniquified
        rt = self.restab
        groups = [0,] #zero group is null
        groups.extend(list(set(list(rt['group']))))
        max_gi = np.amax(groups)
        ngroup = max_gi+2 #keep zero
        npeaks = rt.shape[0]
        ingroup_mat = np.zeros((ngroup,npeaks),dtype=np.bool_)
        system_mat = np.zeros((npeaks,npeaks),dtype=np.int16)
        for group in groups:
            ingroup = rt['group'] == group
            ingroup_mat[group] = ingroup
        system_lists = []
        for i in range(npeaks):
            #peak/sol used interchangeably, may be either diff map peak or model
            peak_model = rt['model'][i]
            system_mat[i,i] = peak_model
            peak_group_sel = self.link_mat[i] > 0
            pi_in_group = np.nonzero(peak_group_sel)[0]
            ass_soli = []
            for pi in pi_in_group: #get associated solvent/peaks
                ass_sel = self.merge_masks([self.pm_mat[pi] >= 0,self.pm_mat[pi]<1.65])
                ass_seli = np.nonzero(ass_sel)[0]
                ass_soli.extend(list(ass_seli))
            ass_soli=list(set(ass_soli))
            all_connect = [i,]
            all_connect.extend(pi_in_group)
            all_connect.extend(ass_soli)
            all_connect = list(set(all_connect))
            cur_npeak = len(all_connect)
            not_closed = True
            cycle = 0
            while not_closed:
                cycle = cycle + 1
                exp_associations = [i,]
                for pi in all_connect:
                    exp_associations.extend(list(np.nonzero(self.link_mat[pi] > 0)[0]))
                    bridge_sel = self.merge_masks([self.pm_mat[pi] >= 0,self.pm_mat[pi]<1.65])
                    bridge_i = np.nonzero(bridge_sel)[0]
                    exp_associations.extend(list(bridge_i))
                exp_associations= list(set(exp_associations))
                if len(exp_associations) == cur_npeak or cycle==10:
                    not_closed = False
                else:
                    all_connect = exp_associations
                    cur_npeak = len(all_connect)
                print "CONN CYCLE",self.i2r(i),cycle,cur_npeak
            system_lists.append(all_connect)
        system_lists.sort(key = lambda ilist: len(ilist),reverse=True)
        #peaks in multiple systems are kept in larger systems, dropped in smaller
        placed_peaks = []
        unique_slists = []
        for slist in system_lists:
            l1 = len(slist)
            tmp_list = list(pi for pi in slist if pi not in placed_peaks)
            placed_peaks.extend(tmp_list)
            l2 = len(tmp_list)
            if len(tmp_list) > 0:
                unique_slists.append(tmp_list)
        for slist in unique_slists:
            for pi in slist:
                for pi2 in slist:
                    sm = rt['model'][pi2]
                    if pi != pi2:
                        system_mat[pi,pi2] = sm

        for pind,system in enumerate(system_mat):
            peaks = list(np.nonzero(system==4)[0])
            solmod = list(np.nonzero(system==3)[0])
            ppicks = " ".join(list(str(rt['pick'][pi]) for pi in peaks))
            spicks = " ".join(list(str(rt['pick'][si]) for si in solmod))
            pstr = " ".join(list(self.i2r(x) for x in peaks))
            sstr = " ".join(list(self.i2r(x) for x in solmod)) 
            #print "SYSTEM",len(peaks),len(solmod),system[pind],"||"," || ".join([ppicks,spicks,pstr,sstr])
            
        self.ingroup_mat = ingroup_mat
        self.system_mat = system_mat

    def get_pm_mat(self,all_peak_db):
        #peak/model association matrix, non-hermetian
        #elements are distances between indexed peaks
        #associated by previous model routine
        pm_mat = np.zeros((self.restab.shape[0],self.restab.shape[0]))-1.0
        for pi,unal in enumerate(self.all_unal):
            pdict = all_peak_db[unal]
            if pdict['model'] == 3:
                mfl = pdict.get('mod_for',[])
                for u,d in mfl:
                    mi = self.u2i[u]
                    pm_mat[pi,mi] = d
            if pdict['model'] == 4:
                sml = pdict.get('sol_mod',[])
                for u,d in sml:
                    pi2 = self.u2i[u]
                    pm_mat[pi,pi2] = d

        #for pi,row in enumerate(pm_mat):
        #    modi = np.nonzero(row >= 0.0)[0]
        #    modfor = []
        #    solmod = []
        #    pmout1 = []
        #    pdict = all_peak_db[self.i2u[pi]]
        #    if len(modi) > 0:
        #        for mi in modi:
        #            pmout1.append((self.u2r[self.i2u[mi]],row[mi]))
        #        if pdict['model'] == 3:
        #            modfor.extend(pdict['mod_for'])
        #        if pdict['model'] == 4:
        #            solmod.extend(pdict['sol_mod'])
        #        print "MODI_1",self.i2r(pi)," | ".join(list("%s %3.2f" % (r,d) for r,d in pmout1))
        #        print "MODI_2",self.i2r(pi)," | ".join(list("%s %3.2f" % (self.u2r[u],d) for u,d in modfor))
        #        print "MODI_3",self.i2r(pi)," | ".join(list("%s %3.2f" % (self.u2r[u],d) for u,d in solmod))
        return pm_mat

    def get_link_mat(self,all_peak_db):
        #cluster information by pseudo adjacency matrix
        #only in-type (clustering boolean)
        #1 = strongly connected
        #2 = weakly connected
        #3 = bridged/connected by multiatom solvent
        rt = self.restab
        cnz = np.count_nonzero
        link_mat = np.zeros((rt.shape[0],rt.shape[0]),dtype=np.int8)
        clust_maj = self.merge_masks([rt['model'] == 4,rt['mfr'] == 1])
        clust_maj_u = list(rt['unal'][clust_maj])
        for unal in clust_maj_u:
            pdict = all_peak_db[unal]
            pi = self.u2i[unal]
            all_clust = pdict['clust_mem']
            to_check = set(self.all_unal) & set(all_clust)
            ilist = list(self.u2i[pi] for pi in to_check)
            ranks_as_w = self.explicit_rank(all_peak_db,ilist,forpick=1)
            for rank in ranks_as_w:
                pi2 = rank[1]
                if rank[3] < 0:
                    linkval = 1
                else:
                    linkval = 2
                link_mat[pi,pi2] = linkval
                link_mat[pi2,pi] = linkval
        #expand clusters linked by solvent
        peak_with_mod = np.logical_and(rt['model'] == 4,rt['mod'] == 1)
        #peaks with multiatom solvent
        pmod_ma = list(unal for unal in rt['unal'][peak_with_mod] if all_peak_db[unal]['mflag'] > 99)
        for unal in pmod_ma:
            pdict = all_peak_db[unal]
            pi = self.u2i[unal]            
            clust_mem_i =np.nonzero(link_mat[pi] > 0)[0]
            all_sol_u = []
            #get all co-clustered peaks for input peak
            all_peak_u = list(rt['unal'][pi] for pi in clust_mem_i)
            #get all solvent models for all associated peaks
            for cunal in all_peak_u:
                all_sol_u.extend(list(u for u,d in all_peak_db[cunal]['sol_mod'] if d < 1.65))
            all_sol_u = list(set(all_sol_u))
            #expand all solvent models by residue
            full_sol_u = []
            for sunal in all_sol_u:
                full_sol_u.extend(self.expand_to_rg(all_peak_db,sunal))
            all_sol_u = list(set(full_sol_u))
            #all_si = list(self.u2i[sunal2] for sunal2 in all_sol_u)
            # get all peaks associated with expanded solvent (reverse)
            for sunal in all_sol_u:
                all_peak_u.extend(list(u for u,d in all_peak_db[sunal]['mod_for'] if d < 1.65))
            all_peak_u = list(set(all_peak_u))
            all_pi = list(self.u2i[peaku] for peaku in all_peak_u)
            for pi1 in all_pi:
                for pi2 in all_pi:
                    if link_mat[pi1,pi2] == 0 and pi1 != pi2:
                        link_mat[pi1,pi2] = 3

        #also mark multiatom solvent
        for si,model in enumerate(rt['model']):
            if model == 3:
                group = rt['group'][si]
                if cnz(group) > 0:
                    gsel = rt['group'] == group
                    link_mat[si,:] = gsel
                    link_mat[:,si] = gsel
            
        np.fill_diagonal(link_mat,1)
        #debugging
        #for pi1,row in enumerate(link_mat):
        #    links = cnz(row)
        #    if links > 1:
        #        link_list = []
        #        ilist = np.nonzero(row)[0]
        #        for pi2 in ilist:
        #            link_list.append((self.u2r[self.i2u[pi2]],row[pi2]))
        #        print "LINK",self.u2r[self.i2u[pi1]]," | ".join(list("%s %s" % (r,d) for r,d in link_list))
        #        print "LINKLEN",self.u2r[self.i2u[pi1]],len(link_list)
        return link_mat

    def prune_water(self,all_peak_db,ilist):
        #takes index list of waters
        #returns three lists, keep,kill,stray
        #pairwise comparison, ranked peaks, 1st compared against it model (if found)
        #better peak kept if passes muster, else 2nd peak, unless both fail
        #if found model is not not in input list, added to stray,
        rt = self.restab
        ranks = self.explicit_rank(all_peak_db,ilist,forpick=1)
        on_deck = list(set(list(rank[1] for rank in ranks)))
        keep = []
        kill = []
        stray = []
        cycle = 1
        while on_deck and cycle < 20: 
            pi = on_deck[0]
            paired = False
            if rt['model'][pi] == 4 and rt['mod'][pi]:
                other_i = rt['modi'][pi]
                paired = True
            elif rt['model'][pi] == 3 and rt['moc'][pi] in [2,3,6,7]:
                other_i = rt['peaki'][pi]
                paired = True

            if paired: #has model
                #passes muster as water
                if rt['status'][pi] > 999 and rt['fc'][pi] in [0,1,6,7]:
                    keep.append(pi)
                    on_deck.remove(pi)
                    if other_i in on_deck:
                        kill.append(other_i)
                        on_deck.remove(other_i)
                    else:
                        stray.append(other_i)
                #doesn't pass, kill
                else:
                    kill.append(pi)
                    on_deck.remove(pi)
                    #does paired peak?
                    if rt['status'][other_i] > 999 and rt['fc'][other_i] in [0,1,6,7]:
                        if other_i in on_deck:
                            keep.append(other_i)
                            on_deck.remove(other_i)
                        else:
                            stray.append(other_i)
                    else:
                        if other_i in on_deck:
                            kill.append(other_i)
                            on_deck.remove(other_i)
                        else:
                            stray.append(other_i)
            # unpaired
            else:
               if rt['status'][pi] > 3 and rt['fc'][pi] in [0,1,6,7]:
                   keep.append(pi)
                   on_deck.remove(pi)
               else:
                   kill.append(pi)
                   on_deck.remove(pi)
                        
            cycle = cycle + 1 #just in case to avoid inf loops
        return keep,kill,stray
                    

    def clust_elect(self,all_peak_db,plist):
        if type(plist) != type([]) or len(plist) == 0:
            return 0
        rt = self.restab        
        if len(plist) == 1:
            return rt[self.u2i[plist[0]]]['pick']
        ilist = list(self.u2i[pi] for pi in plist)

        #hold an election for all associated peaks
        clust_votes = [0,0,0,0,0]
        for punal in plist:
            pi = self.u2i[punal]
            if rt['proc'][pi] > 0:
                pick = rt['pick'][pi]
                qual = rt['qual'][pi]
                not_metal = pick != 4
                #cluster elections never pick metal correctly
                clust_votes[pick] = clust_votes[pick] + qual*not_metal + 1 
        p_win = np.argmax(clust_votes)
        ranks = self.explicit_rank(all_peak_db,ilist,forpick=p_win)
        ranks_as_w = self.explicit_rank(all_peak_db,ilist,forpick=1)
        scores = " ".join(list(" %4.2f" % rank[3] for rank in ranks))
        wscores = list(rank[3] for rank in ranks_as_w)
        nnw = np.count_nonzero(np.array(wscores) < 0)
        wscrout = " ".join(" %4.2f" % x for x in wscores)
        votestr = " ".join(list(" %4.2f" % vote for vote in clust_votes))
        besti = ranks[0][1]
        print "ELECTION %13s" % self.i2r(besti),p_win,nnw,len(ilist)," || ".join([scores,votestr," ".join(list(self.i2r(rank[1]) for rank in ranks)),wscrout])
        return p_win


    def group_rank(self,all_peak_db,g_ulist,pick):
        rt =self.restab
        g_ilist = np.array(list(self.u2i[punal] for punal in g_ulist))
        is_proc = np.array(list(rt['proc'][si] == 1 for si in g_ilist))
        score_arr = np.zeros(len(g_ilist))  
        ranks = np.zeros(len(g_ilist),dtype=np.int16)
        if np.count_nonzero(is_proc) == 0:
            return ranks
        if np.count_nonzero(is_proc) == 1:
            ranks[is_proc] = 1
            return ranks
        for sind,si in enumerate(g_ilist):
            if is_proc[sind]:
                s_pdict = all_peak_db[self.i2u[si]]
                score_arr[sind] = s_pdict['prob_data'][1][pick-1]
            else:
                score_arr[sind] = -9999.99
        ranksort = np.argsort(score_arr)[::-1]
        for pos,rank in enumerate(ranksort):
            ranks[rank] = pos+1
        ranks = np.clip(ranks,1,8)
        ranks[np.invert(is_proc)] = 9
        return ranks
        
    def append_istat(self,all_peak_db):
        for pind,row in enumerate(self.restab):
            unal = row['unal']
            istat = row['istat']
            pdict=all_peak_db[unal]
            is_hist = pdict.get('is_hist',[])
            is_hist.append(istat)
            pdict['is_hist'] = is_hist
    

    def breakdown(self,all_peak_db):
        #successively sieves through all peaks to make assignments, decide fate
        if not self.initialized:
            self.initialize_lists(all_peak_db)

        #convient funcion handles and data refs
        rt = self.restab
        pm_mat = self.pm_mat
        link_mat = self.link_mat
        inv = np.invert 
        cnz = np.count_nonzero
        
        """
        for i in np.arange(0,rt.shape[0],1):
            for j in np.arange(i+1,rt.shape[0],1):
                ranks = self.explicit_rank(all_peak_db,[i,j])
                peaks = " ".join("%13s" % self.i2r(rank[1]) for rank in ranks if rank[0]==1)
                picks = " ".join("%1s" % rank[2] for rank in ranks if rank[0]==1)
                scores = " ".join("%5.3f" % rank[3] for rank in ranks if rank[0] == 1)
                print "COMPARE","%13s to %13s" % (self.i2r(i),self.i2r(j)),"||"," | ".join([peaks,picks,scores])
        """
        #selectors
        allsel = np.ones(rt.shape[0],dtype=np.bool_)
        psel = rt['model'] == 4
        ssel = rt['model'] == 3
        proc = rt['proc'] == 1
        has_mod = rt['mod'] > 0
        claimed = (rt['moc']/2) % 2 == 1
        omit = (rt['mod'] % 2) == 1

        
        #peak groupings
        #  ecl  npeak   nmod
        #    0      0      0     something amiss
        #    1      0      1     unclaimed solo model
        #    2      1      0     solo peak no mod
        #    3      1      1     solo peak solo mod
        #    4      0      >1    unclaimed ma model
        #    5      >1     0     peak cluster no mod
        #    6      1      >1    solo peak ma model
        #    7      >1     1     peak cluster solo model
        #    8      >1     >1    peak cluster ma model

        #peak_groups with no models
        eval_class = np.zeros(rt.shape[0],dtype=np.int16)
        npeak_count = np.nansum(self.system_mat==4,axis=1)
        nmod_count = np.nansum(self.system_mat==3,axis=1)
        mask1 = self.merge_masks([npeak_count==0,nmod_count==1])
        mask2 = self.merge_masks([npeak_count==1,nmod_count==0])
        mask3 = self.merge_masks([npeak_count==1,nmod_count==1])
        mask4 = self.merge_masks([npeak_count==0,nmod_count >1])
        mask5 = self.merge_masks([npeak_count >1,nmod_count==0])
        mask6 = self.merge_masks([npeak_count==1,nmod_count >1])
        mask7 = self.merge_masks([npeak_count >1,nmod_count==1])
        mask8 = self.merge_masks([npeak_count >1,nmod_count >1])
        for maski,mask in enumerate((mask1,mask2,mask3,mask4,mask5,mask6,mask7,mask8)):
            eval_class[mask] = maski+1
            print "EVAL_CLASS",maski+1,cnz(mask)
            for unal in rt['unal'][mask]:
                print "EC",maski+1,self.u2r[unal]

        sol_no_peak = self.merge_masks([eval_class == 1,eval_class == 4,ssel],opp='ui')
        peak_no_sol = self.merge_masks([eval_class == 2,eval_class == 5,psel],opp='ui')


        #all peaks
        self.selprint(allsel,pref="ALL01")
        print "\n".join(self.get_confused(all_peak_db,pref="CMS01"))
        self.append_istat(all_peak_db)

        #divide and conquer

        #first, identify clusters vs. solo and peak-mod pairs / unpaired
        solo_peak = self.merge_masks([eval_class==2,eval_class==3,eval_class==6,psel],opp='uui')
        solo_sol = self.merge_masks([eval_class==1,eval_class==3,eval_class==7,ssel],opp='uui')

        #first, cull out errors and rejects
        p_err_rej = self.merge_masks([rt['status']==1,rt['status'] == 2,inv(proc),psel],opp="uui")
        s_proc_err = self.merge_masks([rt['status'] == 1,rt['status']==2,solo_sol],opp="ui")
        proc_rej = self.merge_masks([p_err_rej,s_proc_err],opp='u')
        rt['istat'][proc_rej] = -1  #ASSIGN ps proc error
        
        
        #rotamer/alt errors 
        p_rot_err = self.merge_masks([solo_peak,proc,rt['status'] == 401])
        s_rot_err = self.merge_masks([solo_sol,proc,rt['status'] == 401])
        rot_err = self.merge_masks([p_rot_err,s_rot_err],opp="u")
        rt['istat'][rot_err] = -2 #ASSIGN ps rot err

        #backbone
        p_back_err = self.merge_masks([solo_peak,proc,rt['status'] == 402])
        s_back_err = self.merge_masks([solo_sol,proc,rt['status'] == 402])
        back_err = self.merge_masks([p_back_err,s_back_err],opp="u")
        rt['istat'][back_err] = -3 #ASSIGN ps back err


        #likely weak/noise
        p_noise = self.merge_masks([rt['status'] == 403,rt['status'] == 431,solo_peak],opp="uii")
        s_noise1 = self.merge_masks([rt['status'] == 431,solo_sol])
        s_noise2 = self.merge_masks([rt['status'] == 413,inv(claimed),solo_sol])
        all_noise = self.merge_masks([p_noise,s_noise1,s_noise2],opp="u")
        rt['istat'][all_noise] = -4 #ASSIGN ps noise


        #bad peaks marked
        self.selprint(allsel,pref="ALL02")
        print "\n".join(self.get_confused(all_peak_db,pref="CMS02"))
        self.append_istat(all_peak_db)


        #sol without models (solo or group)
        sol_no_p_g = list(set(list(rt['group'][sol_no_peak])))
        for sg in sol_no_p_g:
            gsel = self.ingroup_mat[sg]
            if (rt['moc'][gsel] == 0).all(): 
                rt['istat'][gsel] = 7 #ASSIGN s moc=0, keep in place
            else: #at least one in group omitted
                omit_nclaim_nproc = self.merge_masks([rt['moc'] == 1,ssel]) #usually bad water
                omit_nclaim_proc = self.merge_masks([rt['moc'] == 5,ssel]) #did not get claimed (closest peak picked another model)
                if (omit_nclaim_proc[gsel]).all():
                    rt['istat'][gsel] = 8 #ASSIGN s, unclaimed but proc (margin peaks, splits, reprocessed later)
                elif (np.logical_or(omit_nclaim_nproc[gsel],omit_nclaim_proc[gsel])).all():
                    rt['istat'][gsel] = -6 #ASSIGN s, all in group moc=1 or mod=5, kill


        self.split_resolve(all_peak_db)

        #sol to keep and del marked
        self.selprint(allsel,pref="ALL03")
        print "\n".join(self.get_confused(all_peak_db,pref="CMS03"))
        self.append_istat(all_peak_db)

        #clash/quality filters
        no_clash = self.tab_rsel('fc',[0,1,6,7])
        #cumulative
        qual1 = self.merge_masks([rt['qual'] > 6,no_clash,rt['fmk']==0,proc])
        qual2 = self.merge_masks([rt['qual'] > 3,no_clash,rt['fmk']==0,proc])
        qual3 = self.merge_masks([rt['qual'] >0,no_clash,rt['fmk']==0,proc])
        qual4 = self.merge_masks([qual3,rt['fc']==5,rt['status'] == 499,rt['fmk']==0,proc],opp="uui")
        qualall = self.merge_masks([proc,rt['pick'] > 0]) #all scored


        #peaks with no models
        peak_no_m_g = list(set(list(rt['group'][peak_no_sol])))
        for pg in peak_no_m_g:
            gsel = self.ingroup_mat[pg]
            if cnz(gsel) == 1: #solo peak,no satellites
                pi = np.nonzero(gsel)[0]
                if rt['qual'][pi] == 0 and rt['istat'][pi] == 0:
                    rt['istat'][pi] = -5 #ASSIGN p, kill, solo didn't pass muster
                elif rt['istat'][pi] == 0:
                    rt['istat'][pi] = 50 + rt['pick'][pi] #ASSIGN multi p (51,52,53,54), added
            else: #peak cluster
                pass #leave alone for now, processed below


        #peaks to add marked
        self.selprint(allsel,pref="ALL04")
        print "\n".join(self.get_confused(all_peak_db,pref="CMS04"))
        self.append_istat(all_peak_db)

        #next, solo peaks with solo_models
        one_to_one_peaks = self.merge_masks([eval_class == 3,rt['istat'] == 0,psel])
        #sometimes a peak is killed earlier, include here for completeness
        one_to_one_mods =  self.merge_masks([eval_class == 3,rt['istat'] == 0,ssel])
        peaks_for_mods = rt['peaki'][one_to_one_mods]
        one_to_one = list(set(list(one_to_one_peaks) + list(peaks_for_mods)))
        for pi in one_to_one:
            valid_peak = True
            if rt['istat'][pi] != 0:
                ori_istat = rt['istat'][pi]
                valid_peak = False
            if rt['mod'][pi] == 0: #marked as 1to1 system, but model/peak not found
                if rt['istat'][pi] == 0:
                    rt['istat'][pi] = 901 #ASSIGN p, error, should have mod, doesn't
                    continue
            si = rt['modi'][pi]
            rank1,rank2 = self.explicit_rank(all_peak_db,[pi,si])
            print "HEAD_2_HEAD",self.i2r(pi),self.i2r(si),rank1,rank2
            if rank1[0] and rank2[0]: #both are processed
                wini=rank1[1]
                rupi=rank2[1]
                if rt['qual'][wini] == 0 and rt['qual'][rupi] == 0: #both low quality
                    rt['istat'][wini] = -9 #ASSIGN p or s 1to1 both poor, both killed
                    rt['istat'][rupi] = -9
                    continue
                if rank1[2] == rank2[2]: #picks same
                    if rt['qual'][wini] >= rt['qual'][rupi]: #agree on quality and score
                        rt['istat'][wini] = rank1[2] #ASSIGN s or p, 1,2,3,4 picks match, winner gets X, loser -11
                        rt['istat'][rupi] = -11  #ASSIGN s or p, picks match, winner gets 1, loser -11
                    else: #scores disagree with quality (likely flagged peak)
                        rt['istat'][wini] = 10 + rank1[2] #ASSIGN s or p, 11,12,13,14 picks match, winner gets X, loser -21
                        rt['istat'][rupi] = -21  #ASSIGN s or p, picks match, winner gets 1, loser -21

                else: #disputed picks
                    if rt['qual'][wini] == 0:
                        rt['istat'][wini] = -500 # ASSIGN p or s 1to1 disputed, p1 failed quality p2 picked
                        rt['istat'][rupi] = 500 + rt['pick'][rupi] #ASSIGN p or s, 1to1 dispute, p1 failed p2 picked
                    elif rt['qual'][rupi] == 0:
                        rt['istat'][rupi] = -500
                        rt['istat'][wini] = 500 + rt['pick'][wini]
                    else:
                        #pick by quality difference ~80% match label with quality difference >= 1
                        if rt['qual'][pi] - rt['qual'][si] > 0:
                            rt['istat'][pi] = 510 + rt['pick'][pi] # ASSIGN p or s in 1to1 dispute, better by quality,510 + pick
                            rt['istat'][si] = -510 # ASSIGN p or s in 1to1 dispute, loser by quality
                        elif rt['qual'][si] - rt['qual'][pi] > 0:
                            rt['istat'][si] = 510 + rt['pick'][pi]
                            rt['istat'][pi] = -510
                        else: #undetermined
                            if rt['istat'][wini] == 0:
                                rt['istat'][wini] = 701 #ASSIGN p, unresolved 1to1 dispute, better of pair
                            if rt['istat'][rupi] == 0:
                                rt['istat'][rupi] = -701 #ASSIGN s, unresolved 1to1 dispute, lessor of pair
            else: #someother mess, unprocessed peak
                if rt['istat'][pi] == 0:
                    rt['istat'][pi] = -711 #ASSIGN p or s, failed 1to1 check, one peak not processed?
                if rt['istat'][si] == 0:
                    rt['istat'][si] = -711
            if not valid_peak:
                rt['istat'][pi] = ori_istat
        #peaks with models marked as better/worse match/nomatch
        self.selprint(allsel,pref="ALL05")
        print "\n".join(self.get_confused(all_peak_db,pref="CMS05"))
        self.append_istat(all_peak_db)

        ma_groups = eval_class > 4 #all but ma solmod
        ma_groupn = list(set(list(rt['group'][ma_groups])))
        #uniquify groups/clusters,start with one peak from each
        unique_clists = self.uniq_groups(ma_groupn)
        unique_idpeak = list(clist[0] for clist in unique_clists)
        
        clust_lists = [[],[],[],[],[]]
        #breakdown clusters 
        for pi in unique_idpeak:
            gsel = self.system_mat[pi] > 0
            alli = list(np.nonzero(gsel)[0])
            tocheck = self.merge_masks([gsel,proc,rt['istat']==0])
            if cnz(tocheck) == 0: 
                clust_lists[0].append(alli)
                print "CLUST_CULL1",list(self.i2r(pi) for pi in alli)
                continue
            checki = list(np.nonzero(tocheck)[0])
            all_ranks = self.explicit_rank(all_peak_db,checki)  
            ranked_i = list(rank[1] for rank in all_ranks)
            ranked_picks = list(rank[2] for rank in all_ranks)
            culled_ranks = list(rank for rank in all_ranks if (rank[3]> 0.0 and rank[2] > 0)) #only good scores
            if len(culled_ranks) == 0:
                clust_lists[0].append(alli)
                print "CLUST_CULL2",list(self.i2u[pi] for pi in alli)
                print "CLUST_CULL2A",all_ranks
                continue
            culled_i = list(rank[1] for rank in culled_ranks)
            culled_t3 = list(pi for pi in culled_i if rt['model'][pi]==3)
            culled_t4 = list(pi for pi in culled_i if rt['model'][pi]==4)
            if len(culled_t3) > 0:
                label = rt['mlab'][culled_t3[0]]
            else:
                label = 0
                
            culled_picks = list(rank[2] for rank in culled_ranks)
            culled_scores = list(rank[3] for rank in culled_ranks)
            clust_type = 0
            counts = [0,0,0,0,0]
            elect = self.clust_elect(all_peak_db,list(rt['unal'][pi] for pi in culled_i))
            outstr = " ".join(str(x) for x in [label,culled_picks[0],len(ranked_i),len(culled_i),elect,
                                               "||"," ".join(str(x) for x in culled_picks),"||",
                                               " ".join(list("%4.2f" % scr for scr in culled_scores)),
                                               "||"," ".join(list("%13s" % self.i2r(pi) for pi in ranked_i))])
            if len(culled_picks) == 1: #usually very wrong, rescore on entire cluster
                clust_type = culled_picks[0]
                clust_lists[clust_type].append(alli)
                print "CLUST_MATCH0",outstr
            elif len(culled_picks) > 1:
                #all picks/scores agree
                if all(pick == culled_picks[0] for pick in culled_picks) and len(culled_picks) > 2:
                    clust_type = culled_picks[0]
                    clust_lists[clust_type].append(alli)
                    print "CLUST_MATCH1",outstr
                else:    
                    n_good = len(culled_picks)
                    best_pick = culled_picks[0]
                    for pick in culled_picks:
                        counts[pick] = counts[pick]+1
                    #only last peak in cluster of 5 or more is mismatch
                    if n_good > 4 and n_good - counts[best_pick] == 1 and culled_picks[-1] != best_pick:
                        clust_type = best_pick
                        clust_lists[clust_type].append(alli)
                        print "CLUST_MATCH2",outstr
                    #if lower scoring picks weaker than better picks
                    if counts[best_pick] > 1 and clust_type == 0:
                        nonmatch = np.nonzero(culled_picks !=  best_pick)[0]
                        if nonmatch.shape[0] > 0:
                            first_nonbest = nonmatch[0]
                            if first_nonbest > 3:
                                bestav = np.nanmean(culled_scores[0:first_nonbest])
                                if culled_scores[first_nonbest] < 0.17*bestav:
                                    clust_type = best_pick
                                    clust_lists[clust_type].append(alli)
                                    print "CLUST_MATCH3",outstr
                    #all water and metal picks, best is water, is water cluster
                    if n_good > 1 and counts[2] == 0 and counts[3] == 0 and best_pick == 1 and clust_type == 0:
                        clust_type = 1
                        clust_lists[clust_type].append(alli)
                        print "CLUST_MATCH4",outstr
                    #else use election with quality to pick winner, matches label ~90% 
                    if clust_type == 0:
                        clust_type = elect
                        clust_lists[clust_type].append(alli)  
                        print "CLUST_ELECT1",outstr
            else:#  lousy cluster, no positive scores
                clust_lists[0].append(alli)

        #sort,prune,cull clusters (2x nested)
        sorted_clists = [[],[],[],[],[]]
        for ctype,clist in enumerate(clust_lists):
            if ctype == 0:
                for ilist in clist:
                    sorted_clists[0].append(tuple(sorted(ilist,reverse=True)))
            else:
                for ilist in clist:
                    ranks = self.explicit_rank(all_peak_db,ilist,forpick=ctype)
                    sorted_clists[ctype].append(tuple(rank[1] for rank in ranks))
        unique_clists = []
        for clist in sorted_clists:
            unique_clists.append(list(set(clist)))

        #assign clusters
        for cind,clist in enumerate(unique_clists):
            clust_type = cind
            #lousy, noise cluster
            if clust_type == 0:
                for ranked_i in clist:
                    print "CLUST_JUNK",list(self.i2r(pi) for pi in ranked_i)
                    for pi in ranked_i:
                        if rt['istat'][pi] == 0:
                            rt['istat'][pi] = 88 #ASSIGN p or s cluster undetermined
            #water cluster
            elif clust_type == 1: 
                for ranked_i in clist:
                    ranked_i = list(ranked_i)
                    true_w,undec_w,not_w = self.prune_sat_wat(all_peak_db,ranked_i)
                    if len(not_w) > 0:
                        new_elect = self.clust_elect(all_peak_db,list(rt['unal'][pi] for pi in not_w))
                        new_pick = rt['mlab'][not_w[0]]
                        if new_pick in [2,3]: #some core cluster not water
                            to_reassign = not_w + undec_w
                            new_ranks = self.explicit_rank(all_peak_db,to_reassign,forpick=new_pick)
                            unique_clists[new_pick].append(tuple(rank[1] for rank in new_ranks))
                            ranked_i = true_w
                    if len(ranked_i) > 0:
                        rt['istat'][ranked_i] = -101 # ASSIGN p or s wat clust default kill
                        to_check = list(pi for pi in ranked_i if rt['proc'][pi] == 1)
                        keep,kill,stray = self.prune_water(all_peak_db,to_check)
                        rt['istat'][keep] = 101 #ASSIGN p or s as w for water cluster breakdown
                    
            #sulfate cluster
            elif clust_type == 2:
                for ranked_i in clist:
                    ranked_i = list(ranked_i)
                    sat_w,undec_w,not_w = self.prune_sat_wat(all_peak_db,ranked_i)
                    rt['istat'][ranked_i] = 200 #ASSIGN p or s marked as SO4 cluster placeholder
                    if len(not_w)> 0:
                        ranked_as_s = list(pi for pi in not_w if rt['proc'][pi] == 1)
                        best_s = ranked_as_s[0]
                        best_sg = rt['group'][best_s]
                        if rt['model'][best_s] == 3: #model is best
                            if rt['mlab'][best_s] == 2: #also already S (validted)
                                gsel = self.ingroup_mat[best_sg]
                                best_unrg = self.expand_to_rg(all_peak_db,self.i2u[best_s])
                                for n_unal in best_unrg:
                                    gsel[self.u2i[n_unal]] = True
                                rt['istat'][gsel] = -230 # ASSIGN s mod lab 2 all in group
                                rt['istat'][best_s] = 231 #ASSIGN s mod lab 2 as keep for coords
                                ap_list = self.get_close_peaks(list(np.nonzero(gsel)[0]))
                                rt['istat'][ap_list] = -231 #ASSIGN p loser to s as s, kill

                            elif rt['mlab'][best_s] == 3: # built as other, check carefully
                                ranked_as_o = self.explicit_rank(all_peak_db,ranked_as_s,forpick=3)
                                o_scores = list(rank[3] for rank in ranked_as_o)
                                n_good_o = cnz(o_scores > 0)
                                if n_good_o > 1:
                                    #also valid as o, keep
                                    best_o = ranked_as_o[0][1]
                                    best_og = rt['group'][best_o]
                                    gsel = self.ingroup_mat[best_og]
                                    best_unrg = self.expand_to_rg(all_peak_db,self.i2u[best_o])
                                    for n_unal in best_unrg:
                                        gsel[self.u2i[n_unal]] = True

                                    rt['istat'][gsel] = -332 #ASSIGN s mod picked 2, but kept 3, kill sat
                                    rt['istat'][best_o] = 332 #ASSIGN s mod picked 2, but kept 3, best mark
                                    ap_list = self.get_close_peaks(list(np.nonzero(gsel)[0]))
                                    rt['istat'][ap_list] = -342 #ASSIGN p picked 2, kept 3, mark p to kill
                                else: #replace with SO4
                                    gsel = self.ingroup_mat[best_sg]
                                    best_unrg = self.expand_to_rg(all_peak_db,self.i2u[best_s])
                                    for n_unal in best_unrg:
                                        gsel[self.u2i[n_unal]] = True
                                    rt['istat'][gsel] = -331 # ASSIGN p or s of o-->s switch, default
                                    rt['istat'][best_s] = 251 #ASSIGN s mod lab 3 as keep for coords
                                    ap_list = self.get_close_peaks(list(np.nonzero(gsel)[0]))
                                    rt['istat'][ap_list] = -331 #ASSIGN p loser to s as s, kill
                        else:
                            gsel = self.ingroup_mat[best_sg]
                            best_unrg = self.expand_to_rg(all_peak_db,self.i2u[best_s])
                            for n_unal in best_unrg:
                                gsel[self.u2i[n_unal]] = True

                            rt['istat'][gsel] = -201 # ASSIGN p or s as s in cluster sat, kill
                            rt['istat'][best_s] = 201 # ASSIGN p or s as best s in cluster, build
                            ap_list = self.get_close_peaks(list(np.nonzero(gsel)[0]))
                            rt['istat'][ap_list] = -211 # ASSIGN p or s as loser of s cluster build
                                
                    else:
                        for pi in ranked_i:
                            if rt['istat'][pi] == 0:
                                rt['istat'][pi] = 299 # ASSIGN s clust, error of some sort
                    if len(sat_w) > 0:
                        rt['istat'][sat_w] = -102 # ASSIGN SO4 pick sat, default kill
                        to_check = list(pi for pi in sat_w if rt['proc'][pi] == 1)
                        keep,kill,stray = self.prune_water(all_peak_db,to_check)
                        for pi in keep:
                            if rt['istat'][pi] == 0:
                                rt['istat'][pi] = 101 

            #other cluster
            elif clust_type == 3: 
                for ranked_i in clist:
                    ranked_i = list(ranked_i)
                    rt['istat'][ranked_i] = 300 #ASSIGN p or s marked as OTH cluster placeholder
                    sat_w,undec_w,not_w = self.prune_sat_wat(all_peak_db,ranked_i)
                    if len(not_w) > 0:
                        ranked_as_o = list(pi for pi in not_w if rt['proc'][pi] == 1)
                        best_o = ranked_as_o[0]
                        best_sg = rt['group'][best_o]
                        solmod_olab = list(si for si in not_w if rt['mlab'][si] == 3 and rt['model'][si] == 3)
                        if len(solmod_olab) > 0: #keep sol as O
                            best_sol_o = solmod_olab[0]
                            best_sol_sg = rt['group'][best_sol_o]
                            gsel = self.ingroup_mat[best_sol_sg]
                            best_unrg = self.expand_to_rg(all_peak_db,self.i2u[best_sol_o])
                            for n_unal in best_unrg:
                                gsel[self.u2i[n_unal]] = True

                            ap_sel = self.get_close_peaks(list(np.nonzero(gsel)[0]))
                            rt['istat'][gsel] = -330 #ASSIGN s as valid o, keep entire structure, kill sat
                            rt['istat'][best_sol_o] = 331 #ASSIGN s as valid o, keep entire marker for best
                            rt['istat'][ap_sel] = -331 #ASSIGN p as valid o, but kill no build

                        elif rt['mlab'][best_o] == 2: 
                            #picked as oth, but modelled as so4,keep SO4
                            ranked_as_s = self.explicit_rank(all_peak_db,not_w,forpick=2)
                            reranked_as_o = self.explicit_rank(all_peak_db,not_w,forpick=3)
                            s_scores = list(rank[3] for rank in ranked_as_s)
                            o_scores = list(rank[3] for rank in reranked_as_o)
                            best_s_pdict = all_peak_db[self.i2u[ranked_as_s[0][1]]]
                            if best_s_pdict['score'] > 0 and best_s_pdict['cscore'] > 0:
                                keep_s = ranked_as_s[0][1]
                                keep_sg = rt['group'][keep_s]
                                gsel = self.ingroup_mat[keep_sg]
                                best_unrg = self.expand_to_rg(all_peak_db,self.i2u[keep_s])
                                for n_unal in best_unrg:
                                    gsel[self.u2i[n_unal]] = True

                                rt['istat'][gsel] = -261 #ASSIGN p or s as sat of s marked as o to keep s
                                rt['istat'][keep_s] = 261 #ASSIGN p or s as best s marked as o to keep s
                                ap_list = self.get_close_peaks(list(np.nonzero(gsel)[0]))
                                rt['istat'][ap_list] = -262 # ASSIGN p or s as loser of s cluster o to s build
                            else: #rebuild as o
                                gsel = self.ingroup_mat[rt['group'][best_o]]
                                best_unrg = self.expand_to_rg(all_peak_db,self.i2u[best_o])
                                for n_unal in best_unrg:
                                    gsel[self.u2i[n_unal]] = True
                                rt['istat'][gsel] = -310
                                rt['istat'][best_o] = 310

                        else: #rebuild as oth or build as oth
                            gsel = self.ingroup_mat[rt['group'][best_o]]
                            best_unrg = self.expand_to_rg(all_peak_db,self.i2u[best_o])
                            for n_unal in best_unrg:
                                gsel[self.u2i[n_unal]] = True
                            rt['istat'][gsel] = -310 #ASSIGN s or p as sat of best o not already o or good s, kill all
                            rt['istat'][best_o] = 310 #ASSIGN p or s as best in cluster for o not already o or good s mark to build
                    else:
                        for pi in ranked_i:
                            if rt['istat'][pi] == 0:
                                rt['istat'][pi] = 399 # ASSIGN o clust, error of some sort
                        
                    if len(sat_w) > 0:
                        rt['istat'][sat_w] = -103 # ASSIGN sad of OTH clust, default kill
                        to_check = list(pi for pi in sat_w if rt['proc'][pi] == 1)
                        keep,kill,stray = self.prune_water(all_peak_db,to_check)
                        for pi in keep:
                            if rt['istat'][pi] == 0:
                                rt['istat'][pi] = 101 


            elif clust_type == 4:
                for ranked_i in clist:
                    ranked_i = list(ranked_i)
                    rt['istat'][ranked_i] = 400 # ASSIGN p or s cluster ml1 placeholder
                    sat_w,undec_w,not_w = self.prune_sat_wat(all_peak_db,ranked_i)
                    if len(not_w) > 0:
                        ranked_as_m = list(pi for pi in not_w if rt['proc'][pi] == 1)
                        if len(ranked_as_m) > 0:
                            best_m = ranked_as_m[0]
                            candidate_m = [best_m,]
                            best_type = rt['model'][best_m]
                            if best_type == 3 and rt['peaki'][best_m] != 0: #solvent is best
                                candidate_m.append(rt['peaki'][best_m])
                            elif best_type == 4 and rt['modi'][best_m] != 0: 
                                candidate_m.append(rt['modi'][best_m])
                            valid_m = []
                            for cand_m in candidate_m:
                                pdict = all_peak_db[self.i2u[cand_m]]
                                c1 = pdict['c1']
                                if c1 < 2.8:
                                    valid_m.append(cand_m)
                            rt['istat'][candidate_m] = -401 # ASSIGN kill invalid metal as default
                            if len(valid_m) == 1:
                                rt['istat'][valid_m[0]] = 401 # ASSIGN best m in metal cluster
                            elif len(valid_m) > 1:
                                ranked_as_m = self.explicit_rank(all_peak_db,cand_m,forpick=4)
                                rt['istat'][ranked_as_m[0][1]] = 401 # ASSIGN accepted metal in cluster
                    #are the rest water?
                    ap_ilist = list(pi for pi in ranked_i if rt['istat'][pi] == 400)
                    keep,kill,stray = self.prune_water(all_peak_db,ap_ilist)
                    for pi in keep:
                        rt['istat'][pi] = 141 # ASSIGN metal cluster, coord water to keep
                    for pi in kill:
                        rt['istat'][pi] = -141 # ASSIGN metal cluster, coord water to kill
                    for pi in stray:
                        rt['istat'][pi] = 498 #some mystery

            else:
                for ranked_i in clist:
                    for pi in ranked_i:
                        if rt['istat'][pi] == 0:
                            rt['istat'][pi] = 988 # ASSIGN cluster error


        #after cluster breakdown
        self.selprint(allsel,pref="ALL06")
        print "\n".join(self.get_confused(all_peak_db,pref="CMS06"))
        self.append_istat(all_peak_db)

        #leftover unclaimed waters, mod=5, no splits, just distant
        sg_left = self.merge_masks([rt['istat'] == 8,rt['pick']==1,ssel])
        cluster_left = self.merge_masks(list(rt['istat'] == stat for stat in [200,300]),opp='u')
        stray_peaks = list(set(list(np.nonzero(sg_left)[0]) + list(np.nonzero(cluster_left)[0])))
        for si in stray_peaks :
            unal = self.i2u[si]
            pdict = all_peak_db[unal]
            margin_psel = self.pm_mat[si] >=0.0
            clp_id = []
            valid = False
            for clp in np.nonzero(margin_psel)[0]:
                clp_id.append((clp,self.pm_mat[si,clp]))
            clp_id.sort(key = lambda idt: idt[1])
            pnw = pdict['prob']
            den = pdict['2fofc_sigo_scaled']
            edc = pdict['edc']
            cc = pdict['cc']
            if len(clp_id) == 0:
                valid = True
            else:
                clpi,clpd = clp_id[0]
                if clpd > 2.3: #could be another valid water
                    valid = True
                elif rt['istat'][clp] < 0: #other peak not valid, examine
                    valid = True
            print "MARGIN_CLAIMS",self.u2r[unal],list("%s %5.3f %g" % (self.i2r(pi),d,rt['istat'][pi]) for pi,d in clp_id)
            if valid and pnw < 0.1 and den > 0.8 and rt['qual'][si] > 3 and cc+edc > 11:
                rt['istat'][si] = 81 # ASSIGN s, was 8, now valid water
            else:
                rt['istat'][si] = -81 # ASSIGN s, was 8, failed quality, killed
        #kill the rest
        sg_still_left = self.merge_masks([rt['istat'] == 8,ssel])
        rt['istat'][sg_still_left] = -82 # ASSIGN s, was 8, not split, not validated otherwise, killed

        #after moc=5 waters picked up
        self.selprint(allsel,pref="ALL07")
        print "\n".join(self.get_confused(all_peak_db,pref="CMS07"))
        self.append_istat(all_peak_db)


        
        #look by group:
        groups = list(set(list(rt['group'])))
        groups.sort()
        for group in groups:
            gsel = rt['group'] == group
            gseli = np.nonzero(gsel)[0]
            gsize = cnz(gsel)
            picks = list(rt['pick'][gsel])
            cpicks = list(rt['cpick'][gsel])
            labs =  list(rt['mlab'][gsel])
            stats =  list(rt['status'][gsel])
            istats =  list(rt['istat'][gsel])
            resats = " ".join(list("%13s" % self.i2r(pi) for pi in gseli))
            pkout  = " ".join(str(x) for x in picks)
            cpkout  = "CP "+str(cpicks[0])
            labout = " ".join(str(x) for x in labs)
            stout  = " ".join("%4s" % str(x) for x in stats)
            istout = " ".join("%4s" % str(x) for x in istats)
            print "GSCORE:"," || ".join([resats,pkout,cpkout,labout,stout,istout])
        


        #still_undecided = [8,99,198,199,299,399,499,701,702,801,802,803,901,911,988]
        #su_sel = self.merge_masks(list(rt['istat'] == istat for istat in still_undecided),opp="u")

        all_wi = self.collect_water(all_peak_db)
        for pi in all_wi:
            rt['npick'][pi] = 1
            pdict = all_peak_db[self.i2u[pi]]
            edc = pdict['edc']
            cc = pdict['cc']
            pnw = pdict['prob']
            fc = pdict['fc']
            den = pdict['2fofc_sigo_scaled']
            if edc + cc > 9 and pnw < 0.5 and fc in [0,1,4,5] and den > 0.5:
                rt['fmk'][pi] = 1
            else:
                rt['fmk'][pi] = -1
        to_build = list(np.nonzero(rt['fmk'] == 1)[0])
        allw_ranked = self.explicit_rank(all_peak_db,to_build)
        ranked_wi = list(rank[1] for rank in allw_ranked)
        w_pdb_str = self.get_sel_pdb(all_peak_db,ranked_wi,chainid="W")


        all_so4i = self.collect_so4(all_peak_db)
        for pi in all_so4i:
            rt['npick'][pi] = 2
            pdict = all_peak_db[self.i2u[pi]]
            edc = pdict['edc']
            cc = pdict['cc']
            rc = pdict['rc']
            pnw = pdict['prob']
            fc = pdict['fc']
            den = pdict['2fofc_sigo_scaled']
            #if not modelled as SO4 already, check all flags
            if rt['model'][pi] == 4 or (rt['model'][pi] == 3 and rt['mlab'][pi] != 2):
                if edc > 0 and cc > 0 and rc > 0 and abs(edc-cc) < 3 and edc + cc < 10 and pnw > 0.9 and fc in [0,1]:
                    rt['fmk'][pi] = 2
                else:
                    rt['fmk'][pi] = -2
            elif rt['model'][pi] == 3 and rt['mlab'][pi] == 2:
                    rt['fmk'][pi] = 2
            else:
                rt['fmk'][pi] = -9 #shouldn't ever be reached

        to_build = list(np.nonzero(rt['fmk'] == 2)[0])
        alls_ranked = self.explicit_rank(all_peak_db,to_build)
        ranked_si = list(rank[1] for rank in alls_ranked)
        s_pdb_str = self.get_sel_pdb(all_peak_db,ranked_si,chainid="S")

        all_othi = self.collect_oth(all_peak_db)
        for pi in all_othi:
            rt['npick'][pi] = 3
            valid_oth = True
            #oth much harder to weed out, no clear patterns
            pdict = all_peak_db[self.i2u[pi]]
            edc = pdict['edc']
            cc = pdict['cc']
            rc = pdict['rc']
            pnw = pdict['prob']
            fc = pdict['fc']
            den = pdict['2fofc_sigo_scaled']
            #if adding a new OTH, stringent quality
            if rt['istat'][pi] == 53:
                if rt['qual'][pi] < 5:
                    valid_oth = False
            #if switching from another label, stringent quality
            if rt['mlab'][pi] in [1,2,4]:
                if rt['qual'][pi] < 5:
                    valid_oth = False
            if edc not in [4,5] and  pnw > 0.5 and fc in [0,1,5] and valid_oth:
                rt['fmk'][pi] = 3
            else:
                rt['fmk'][pi] = -3

        to_build = list(np.nonzero(rt['fmk'] == 3)[0])
        allo_ranked = self.explicit_rank(all_peak_db,to_build)
        ranked_oi = list(rank[1] for rank in allo_ranked)
        o_pdb_str = self.get_sel_pdb(all_peak_db,ranked_oi,chainid="O")

        all_meti = self.collect_metal(all_peak_db)
        for pi in all_meti:
            rt['npick'][pi] = 4
            pdict = all_peak_db[self.i2u[pi]]
            c1 = pdict['c1']
            den = pdict['2fofc_sigo_scaled']
            if c1 < 2.4 and den > 2.0:
                rt['fmk'][pi] = 4
            else:
                rt['fmk'][pi] = -4
        to_build = list(np.nonzero(rt['fmk'] == 4)[0])
        allm_ranked = self.explicit_rank(all_peak_db,all_meti)
        ranked_mi = list(rank[1] for rank in allm_ranked)
        m_pdb_str = self.get_sel_pdb(all_peak_db,ranked_mi,chainid="M")

        all_pdb_str = w_pdb_str+s_pdb_str+o_pdb_str+m_pdb_str
        self.write_pdbstr(all_peak_db,all_pdb_str)
        #phenix iotbx doesn't seem to like my segids, so write strings directly
        #pdb_raw = open(self.outstem+"_sm_raw.pdb",'w')
        #print >> pdb_raw,all_pdb_str
        #pdb_raw.close()


        #peaks to build marked
        self.selprint(allsel,pref="ALL08")
        print "\n".join(self.get_confused(all_peak_db,pref="CMS08"))

        for unal in rt['unal']:
            is_hist = all_peak_db[unal].get('is_hist',[])
            self.peak_report(all_peak_db,unal)


    def prune_sat_wat(self,all_peak_db,ilist):
        rt = self.restab
        true_w,undec_w,not_w = [],[],[]
        new_ranks = []
        for pi in ilist:
            pdict = all_peak_db[self.i2u[pi]]
            if pdict['edc'] > 0 and pdict['cc'] > 0 and rt['proc'][pi] == 1:
                new_watp = np.log(pdict['prob']/(pdict['edc']*pdict['cc']))
                if new_watp > -5.0:
                    not_w.append(pi)
                else:
                    true_w.append(pi)
            else:
                undec_w.append(pi)
        new_ranks.sort(key = lambda si: si[0], reverse=True)
        return true_w,undec_w,not_w


    def get_close_peaks(self,ilist):
        ap_list = []
        for si in ilist:
            i_apsel = self.merge_masks([self.pm_mat[si] >= 0,self.pm_mat[si] < 1.65])
            ap_list.extend(list(np.nonzero(i_apsel)[0]))
        ap_list = list(set(ap_list))
        by_group = []
        for pi in ap_list:
            p_group = self.restab['group'][pi]
            by_group.extend(list(np.nonzero(self.ingroup_mat[p_group])[0]))
        ap_list = list(set(by_group))
        return ap_list

    def collect_water(self,all_peak_db):
        rt = self.restab
        w_istat = list(k for k,v in self.istat_dict.iteritems() if v['out'] == 1)
        print "WATER SELECT",w_istat
        s_w_nomod_keep = self.merge_masks([rt['model'] == 3,rt['mlab'] == 1,rt['moc'] == 0,rt['istat']==7])
        w_sel = self.merge_masks(list(rt['istat'] == stat for stat in w_istat),opp='u')
        all_wat = self.merge_masks([w_sel,s_w_nomod_keep],opp='u')
        wati = list(np.nonzero(all_wat)[0])
        return wati

    def collect_so4(self,all_peak_db):
        rt = self.restab
        s_istat = list(k for k,v in self.istat_dict.iteritems() if v['out'] == 2)
        print "SO4 SELECT",s_istat
        s_sel = self.merge_masks(list(rt['istat'] == stat for stat in s_istat),opp='u')
        s_s_nomod_keep = self.merge_masks([rt['model'] == 3,rt['mlab'] == 2,rt['moc'] == 0,rt['istat']==7])
        all_so4 = self.merge_masks([s_sel,s_s_nomod_keep],opp='u')
        so4i = list(np.nonzero(all_so4)[0])
        return so4i

    def collect_oth(self,all_peak_db):
        rt = self.restab
        o_istat = list(k for k,v in self.istat_dict.iteritems() if v['out'] == 3)
        print "OTH SELECT",o_istat
        o_sel = self.merge_masks(list(rt['istat'] == stat for stat in o_istat),opp='u')
        s_o_nomod_keep = self.merge_masks([rt['model'] == 3,rt['mlab'] == 3,rt['moc'] == 0,rt['istat']==7])
        all_oth = self.merge_masks([o_sel,s_o_nomod_keep],opp='u')
        othi = list(np.nonzero(all_oth)[0])
        return othi

    def collect_metal(self,all_peak_db):
        rt = self.restab
        m_istat = list(k for k,v in self.istat_dict.iteritems() if v['out'] == 4)
        print "ML1 SELECT",m_istat
        m_sel = self.merge_masks(list(rt['istat'] == stat for stat in m_istat),opp='u')
        s_m_nomod_keep = self.merge_masks([rt['model'] == 3,rt['mlab'] == 4,rt['moc'] == 0,rt['istat']==7])
        all_met = self.merge_masks([m_sel,s_m_nomod_keep],opp='u')
        meti = list(np.nonzero(all_met)[0])
        return meti




    def uniq_groups(self,glist,explicit_lists=None):
        #uniquify groups/clusters
        #takes by group number, returns 
        if explicit_lists is None:
            unique_clists = []
            for group in glist:
                gsel = self.ingroup_mat[group]
                groupi = list(np.nonzero(gsel)[0])
                #sorted tuples hashable
                unique_clists.append(tuple(sorted(groupi)))
            unique_clists = list(set(unique_clists))
        else:
            unique_clists = explicit_lists
        #on occasion, peaks end up in multiple groups (eg: if linked by multi atom solvent)
        #keep in longest list, remove from rest
        unique_clists.sort(key = lambda clist: len(clist),reverse=True)
        placed_i = []
        new_clists = []
        for clist in unique_clists:
            unplaced = list(pi for pi in clist if pi not in placed_i)
            placed_i.extend(unplaced)
            new_clists.append(unplaced)
        new_clists.sort(key = lambda clist: len(clist),reverse=True)
        unique_clists = list(clist for clist in new_clists if len(clist) > 0)
        return unique_clists


    def split_resolve(self,all_peak_db):
        rt = self.restab
        #revisit sol without peaks but was processed (ancillary peaks)
        proc_sol_istat8 = list(set(list(rt['group'][rt['istat']==8])))
        uniq_glists = self.uniq_groups(proc_sol_istat8)
        for ilist in uniq_glists:
            si = ilist[0]
            if len(ilist) > 1 and rt['mlab'][si] == 1: #split input model (alt water)
                keep,kill,assp = [],[],[]
                ranked = self.explicit_rank(all_peak_db,ilist,forpick=1)
                for rank in ranked:
                    ri = rank[1]
                    proc = rank[0]
                    #if good w score, keep, else kill
                    if rank[3] > -1.0 and proc and rt['pick'][ri] == 1 and rt['qual'][ri] > 3 and rt['fc'][ri] == 0:
                        keep.append(ri)
                        pdict = all_peak_db[self.i2u[ri]]
                        assp_peaks = list(self.u2i[u] for u,d in pdict['sol_mod'])
                        assp.extend(assp_peaks)
                    else:
                        kill.append(ri)
                if len(keep) > 1:
                    keep2,kill2,stray = self.prune_water(all_peak_db,keep)
                else:
                    keep2 = keep
                    kill2 = kill
                for si in keep2:
                    rt['istat'][si] = 91 #ASSIGN water alt kept
                for si in kill2:
                    rt['istat'][si] = -91 #ASSIGN water alt killed
                    
                print "ALTW",list(self.i2r(si) for si in keep2),"|<keep|kill>|",list(self.i2r(si) for si in kill2),
            elif len(ilist) == 1: #solo atom
                #peaks within range, but do not claim si as model
                margin_peaks = list(np.nonzero(self.pm_mat[si]>=0.0)[0])
                #really alone, was omitted and processed, but no association with fofc
                #density peak, likely weak water, but check anyway
                if len(margin_peaks) == 0:
                    pdict = all_peak_db[self.i2u[si]]
                    den = pdict['2fofc_sigo_scaled']
                    if rt['pick'][si] == 1 and rt['qual'][si] > 5 and rt['fc'][si] in [0,1,6,7] and den>1.0:
                        rt['istat'][si] = 77 #ASSIGN s moc=5, keep as as w
                        continue
                    else:
                        rt['istat'][si] = -66 #ASSIGN s moc=5, kill likely weak
                        continue
                else: #there is a peak around, have a closer look
                    punal = rt['unal'][si]
                    pdict=all_peak_db[punal]
                    #closest peak
                    pcont1 = pdict.get('peak_contacts',[{'distance':6.01},])[0]
                    cont_pu = pcont1['unal']
                    c_pdict = all_peak_db[cont_pu]
                    peak_mflag = c_pdict['mflag']
                    c_sc1_u = list(u for u,d in c_pdict['sol_mod'])
                    closest_peakmod_ud = None
                    #find the solvent to which closest peak is associated
                    for cont in pdict['sol_contacts']:
                        if cont['unal'] in c_sc1_u:
                            closest_peakmod_ud = (cont['unal'],cont['distance'])
                            break
                    if closest_peakmod_ud is not None:
                        osu,dist = closest_peakmod_ud
                        # mflag = 5, both solvent are similar distance from peak, si was not claimed, osu was
                        if peak_mflag == 5: 
                            cpi = self.u2i[cont_pu]
                            osi = self.u2i[osu]
                            split_list = [si,cpi,osi] #find best out of 3, solvent, closest peak, or other solvent
                            ranked = self.explicit_rank(all_peak_db,split_list,forpick=1)
                            wini = ranked[0][1]
                            if wini == cpi and dist < 3.1: #peak is best, resolve 2 split sol into 1 peak
                                rt['istat'][cpi] = 61 #ASSIGN p, winner of split, kill two sol
                                rt['istat'][si] = -67 #ASSIGN s, loser of split, peak kept
                                rt['istat'][osi] = -67 
                            else: #solvent are better, kill peak, keep split
                                rt['istat'][cpi] = -61 #ASSIGN p, loser of split, kill
                                rt['istat'][si] = 67 #ASSIGN s, keep split
                                rt['istat'][osi] = 67 #ASSIGN s, keep split
                        elif peak_mflag == 3: #si is further away than osi
                            if rt['istat'][si] == 0:
                                rt['istat'][si] = -68 #ASSIGN s, far from peak/sol, 
                    else:
                        if rt['istat'][si] == 0:
                            rt['istat'][si] = 802 #ASSIGN s, moc=5, but mflag not 5 or 3, tbd
            else:
                print "ERROR, no peaks in ilist for mod3/5 split check!",self.i2r(si)
                

    def get_sel_pdb(self,all_peak_db,ilist,chainid=None):
        rt = self.restab
        pdb_records = []
        towrite = list(self.i2u[pi] for pi in ilist)
        isd = self.istat_dict
        s_tobuild = "SO4"
        oneg_tobuild = "CL"
        opos_tobuild = "NA"
        o_tobuild = oneg_tobuild
        m_tobuild = "MG"
        for uind,unal in enumerate(towrite):
            pdict = all_peak_db[unal]
            pi = self.u2i[unal]
            istat = rt['istat'][pi]
            if istat not in isd.keys():
                ist[istat] = {"out":1, "xyz":0,"dest":("err","XX"),"verdict":"iStat Error"}
            resid = uind+1

            if chainid is None:
                chain = isd[istat].get('dest',("err","XX"))[1]
            else:
                chain = chainid
            occ = 1.0 
            bfac = 30.0 
            molout = isd[istat].get("out",1)
            coord_source = isd[istat].get('xyz',0)
            model = pdict['model']
            label = rt['mlab'][pi]
            resn,chainres,atom = pdict['resat'].split("_")
            ori_resname = resn.strip()
            ori_rid = str(pdict['resid'])
            ori_chain = str(pdict['chainid'])
            #fake segid to keep track of atom origin
            #seems to confuse pymol, coot, and phenix (wow!), so try appending
            fake_segid = "   "+pdict['db_id']
            get_an = lambda resat: resat.split("_")[2].strip()
            xyzin = []
            resname = "UNK"
            counts = [0,0,0,0]
            if molout == 1:
                resname = "HOH"
                score = pdict['prob_data'][1,0]
                if coord_source == 1:
                    xyzin.append(("O",pdict['wat_2fofc_ref_oricoords']))
                else:
                    xyzin.append(("O",pdict['coord']))
                    
                print "BUILD",1,"%13s" % self.u2r[unal],"STAT","%4d" % istat,"TYPE",model,"PL",rt['pick'][pi],label,"SS","%4d" % rt['status'][pi],"%4.2f" % score
                counts[0] = counts[0] + 1
            elif molout == 2:
                if ori_resname == "PO4":
                    s_tobuild = ori_resname #switch to PO4 if PO4 found in structure
                score = pdict['prob_data'][1,1]
                if model == 3 and label == 2: #originally built as S, keep coords
                    s_ulist = list(unal for unal in pdict['clust_mem'])
                    xyzin.extend(list((get_an(all_peak_db[sunal]['resat']),(all_peak_db[sunal]['coord'])) for sunal in s_ulist))
                else: #use refined coordinates in original setting
                    xyzin.extend(pdict['so4_2fofc_ref_oricoords'])
                print "BUILD",2,"%13s" % self.u2r[unal],"STAT","%4d" % istat,"TYPE",model,"PL",rt['pick'][pi],label,"SS","%4d" % rt['status'][pi],"%4.2f" % score
                resname = s_tobuild
                counts[1] = counts[1] + 1
            elif molout == 3:
                score = pdict['prob_data'][1,2]
                mm1 = pdict['mm_contacts'][0]
                print "OUTOTH",pdict['orires'],pdict['charge'],mm1['name'],mm1['element'],mm1['resname'],mm1['distance']
                if label == 3 and model == 3: #keep original coordinates
                    o_ulist = list(unal for unal in pdict['clust_mem'])
                    resname = ori_resname
                    xyzin.extend(list((get_an(all_peak_db[ounal]['resat']),(all_peak_db[ounal]['coord'])) for ounal in o_ulist))
                elif model == 4 and label == 3: #cannot yet build oth, keep original
                    s_pdict = all_peak_db[self.i2u[rt['modi'][pi]]]
                    o_ulist = list(unal for unal in s_pdict['clust_mem'])
                    resname = ori_resname
                    xyzin.extend(list((get_an(all_peak_db[ounal]['resat']),(all_peak_db[ounal]['coord'])) for ounal in o_ulist))
                else: #build element as generic other (placeholder)
                    xyzin.append((o_tobuild,pdict['wat_2fofc_ref_oricoords']))
                    resname = o_tobuild
                print "BUILD",3,"%13s" % self.u2r[unal],"STAT","%4d" % istat,"TYPE",model,"PL",rt['pick'][pi],label,"SS","%4d" % rt['status'][pi],"%4.2f" % score
                counts[2] = counts[2] + 1
            elif molout == 4:
                score = pdict['prob_data'][1,3]
                if label == 4 and model == 3:
                    xyzin.append((pdict['orires'],pdict['coord']))
                    resname = ori_resname
                    m_tobuild = resname
                elif model == 4 and label == 4:
                    xyzin.append((pdict['orires'],pdict['wat_2fofc_ref_oricoords']))
                    s_pdict = all_peak_db[self.i2u[rt['modi'][pi]]]
                    resn2,chainres2,atom2 = s_pdict['resat'].split("_")
                    resname = resn2.strip()
                    m_tobuild = resname
                else: #build generic magnesium ( most common with zinc, but less troublesome?)
                    xyzin.append((m_tobuild,pdict['wat_2fofc_ref_oricoords']))
                    resname = m_tobuild
                print "BUILD",4,"%13s" % self.u2r[unal],"STAT","%4d" % istat,"TYPE",model,"PL",rt['pick'][pi],label,"SS","%4d" % rt['status'][pi],"%4.2f" % score
                counts[3] = counts[3] + 1

            print "COORDS:"
            for cout in xyzin:
                print "   ",cout
            for an,coord in xyzin:
                serial = len(pdb_records)+1
                x,y,z = coord
                if an in self.ppio.common_elem or an in self.ppio.common_met:
                    elem = an.strip()
                else:
                    elem = an.strip()[0]
                atrec = self.pput.write_atom(serial,an,"",resname,chain,resid,"",x,y,z,occ,bfac,elem,"")
                atrec = atrec.strip()+fake_segid+"\n"
                pdb_records.append(atrec)
            pdict['pdb_out'] = pdb_records[-1][17:26]
        pdbstr = "".join(pdb_records)
        return pdbstr
                        

    def write_pdbstr(self,all_peak_db,pdb_str,outstem="solvent_model.pdb"):
        cs = all_peak_db[-8861610501908601326]['info']['symmetry']
        output_hier = iotbx.pdb.input(source_info=None,lines=flex.split_lines(pdb_str))
        outfile = self.outstem+"_"+outstem
        output_hier.write_pdb_file(file_name = outfile,crystal_symmetry=cs,append_end=True,anisou=False)


    def write_ref_pdb(self,all_peak_db):
        cs = all_peak_db[-8861610501908601326]['info']['symmetry']
        pdb_records=[]
        towrite = list(row['unal'] for row in self.restab if row['model'] == 4)
        towrite.sort(key = lambda unal: all_peak_db[unal]['db_id'])
        for uind,unal in enumerate(towrite):
            pdict = all_peak_db[unal]
            serial = len(pdb_records)+1
            resid = pdict['resid']
            wx,wy,wz = pdict['wat_2fofc_ref_oricoords']
            atrec = self.pput.write_atom(serial,"O","","HOH","P",resid,"",wx,wy,wz,1.0,35.0,"O","")
            pdb_records.append(atrec)
            for an,coord in pdict['so4_2fofc_ref_oricoords']:
                serial = len(pdb_records)+1
                sx,sy,sz = coord
                elem = an.strip()[0]
                atrec = self.pput.write_atom(serial,an,"","SO4","P",resid,"",sx,sy,sz,1.0,35.0,elem,"")
                pdb_records.append(atrec)
        pdb_str = "".join(pdb_records)
        output_hier = iotbx.pdb.input(source_info=None,lines=flex.split_lines(pdb_str))
        outstem = self.outstem
        output_hier.write_pdb_file(file_name = outstem+"_refined_coords.pdb",crystal_symmetry=cs,append_end=True,anisou=False)


    def get_istat_dict(self):
        #actions, information, etc, for pdb writing
        # out = output atom/molecule 0=pick,1-4 = wsom
        # xyz = coordinate source
        #     0 = original input pdb coords
        #     1 = refined coords w2fofc
        # dest = destination (file_suffix,chain)
        # verdict = explanation of peak fate
        


        istat_dict={
        -711:{"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"ERROR      --> Failed 1to1 check"},
        -701:{"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Undecided  --> Lessor score in disputed 1to1 check"},
        -510:{"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Loser by quality in 1to1 check"},
        -500:{"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Not valid in 1to1 check"},
        -401:{"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Outscored as ML1 by neighbor"},
        -342:{"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Peak picked as SO4, dropped for Mod OTH"},
        -332:{"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Picked as SO4, local cluster is OTH"},
        -331:{"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Valid OTH, not used"},
        -330:{"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Satellite of Valid OTH"},
        -310:{"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Satellite of SO4/OTH swap"},
        -262:{"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Satellite (cross-model) of picked OTH kept as SO4"},
        -261:{"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Satellite (same-model) of picked OTH kept as SO4"},
        -231:{"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Satellite (cross-model) of SO4 both Peak and Mod"},
        -230:{"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Satellite (same-model) of SO4 both Peak and Mod"},
        -211:{"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Lessor scoring Peak/Mod or satellite in SO4 pick"},
        -201:{"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Satellite (same-model) of Peak/Mod SO4"},
        -141:{"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Possible water coordinating ML1, failed quality tests"},
        -103:{"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Satellite water of OTH, poor/lessor score"},
        -102:{"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Satellite water of SO4, poor/lessor score"},
        -101:{"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Part of HOH cluster, poor/lessor score"},
        -91: {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> HOH build as alternate, failed quality tests"},
        -82: {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Mod HOH, unvalidated, unclaimed"},
        -81: {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Mod HOH, unclaimed, poor quality"},
        -68: {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Mod HOH split, but remote from Peak"},
        -67: {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> One of 2 Mod HOH split combined to single Peak"},
        -66: {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Mod unclaimed, poor quality"},
        -61: {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Peak with 2 Mod split, lessor quality"},
        -21: {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Peak/Mod pick match, loser by score"},
        -11: {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Peak/Mod pick match, loser by quality and score"},
        -9:  {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Both Peak and Mod failed 1to1 test"},
        -6:  {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Mod omitted, not recovered in FoFc map"},
        -5:  {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Peak without Mod, poor quality"},
        -4:  {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Peak/Mod noise/spurrious"},
        -3:  {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Peak/Mod likely backbone/peptide error"},
        -2:  {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Peak/Mod likely rotamer/alternate error"},
        -1:  {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Killed     --> Peak/Mod processing error or outlier rejection"},
        0:   {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Undecided  --> Null Peak or Mystery?"},
        1:   {"out":1, "xyz":0,"dest":("suffix","AA"),"verdict":"Passed     --> Peak/Mod (solo) with matching Peak/Mod pick as HOH, best scr/qual"},
        2:   {"out":2, "xyz":0,"dest":("suffix","AA"),"verdict":"Passed     --> Peak/Mod (solo) with matching Peak/Mod pick as SO4, best scr/qual"},
        3:   {"out":3, "xyz":0,"dest":("suffix","AA"),"verdict":"Passed     --> Peak/Mod (solo) with matching Peak/Mod pick as OTH, best scr/qual"},
        4:   {"out":4, "xyz":0,"dest":("suffix","AA"),"verdict":"Passed     --> Peak/Mod (solo) with matching Peak/Mod pick as ML1, best scr/qual"},
        7:   {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Kept       --> Mod not omitted, unscored"},
        8:   {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Undecided  --> Mod unclaimed by peak, no clear fate"},
       11:   {"out":1, "xyz":0,"dest":("suffix","AA"),"verdict":"Passed     --> Peak/Mod (solo) with matching Peak/Mod pick as HOH, by score"},
       12:   {"out":2, "xyz":0,"dest":("suffix","AA"),"verdict":"Passed     --> Peak/Mod (solo) with matching Peak/Mod pick as SO4, by score"},
       13:   {"out":3, "xyz":0,"dest":("suffix","AA"),"verdict":"Passed     --> Peak/Mod (solo) with matching Peak/Mod pick as OTH, by score"},
       14:   {"out":4, "xyz":0,"dest":("suffix","AA"),"verdict":"Passed     --> Peak/Mod (solo) with matching Peak/Mod pick as ML1, by score"},
        50:  {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"ERROR      --> Placeholder, shouldn't arise"},
        51:  {"out":1, "xyz":0,"dest":("suffix","AA"),"verdict":"Added      --> Peak without Mod picked as HOH"},
        52:  {"out":2, "xyz":0,"dest":("suffix","AA"),"verdict":"Added      --> Peak without Mod picked as SO4"},
        53:  {"out":3, "xyz":0,"dest":("suffix","AA"),"verdict":"Added      --> Peak without Mod picked as OTH"},
        54:  {"out":4, "xyz":0,"dest":("suffix","AA"),"verdict":"Added      --> Peak without Mod picked as ML1"},
        61:  {"out":1, "xyz":0,"dest":("suffix","AA"),"verdict":"Passed     --> Peak with 2 split Mod, split removed"},
        67:  {"out":1, "xyz":0,"dest":("suffix","AA"),"verdict":"Passed     --> Mod split with shared Peak, kept"},
        77:  {"out":1, "xyz":0,"dest":("suffix","AA"),"verdict":"Passed     --> Mod not claimed, but passes qualty test, kept"},
        81:  {"out":1, "xyz":0,"dest":("suffix","AA"),"verdict":"Passed     --> Mod not claimed, but valid water, kept"},
        88:  {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Undecided  --> Peak/Mod cluster, unable to determine"},
        91:  {"out":1, "xyz":0,"dest":("suffix","AA"),"verdict":"Passed     --> Mod with HOH alt, kept"},
        100: {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"ERROR      --> Placeholder for HOH cluster, shouldn't arise"},
        101: {"out":1, "xyz":0,"dest":("suffix","AA"),"verdict":"Passed     --> Peak/Mod as HOH in HOH cluster"},
        141: {"out":1, "xyz":0,"dest":("suffix","AA"),"verdict":"Passed     --> Peak/Mod HOH possibly coordinating a metal"},
        200: {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"ERROR      --> Placeholder for SO4 cluster, shouldn't arise"},
        201: {"out":2, "xyz":0,"dest":("suffix","AA"),"verdict":"Passed     --> Peak picked as SO4 in cluster"},
        231: {"out":2, "xyz":0,"dest":("suffix","AA"),"verdict":"Passed     --> Mod picked as SO4, best score, kept"},
        251: {"out":2, "xyz":1,"dest":("suffix","AA"),"verdict":"Swapped    --> Mod is OTH, replaced with SO4"},
        261: {"out":2, "xyz":1,"dest":("suffix","AA"),"verdict":"Passed     --> Picked as OTH, but kept as SO4 from model"},
        299: {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"ERROR      --> SO4 cluster not correctly processed"},
        300: {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"ERROR      --> Placeholder for OTH cluster, shouldn't arise"},
        310: {"out":3, "xyz":0,"dest":("suffix","AA"),"verdict":"Swapped    --> Mod is SO4, but scores suggest OTH"},
        331: {"out":3, "xyz":0,"dest":("suffix","AA"),"verdict":"Passed     --> Peak/Mod cluster agree on OTH, Mod kept"},
        332: {"out":3, "xyz":0,"dest":("suffix","AA"),"verdict":"Passed     --> Mod picked as SO4, but kept as OTH"},
        400: {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"ERROR      --> Placeholder for ML1 cluster, shouldn't arise"},
        401: {"out":4, "xyz":0,"dest":("suffix","AA"),"verdict":"Passed     --> Peak/Mod likely ML1"},
        498: {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Undecided  --> Peak/Mod in ML1 cluster, undecided"},
        499: {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"ERROR      --> Peak/Mod in ML1 cluster, processing error"},
        501: {"out":1, "xyz":0,"dest":("suffix","AA"),"verdict":"Passed     --> HOH where one of Peak/Mod failed 1to1 check"},
        502: {"out":2, "xyz":0,"dest":("suffix","AA"),"verdict":"Passed     --> SO4 where one of Peak/Mod failed 1to1 check"},
        503: {"out":3, "xyz":0,"dest":("suffix","AA"),"verdict":"Passed     --> OTH where one of Peak/Mod failed 1to1 check"},
        504: {"out":4, "xyz":0,"dest":("suffix","AA"),"verdict":"Passed     --> ML1 where one of Peak/Mod failed 1to1 check"},
        511: {"out":1, "xyz":0,"dest":("suffix","AA"),"verdict":"Passed     --> Peak/Mod disagree, HOH assigned by quality"},
        512: {"out":2, "xyz":0,"dest":("suffix","AA"),"verdict":"Passed     --> Peak/Mod disagree, SO4 assigned by quality"},
        513: {"out":3, "xyz":0,"dest":("suffix","AA"),"verdict":"Passed     --> Peak/Mod disagree, OTH assigned by quality"},
        514: {"out":4, "xyz":0,"dest":("suffix","AA"),"verdict":"Passed     --> Peak/Mod disagree, ML1 assigned by quality"},
        701: {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Undecided  --> Peak/Mod disagree, no clear choice"},
        802: {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Undecided  --> Mod unclaimed, no clear associated Peak"},
        803: {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Undecided  --> Mod group unclaimed, hot mess?"},
        901: {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"Undecided  --> Peak is missing model"},
        988: {"out":0, "xyz":0,"dest":("suffix","AA"),"verdict":"ERROR      --> Peak/Mod cluster processing error"}
        }
        return istat_dict



    def get_confused(self,all_peak_db,filter=None,pref="CMAT1"):
        #Generate "Confusion Matrix"
        #based on input labels
        rt = self.restab
        cnz = np.count_nonzero
        #scoreing output
        print "_"*79
        pops = ["ALL","HOH","SO4","OTH","ML1"]
        if filter is None:
            filter = rt['istat'] > 0
        labeled_and_qfiltered = self.merge_masks([rt['mlab'] > 0,filter])
        datain = rt[labeled_and_qfiltered]
        preds = datain['pick']
        labels = datain['mlab']

        have_updated_picks = datain['npick']> 0
        preds[have_updated_picks] = datain['npick'][have_updated_picks]
        cmat_out = []
        cmat_out.append( "CONFUSION MATRIX on %g Peak/Mod pairs:" % cnz(labeled_and_qfiltered))
        cmatrix = self.ppstat.cmatrix(labels,preds)
        cmat_out.append( "%s            COUNTS   %s" % (pref[0:5]," ".join("P-%s" % pops[i+1] for i in range(cmatrix.shape[0]-1))))
        for rowi,row in enumerate(cmatrix):
            if rowi > 0:
                cmat_out.append( "%s_C Lab-%3s | %5g | %s" % (pref[0:5],pops[rowi],row[0]," ".join("%5g" % x for x in row[1::]))) 
        cmat_out.append("-"*55)
        cm_by = np.zeros(cmatrix.shape)
        
        ppv = list(0.0 if cmatrix[0,i] == 0 else float(cmatrix[i,i])/cmatrix[0,i] for i in range(1,cmatrix.shape[0]))
        cmat_out.append("%s F1B           REC   %s <--PREC" % (pref[0:5]," ".join("%4.3f" % x for x in ppv)))
        f1_val = []
        for rowi,row in enumerate(cmatrix):
            f1_str = []
            if rowi > 0:
                for coli in range(1,row.shape[0]):
                    if row[0] > 0:
                        rec = float(row[rowi])/row[0]
                        if ppv[coli-1]>0.0 and coli == rowi:
                            f1score = 2.0*((rec*ppv[coli-1])/(rec+ppv[coli-1]))
                            f1_str.append("%4.3f" % f1score)
                            f1_val.append(f1score)
                        else:
                            f1_str.append("-----")
                    else:
                        rec = 0.0
                        f1_str.append("-----")

                cmat_out.append( "%s_N Lab-%3s | %4.3f | %s" % (pref[0:5],pops[rowi],rec," ".join(f1_str)))
        if len(f1_val) > 0:
            grand_f1 = np.nanmean(f1_val)
        else:
            grand_f1 = 0.0
        cmat_out.append( "%s_F1: %4.3f VAL %g RES: %3.2f POPCOUNTS %s" % (pref[0:5],grand_f1,len(f1_val),float(all_peak_db[self.i2u[2]]['resolution'])," ".join(str("%5g" % x) for x in cmatrix[:,0])))
        return cmat_out


    #
    #  Functions for output reports 
    #
    #

    def model_rep(self,pdict):
        rt = self.restab
        all_peak_db = pdict['peak_unal_db']
        mplist = pdict['sol_mod'] #list of unal,dist
        mflag = pdict['mflag'] % 20
        report = []
        if pdict['model'] == 4:
            if mflag in [0,4]:
                report.append("Unknown Solvent Model")
            if mflag == 1:
                report.append("Unmodeled")
            if mflag == 2:
                mp_ud = mplist[0]
                report.append("Modeled as %s (%2.1fA away)" % (all_peak_db[mp_ud[0]]['resat'],mp_ud[1]))
            if mflag == 3:
                mp_ud = mplist[0]
                report.append("Possibly modeled as %s, but %2.1fA away" % (all_peak_db[mp_ud[0]]['resat'],mp_ud[1]))
            if mflag == 5:
                report.append("Split models [%s] for single peak" % " ".join("%s %3.2fA" % (all_peak_db[u]['resat'],d) for u,d in mplist[0:2]))
            if mflag == 6:
                report.append("Possible split model: [%s]" % " ".join("%s %3.2fA" % (all_peak_db[u]['resat'],d) for u,d in mplist[0:2]))
            if mflag == 7:
                report.append("Ambiguous models for single peak  [%s]" % " ".join("%s %3.2fA" % (all_peak_db[u]['resat'],d) for u,d in mplist[0:2]))
            if mflag == 8:
                report.append("Reverse split with 2nd peak")
            if mflag == 9:
                report.append("Possible reverse split with 2nd peak")
            if mflag == 10:
                report.append("Ambiguous model")
        else:
            pi = self.u2i[pdict['unal']]
            claimed = int(rt['moc'][pi]/2) % 2 == 1
            if claimed:
                ass_pi = rt['peaki'][pi]
                ass_resat = self.i2r(ass_pi)
                ass_d = self.pm_mat[pi,ass_pi]
                report.append("Model claimed by peak %s %2.1fA away" % (ass_resat,ass_d))
            else:
                report.append("Model --unclaimed")
        return report

    def contact_rep(self,pdict):
        fc = pdict['fc']
        edc = pdict['edc']
        report = []
        shortest_worst = pdict['worst_mm']
        if fc == 0:
            report.append("  No contact issues.")
        if fc == 1: #special
            report.append("  Special Position!")

        if pdict['status'] == 401:
            report.append("--> ALT/ROT error at %s?" % shortest_worst['resat'])
        
        if pdict['status'] == 402:
            report.append("--> BACKBONE error/alt at %s?" % shortest_worst['resat'])
        if pdict['status'] in [403,413,431]:
            report.append("--> Spurious/Noise Peak?")

        if fc == 3 or fc == 4: #less bad contacts
            probs = pdict['prob_data']
            if probs[2][3] > 0.5: #metal kde/dir prob is > 50%
                if pdict['edc'] < 7 and pdict['fofc_sigo_scaled'] > 2.0:
                    report.append("Likely Metal Ion")
                else:
                    report.append("Likely Model Error at %s, possibly Metal Ion" % shortest_worst['resat'])
            elif pdict['score'] > 0 and pdict['status'] not in [401,402,403,413,431]:
                if shortest_worst['name'] not in ['N','C','CA','O']:
                    report.append("--> ALT/ROT error at %s?" % shortest_worst['resat'])
                else:
                    report.append("--> BACKBONE error/alt at %s?" % shortest_worst['resat'])
            else:
                report.append("--> Spurious/Noise Peak?")

        if fc == 5: #one close contact
            pdict['warnings'].append("CONT3")
            probs = pdict['prob_data']
            if probs[2][3] > 0.5: 
                if pdict['edc'] < 7:
                    report.append("Possible Metal Ion")
                else:
                    report.append("Likely Model Error at %s" % shortest_worst['resat'] )
            elif probs[2][0] > 0.5:
                if pdict['edc'] > 6:
                    report.append("--> Close Water")
                else:
                    report.append("Possible Close Water")
            else:
                report.append("Ambiguous, Inspect Manually?")
        if fc == 6: #weak
            report.append("Weak Density, likely noise peak")
        if fc == 7: #remote
            all_peak_db = pdict['peak_unal_db']
            anchor = all_peak_db.get('anchor',None)
            if anchor is not None:
                if all_peak_db['anchor']['model'] == 3:
                    report.append("REMOTE, connected to MM by %s" % pdict['anchor']['resat'])
                else:
                    report.append("REMOTE, no connection to MM")
        if pdict['model'] == 3:
            if pdict['label'] == 1 and pdict['c1'] > 3.5:
                report.append("  Model water missing anchor?")
        if pdict['model'] == 4:
            if pdict['edc'] - pdict['cc'] > 3:
                report.append("  Class mismatch, check for missing anchor?")
        return report

    def clust_rep(self,pdict):
        rep_str = []
        mf = pdict.get('mf',0)
        if mf == 0:
            return ["Unclustered/Solo"]
        clust_size = int(mf/100)
        issat = mf % 100 > 19
        clust_scr = mf % 10
        cmem = pdict.get('clust_mem',[])
        cent = pdict.get('clust_cent',0.0)
        all_peak_db = pdict['peak_unal_db']
        #collect info on cluster (scores and ids)
        mem_scr = []
        mem_id = []
        for unal in cmem:
            cpdict = all_peak_db[unal]
            mem_sc1 = cpdict.get('clust_cent',0.0)
            mem_id.append(cpdict['db_id'])
            mem_scr.append(mem_sc1)
        #break long list down by line
        mem_idstr = []
        batch = []
        added = 0
        while len(batch) < 6 and added < len(mem_id):
            batch.append(mem_id[added])
            added = added + 1
            if len(batch) == 6 or added == len(mem_id):
                if len(batch)>0:
                    mem_idstr.append(" ".join(x for x in batch))
                    batch = []
        #score ratio
        master_score = np.amax(mem_scr)
        if master_score > 0.0:
            rel_score = cent/master_score
        else:
            rel_score = -1.0 #error

        #marks as satellite or major:
        if issat:
            sat_stat = "SATELLITE/MINOR"
        else:
            sat_stat = "MAJOR/CENTRAL"
        #initial summary
        summary = "Cluster size: %s Cluster Score: %s Score in Cluster: %3.2f %s" % (clust_size,clust_scr,rel_score,sat_stat)
        rep_str.append(summary)
        #details
        if clust_scr > 3: #strong cluster
            if clust_size > 4: #big cluster
                rep_str.append("-->Large Cluster, possible unmodelled ligand?")
            else:
                rep_str.append("-->Small Cluster, small solvent or split peak?")
            rep_str.append("Associated peaks:")
            for mstr in mem_idstr:
                rep_str.append("  "+mstr)     
        else:
            rep_str.append("-->Weak cluster, water constellation?")
            for mstr in mem_idstr:
                rep_str.append("  "+mstr) 
        return rep_str


    def score_fmt(self,pdict):
        probs = np.clip(pdict['prob_data'],-9.9,99.9)
        p_as_str = []
        for row in probs:
            pstr = " ".join(("%4.1f" % x)[0:4] for x in row)+" "
            p_as_str.append(pstr)
        return p_as_str

    def score_report(self,pdict,resid_names,model_unal=None):
        rt = self.restab
        pi = self.u2i[pdict['unal']]
        lstr_outputs = []
        sstr_outputs = []
        pick_outputs = []
        probs = pdict.get('prob_data',None)
        pick = rt['pick'][pi]
        pick_name = resid_names[pick-1]
        if probs is None:
            lstr_outputs = ["Unscored",]
            sstr_outputs = ["Unscored",]
            pick_outputs = ["Unscored",]
        peak_scr = self.score_fmt(pdict)
        sstr_outputs.append("| ".join(x for x in peak_scr))
        mpdict = pdict['peak_unal_db'].get(model_unal,None)
        inc_mod = False
        if mpdict is not None:
            mod_scr = self.score_fmt(mpdict)
            inc_mod = True
        header = "                 %s" % "  ".join(resid_names)
        p_info = "      Pick: %s Qual: %s" % (pick_name,rt['qual'][pi])
        if inc_mod:
            mi = rt['modi'][pi]
            mpick = rt['mpick'][pi]
            mp_name = resid_names[mpick-1]
            header = header+"   "+"     %s" % "  ".join(resid_names)
            for oind in range(len(peak_scr)):
                peak_scr[oind] = peak_scr[oind]+" "*6+mod_scr[oind]
            p_info = p_info+"         Pick: %s Qual: %s" % (mp_name,rt['qual'][mi])

        lstr_outputs.append(header)
        lstr_outputs.append("   Flat_prior:  %s" % peak_scr[0] )
        lstr_outputs.append("   Bias_prior:  %s" % peak_scr[1] )
        lstr_outputs.append("   Popc_prior:  %s" % peak_scr[2] )
        pick_outputs.append(p_info)

        
        

        pdict['score_lstr'] = lstr_outputs
        pdict['score_sstr'] = sstr_outputs
        pdict['pick_info'] = pick_outputs

    def peak_report(self,all_peak_db,unal):
        rt = self.restab
        pi = self.u2i[unal]
        model = rt['model'][pi]
        paired = False
        si = 0
        if model == 4 and rt['mod'][pi] == 1:
            paired = True
            si = rt['modi'][pi]
        if model == 3 and rt['peaki'][pi] != 0:
            paired = True
            si = rt['peaki'][pi]
        pdict = all_peak_db[unal]
        if pdict['model'] not in [3,4]:
            pdict['full_rep'] = ""
            pdict['short_rep'] = ""
            return
        #DATA FOR OUTPUT
        fsig =  pdict['fofc_sigo_scaled']
        f2sig = pdict['2fofc_sigo_scaled']
        scr =  pdict['score']
        cscr = pdict['cscore']
        edc = pdict['edc']
        cc = pdict['cc']
        mf = pdict['mf']
        fc = pdict['fc']
        status = pdict['status']
        prob = pdict['prob']
        cdist = pdict['c1']
        ori = pdict['orires']
        
        if 'resid_names' not in pdict:
            mdict = pdict['master_dict']
            if 'kde' in mdict.keys():
                resid_names = mdict['kde']['populations']
            else:
                resid_names = pdict.get('resid_names',['HOH','SO4','OTH','ML1'])
        else:
            resid_names = pdict.get('resid_names',['HOH','SO4','OTH','ML1'])
        peakid = pdict['db_id']
        peakch = pdict['chainid']
        peakrn = pdict['resid']
        resat = pdict['resat']
        resname = resat.split("_")[0]
        probs = pdict['prob_data']
        preds = np.argsort(probs,axis=1)[:,::-1]+1
        pick1 = pdict['pick']
        pick_name = pdict['pick_name']
        mflag = pdict['mflag'] % 10
        label = pdict['label']
        lname = resid_names[label-1]

        pdict['contact_rep'] = self.contact_rep(pdict)    
        pdict['model_rep'] = self.model_rep(pdict)
        pdict['clust_rep'] = self.clust_rep(pdict)

        if pdict['model'] == 4:
            ptype = "Peak "
            otype = "Model"
        else:
            ptype = "Model"
            otype = "Peak "
        peak_desc = "%s %s (%s)" % (ptype,peakid,resat)
        if pdict['model'] == 4:
            if mflag in [2,3]:
                best_soli = rt['modi'][pi]
                bs_dist = self.pm_mat[pi,best_soli]
                best_sol = all_peak_db[self.i2u[best_soli]]
                mpeak_scored = best_sol['status'] not in [1,3,4,6,7]
                if mpeak_scored:
                    self.score_report(pdict,resid_names,model_unal=best_sol['unal'])
                    peak_desc = "%s %s <--> %s %s" % (ptype.strip(),resat,best_sol['resat'],otype.strip())
                else:
                    self.score_report(pdict,resid_names,model_unal=None)
                    peak_desc = peak_desc+"  no model scores for %s" % best_sol['resat']
            elif mflag in [5,6] and len(pdict['sol_mod']) > 1:
                bs1u,bs1d = pdict['sol_mod'][0]
                bs2u,bs2d = pdict['sol_mod'][1]
                bs1 = all_peak_db[bs1u]
                bs2 = all_peak_db[bs2u]
                bs1_max = np.amax(bs1.get('kde',0.0))
                bs2_max = np.amax(bs2.get('kde',0.0))
                if mflag % 10 == 5:
                    if bs1_max > bs2_max:
                        best_sol = bs1
                        bs_dist = bs1d
                    else:
                        best_sol = bs2
                        bs_dist = bs2d
                else: #pick closest
                    best_sol = bs1
                    bs_dist = bs2d
                if best_sol['status'] not in [1,3,4,6,7]:
                    mu = best_sol['unal']
                    peak_desc = peak_desc+" 2+ solvent models (split?), output for %s" % best_sol['resat']
                else:
                    mu = None
                    peak_desc = peak_desc+" 2+ solvent models (split?), no scores for %s" % best_sol['resat']
                self.score_report(pdict,resid_names,model_unal=mu)

            elif mflag == 4:
                bsu,bs_dist = pdict['sol_mod'][0]
                best_sol = all_peak_db[bsu]
                self.score_report(pdict,resid_names,model_unal=None)
                peak_desc = peak_desc+" closest solvent %s associated with another peak" % best_sol['resat']
            elif mflag % 10 == 1:
                best_sol,bs_dist = {"resat":"None"},6.01
                peak_desc = peak_desc+" has no solvent model"
                self.score_report(pdict,resid_names,model_unal=None)
            elif mflag % 10 in [7,8]:
                bsu,bs_dist = pdict['sol_mod'][0]
                best_sol = all_peak_db[bsu]
                peak_desc = peak_desc+" ambiguous solvent model, check %s" % best_sol['resat']
                self.score_report(pdict,resid_names,model_unal=None)
            elif mflag % 10 == 0:
                best_sol,bs_dist = {"resat":"Unk"},6.01
                peak_desc = peak_desc+" unknown solvent model" 
                self.score_report(pdict,resid_names,model_unal=None)
            bs_resat=best_sol['resat']
        else:
            if rt['peaki'][pi] !=0:
                ass_pi = rt['peaki'][pi]
                peak_desc = peak_desc+" is model for %s" % self.i2r(ass_pi)
                bs_resat = "[%s]" % self.i2r(ass_pi)
                bs_dist = self.pm_mat[pi,ass_pi]
                self.score_report(pdict,resid_names,model_unal=self.i2u[ass_pi])
            else:
                peak_desc = peak_desc+" is not associated with any Peak" 
                bs_resat = "---unclaimed"
                bs_dist = -1.0
                self.score_report(pdict,resid_names,model_unal=None)

            
        if paired:
            score_txt = "        Grid Scores:           %5s                    %5s" % (ptype.upper(),otype.upper())
        else:
            score_txt = "        Grid Scores:           %5s" % (ptype.upper())
        anchor = pdict.get('anchor',None)
        if anchor is not None:
            anc_resat = anchor['resat']
            adist = anchor['distance']
        else:
            anc_resat = "None"
            adist = -2.0
        #contacts
        #original model
        clori = pdict['contacts'][0]
        #existing solvent
        clsol = pdict['sol_contacts'][0]
        #macromolecule
        clmac = pdict['mm_contacts'][0]
        #other peaks, 0 is self?
        clpeak = pdict['peak_contacts'][0]
        if clpeak['resname'] == "NUL":
            pcout = "None"
        else:
            pcout = "%2.1fA" % clpeak['distance']

        #clusters/satellites
        if rt['model'][pi] == 4:
            if mf == 0 or mf % 10 == 0: 
                peak_clust = "No"
            else:
                peak_clust = "Yes"
        else:
            if pdict['mf'] != 0:
                peak_clust = "MA_Mod"
            else:
                peak_clust = "No"
        
        if pdict['fc'] == 1:
            peak_spl = "Yes"
        else:
            peak_spl = "No"

        if rt['mod'][pi] == 1 and rt['model'][pi] == 4:
            mod_flag = "Yes"
        elif rt['model'][pi] == 3 and rt['moc'][pi]  > 2:
            mod_flag = "Yes"
        else:
            mod_flag = "No"
            
        cls = "".join("%s" % x for x in [pdict['fc'],pdict['rc'],pdict['edc'],pdict['cc']])
        label_pick_npick_fate = str(rt['mlab'][pi])+str(rt['pick'][pi])+str(rt['npick'][pi])+str(max([rt['fmk'][pi],0]))

        p_verdict = ptype.upper()+": %s" % self.istat_dict[rt['istat'][pi]]['verdict']
        verdict = [p_verdict,]
        if paired:
            m_verdict = otype.upper()+": %s" % self.istat_dict[rt['istat'][si]]['verdict']
            verdict.append(m_verdict)
        if rt['fmk'][pi] > 0:
            final_pick = rt['fmk'][pi]
            pdb_out = pdict.get('pdb_out',resid_names[final_pick-1])
            verdict.append("FINAL: %s Built as %s in solvent model" % (ptype.strip(),"[ %s ]" % pdb_out.strip()))
        elif paired and rt['fmk'][si] > 0:
            si_final_pick = rt['fmk'][si]
            verdict.append("FINAL: Dropped in favor of %s %s as %s in solvent model" % (otype.strip(),self.i2r(si),resid_names[si_final_pick-1]))
        elif paired:
            verdict.append("FINAL: Both Peak and Model dropped")
        else:
            verdict.append("FINAL: Dropped --> failed quality tests")
        fate = p_verdict[0:11].strip()
        #ASSEMBLE OUTPUT STRINGS
        pout01 = "%s %s (%s) Status: %s %s Flags: %s PNW: %s Fate: %s " % (ptype,peakid,resat,status,rt['istat'][pi],cls,str(pdict['prob'])[0:5],fate)
        pout02 = "   DEN: FoFc %5.2f 2FoFc %5.2f ED_score %4.1f ED_class %s" % (fsig,f2sig,scr,edc)
        pout03 = "   ENV: Contact_score %4.1f Contact_class %s Anc_Dist %2.1fA" % (cscr,cc,adist)
        pout04 = "        CONTACTS Closest: %2.1fA %s  MacMol: %2.1fA %s Closest_Peak: %s" % (clori['distance'],clori['resat'],clmac['distance'],clmac['resat'],pcout)
        pout05 = "        -- Clashes: %s   Special: %s   Cluster: %s   Modelled/Claimed: %s" % (pdict['wt'],peak_spl,peak_clust,mod_flag)
        pout06 = self.rep_print(pdict['contact_rep'],12)
        pout07 = "   MODEL/PEAK PAIRING: %s" % peak_desc
        pout08 = self.rep_print(pdict['model_rep'],12)
        pout09 = "   SCORING:"
        pout10 = score_txt
        pout11 = self.rep_print(pdict['score_lstr'],8)
        pout12 = "        Results:   "+pdict['pick_info'][0]
        pout13 = "   CLUSTERING:" 
        pout14 = self.rep_print(pdict['clust_rep'],12)
        pout15 = "   VERDICT:"
        pout16 = self.rep_print(verdict,8)

        #short version
        fmtstr = ('{:>12} {:>12} LPNF {:4s} SCR {:5.1f} {:5.1f} Pnw {:3.2f} C1 {:3.2f} {:>12}' 
                  ' PAIR {:>12} {:3.2f} F{:4} {:>3} {:>3} STAT {:4d} {:4d} || {:>62}')
        short_p = pdict['score_sstr'][0]
        outstr = fmtstr.format(peakid,resat,label_pick_npick_fate,scr,cscr,prob,adist,anc_resat,
                               bs_resat,abs(bs_dist),cls,mf,mflag,status,rt['istat'][pi],short_p)
        pout99 = "SHORT "+outstr

        full_rep = []
        for rep_str in (pout01,pout02,pout03,pout04,pout05,pout06,pout07,pout08,pout09,pout10,
                        pout11,pout12,pout13,pout14,pout15,pout16):
            full_rep.append(rep_str)
        pdict['full_rep'] = full_rep
        pdict['short_rep'] = pout99


    def rep_print(self,str_list,indent):
        outst_list = []
        for string in str_list:
            outst_list.append(" "*indent+string)
        return "\n".join(outst_list)

    def report_preamble(self,all_peak_db):
        rt = self.restab
        cnz = np.count_nonzero

        null_peak = all_peak_db[-8861610501908601326]
        omit_mode = null_peak['info'].get('omit_mode','omitsw')

        peaks = rt['model'] == 4
        mods = rt['model'] == 3
        mp_out = rt['fmk'] > 0
        lab_match = np.logical_and(rt['mlab'] > 0,np.equal(rt['pick'],rt['mlab']))
        mp_match = np.logical_and(rt['pick'] > 0,np.equal(rt['pick'],rt['mpick']))
        new_peaks = self.merge_masks([peaks,rt['mod'] == 0,mp_out])
        killed = np.invert(mp_out)
        killed_peaks = self.merge_masks([peaks,killed])
        paired = np.logical_or(np.logical_and(peaks,rt['modi']>0),np.logical_and(mods,rt['peaki']>0))
        unpaired = np.logical_or(np.logical_and(peaks,rt['modi']==0),np.logical_and(mods,rt['peaki']==0))
        if omit_mode == 'valsol':
            ngroups = np.amax(rt['group'])
            unique_in = np.zeros(rt.shape[0],dtype=np.bool_)
            for groupn in np.arange(1,ngroups):
                gsel = rt['group'] == groupn
                tomark = np.nonzero(gsel)[0][0]
                unique_in[tomark] = True
            unique_in[rt['mlab']==1] = True
            unique_in[rt['model'] != 4] = False
        else:
            unique_in = self.merge_masks([rt['grank']==0,rt['grank']==1,rt['mlab']>0,mods],opp='uii')
        mod_groups = list(set(list(rt['group'][mods])))
        pop_in = list(np.logical_and(rt['mlab'] == x,unique_in) for x in [1,2,3,4])
        pop_sel = list(rt['fmk'] == x for x in [1,2,3,4])
        res_names = ["HOH","SO4","OTH","ML1"]
        pop_insel = lambda popsel,outsel: " ".join(("%4s" % cnz(np.logical_and(popsel,outsel==i+1)))[0:4] for i in range(4))
        pout_with_mod = self.merge_masks([peaks,paired,mp_out])
        pwm_labmatch =  self.merge_masks([pout_with_mod,lab_match])
        pwm_mismatch =  self.merge_masks([pout_with_mod,np.invert(lab_match)])
        mout_with_mod = self.merge_masks([mods,paired,mp_out])
        mod_uncl_out =  self.merge_masks([mods,mp_out,unpaired])
        cond_md_all = np.zeros(rt.shape[0],dtype=np.bool_)
        cond_md_pair = np.zeros(rt.shape[0],dtype=np.bool_)
        cond_md_unpair = np.zeros(rt.shape[0],dtype=np.bool_)
        cond_mout_all = np.zeros(rt.shape[0],dtype=np.bool_)
        cond_mout_pair = np.zeros(rt.shape[0],dtype=np.bool_)
        cond_mout_unpair = np.zeros(rt.shape[0],dtype=np.bool_)
        reject_byres = np.zeros(rt.shape[0],dtype=np.bool_)
        for group in mod_groups:
            gsel = rt['group'] == group
            gmark = self.merge_masks([rt['grank']==0,rt['grank']==1,gsel],opp='ui')
            if (rt['fmk'][gsel] == 0).all():
                cond_md_all[gmark] = True
                if (paired[gsel] == True).any():
                    cond_md_pair[gmark] = True
                else:
                    cond_md_unpair[gmark] = True
            if (rt['fmk'][gsel] > 0).any():
                cond_mout_all[gmark] = True
                if (paired[gsel] == True).any():
                    cond_mout_pair[gmark] = True
                else:
                    cond_mout_unpair[gmark] = True

        mout_wp_labmat =  self.merge_masks([cond_mout_pair,np.equal(rt['fmk'],rt['mlab'])])
        mout_wp_mismat =  self.merge_masks([cond_mout_pair,np.not_equal(rt['fmk'],rt['mlab'])])
        #TAB     POP  ACTION/DESC                  TOTAL              HOH  SO4  OTH  ML1
        str01 = "PProbe RUN at  %s" % time.ctime()
        str02 = "PEAK/MODEL INFORMATION:"
        str03 = "Peaks Input (by atom)               TOTAL                HOH  SO4  OTH  ML1:"
        str04 = "--Total                        --> %5s                   " % cnz(peaks)
        str05 = "--Built                        --> %5s                %s " % (cnz(np.logical_and(peaks,mp_out)),pop_insel(peaks,rt['fmk']))
        str06 = "----New/Added                  --> %5s                %s " % (cnz(new_peaks),pop_insel(new_peaks,rt['fmk']))
        str07 = "----With Model                 --> %5s                %s " % (cnz(pout_with_mod),pop_insel(pout_with_mod,rt['fmk']))
        str08 = "--------Agree with Mod Label   --> %5s                %s " % (cnz(pwm_labmatch),pop_insel(pwm_labmatch,rt['fmk']))
        str09 = "--------Disagree with Label    --> %5s                %s " % (cnz(pwm_mismatch),pop_insel(pwm_mismatch,rt['fmk']))
        str10 = "--Rejected                     --> %5s                   " % cnz(killed_peaks)
        str11 = ""
        str12 = ""
        str13 = "Models Input (by residue):"
        str14 = "--Total                        --> %5s                %s " % (cnz(unique_in),pop_insel(unique_in,rt['mlab']))
        str15 = "--Built (by residue)           --> %5s                %s " % (cnz(cond_mout_all),pop_insel(cond_mout_all,rt['npick']))
        str16 = "----Paired with Peak           --> %5s                %s " % (cnz(cond_mout_all),pop_insel(cond_mout_pair,rt['npick']))
        str17 = "--------Agree with input label --> %5s                %s " % (cnz(mout_wp_labmat),pop_insel(mout_wp_labmat,rt['mlab']))
        str18 = "--------Disagree with label    --> %5s                %s " % (cnz(mout_wp_mismat),pop_insel(mout_wp_mismat,rt['mlab']))
        str19 = "----Unclaimed                  --> %5s                %s " % (cnz(cond_mout_unpair),pop_insel(cond_mout_unpair,rt['mlab']))
        str20 = "--Dropped (by residue)         --> %5s                %s " % (cnz(cond_md_all),pop_insel(cond_md_all,rt['mlab']))
        str21 = "----With paired Peak           --> %5s                %s " % (cnz(cond_md_pair),pop_insel(cond_md_pair,rt['mlab']))
        str22 = "----Unclaimed                  --> %5s                %s " % (cnz(cond_md_unpair),pop_insel(cond_md_unpair,rt['mlab']))
        str23 = ""
        str23 = "Solvent Model Output:          --> %5s                %s " % (cnz(rt['fmk'] > 0),pop_insel(rt['fmk']>0,rt['fmk']))
        str24 = "-"*79
        str25 = "BREAKDOWN INS/OUTS"
        str26 = "--> applies to paired Peaks/Mods with labeled Model"
        str27 = "       will be all zero if no solvent input!"
        str28 = "\n".join(self.get_confused(all_peak_db,pref="CMAT",filter=rt['fmk']>0)[:-1])
        str29 = ""
        preamble = "\n".join([str01,str02,str03,str04,str05,str06,str07,str08,str09,str10,
                              str11,str12,str13,str14,str15,str16,str17,str18,str19,str20,
                              str21,str22,str23,str24,str25,str26,str27,str28,str29])
        return preamble


    def report_on_errors(self,all_peak_db):
        rt=self.restab
        cnz = np.count_nonzero
        proc_err = rt['istat'] == -1
        rot_alt_err = rt['istat'] == -2
        back_err = rt['istat'] == -3
        err_out = []
        if cnz(proc_err) == 0:
            err_out.append("   None!")
        else:
            for dat in rt[proc_err]:
                unal = dat['unal']
                pdict=all_peak_db[unal]
                resat = self.u2r[unal]
                dbid = pdict['db_id']
                errstr = "   PROCERR %s %s Failed during processing/analysis" % (dbid,resat)
                err_out.append(errstr)
        rot_alt_out = []
        if cnz(rot_alt_err) == 0:
            rot_alt_out.append("   None!")
        else:
            for dat in rt[rot_alt_err]:
                unal = dat['unal']
                pdict=all_peak_db[unal]
                resat = self.u2r[unal]
                dbid = pdict['db_id']
                anchor = pdict.get('anchor',None)
                if anchor is not None:
                    anc_resat = anchor['resat']
                    adist = anchor['distance']
                else:
                    anc_resat = "Unknown"
                    adist = -9.9
                errstr = "   ROTERR %s %s CLASH WITH %s %4.2fA" % (dbid,resat,anc_resat,adist)
                rot_alt_out.append(errstr)
        back_out = []
        if cnz(back_err) == 0:
            back_out.append("   None!")
        else:

            for dat in rt[back_err]:
                unal = dat['unal']
                pdict=all_peak_db[unal]
                resat = self.u2r[unal]
                dbid = pdict['db_id']
                anchor = pdict.get('anchor',None)
                if anchor is not None:
                    anc_resat = anchor['resat']
                    adist = anchor['distance']
                else:
                    anc_resat = "Unknown"
                    adist = -9.9
                errstr = "   BACKERR %s %s CLASH WITH %s %4.2fA" % (dbid,resat,anc_resat,adist)
                back_out.append(errstr)

        str01 = "PROCESSING ERRORS --> check manually?"
        str02 = "\n".join(err_out)
        str03 = "-"*79
        str04 = "SUSPECT RESIDUE ALTERNATE/ROTAMTER ERRORS --> check manually?"
        str05 = "\n".join(rot_alt_out)
        str06 = "-"*79
        str07 = "SUSPECT BACKBONE/PEPTIDE ERRORS --> check manually?"
        str08 = "\n".join(back_out)
        str09 = "-"*79
        return "\n".join([str01,str02,str03,str04,str05,str06,str07,str08,str09])

    def assemble_report(self,all_peak_db):
        rt=self.restab

        id_unal_list = []
        peaks = list(pdict for pdict in all_peak_db.values() if pdict['model'] == 4)
        mods = list(pdict for pdict in all_peak_db.values() if pdict['model'] == 3)
        for pdict in sorted(peaks, key = lambda pd: pd['db_id']):
            id_unal_list.append((pdict['db_id'],pdict['unal']))
        for pdict in sorted(mods, key = lambda pd: pd['db_id']):
            id_unal_list.append((pdict['db_id'],pdict['unal']))

        report_file=self.outstem+"_report.log"
        report = open(report_file,'w')
        short_outputs = []
        print >> report,self.report_preamble(all_peak_db)
        print >> report,"_"*79
        print >> report,"ERROR REPORT:"
        print >> report,"-"*79
        print >> report,self.report_on_errors(all_peak_db)
        print >> report,"_"*79
        print >> report,"FULL REPORT FOR EVERY PEAK/MOD"
        for dbid,unal in id_unal_list:
            print >> report,"-"*79
            for outstr in all_peak_db[unal]['full_rep']:
                print >> report, outstr
            short_outputs.append(all_peak_db[unal]['short_rep'])
        print >> report,"_"*79
        print >> report,"ABBREVIATED REPORT FOR EVERY PEAK/MOD (for fans of unix grep)"
        print >> report,"-"*79
        print >> report,"\n".join(short_outputs)
        report.close()

