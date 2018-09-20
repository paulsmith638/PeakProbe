from __future__ import division
#generic imports
import sys,math,ast,copy
import numpy as np
#PProbe imports
from PProbe_util import Util
from PProbe_dataio import DataIO
import cProfile

class Graph:
    def __init__(self,verbose=False):
        pass

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

    def pairind_gen(self,clust):
        #generates all unique non-self pairs of indices
        return list([v1,clust[j]] for i,v1 in enumerate(clust) for j in range(i+1,len(clust)))

    def clust_mean(self,clust,score_mat):
        #calculats mean edge weight for cluster (undirected)
        if len(clust) < 2: #no score for null or single peak
            return 0.0
        else:
            clust_pairs = self.pairind_gen(sorted(clust))
            return np.nanmean(list(score_mat[ind[0],ind[1]] for ind in clust_pairs))
            
    def graph_bfs(self,input_mat,start):
        #Breadth first search
        adj_mat = (np.abs(input_mat) < 1E-10).astype(np.int16)
        np.fill_diagonal(adj_mat,0)
        queue,visited = [start],[]
        while queue:
            vertex = int(queue.pop(0))
            if vertex not in visited:
                visited.append(vertex)
                neighbors = set(np.nonzero(adj_mat[vertex] > 0)[0])
                queue.extend(list(neighbors-set(visited)))
        print visited
        return visited
                         

    def graph_cut(self,all_sim_mat,smat_ind,use_svd=True):
        #graph partitioning
        #if not connected, find connected components by dfs
        #if connected, use Fiedler's vector cut
        if len(smat_ind) == 2:
            return [smat_ind[0],],[smat_ind[1],]
        sub_sim_mat = all_sim_mat.copy()
        sub_sim_mat = sub_sim_mat[smat_ind,:]
        sub_sim_mat = sub_sim_mat[:,smat_ind]

        sub_dim_mat = np.diag(np.nansum(sub_sim_mat,axis=0))
        sub_lmat = np.subtract(sub_dim_mat,sub_sim_mat)
        if not use_svd:
            #painfully slow sometimes?
            sub_eigval,sub_eigvect = np.linalg.eigh(sub_lmat)
        else:
            sub_eigvect,sub_eigval,v = np.linalg.svd(sub_lmat)
            eigsort = np.argsort(sub_eigval)[::-1]
            sub_eigvect = sub_eigvect[:,eigsort]
            sub_eigval = sub_eigval[eigsort]
        no_conn = np.count_nonzero(sub_eigval < 1E-10)
        paths = []
        if no_conn > 1: #contains disjoint sets
            internal_paths = []
            unfound_ind = list(x for x in range(len(smat_ind)))
            while len(unfound_ind) > 0:
                new_path = self.graph_dfs(sub_sim_mat,unfound_ind[0])
                internal_paths.append(new_path)
                unfound_ind = list(set(unfound_ind) - set(new_path))
            for path in internal_paths:
                paths.append(list(smat_ind[i] for i in path))
        else:
            partition = sub_eigvect[:,1] > 0
            invp = np.invert(partition)
            for partn in (partition,invp):
                paths.append(list(smat_ind[i] for i in range(len(smat_ind)) if partn[i]))
        return paths

    def graph_fwpath(self,dist_mat):
        #Floyd-Warshall all shortest path, dist_mat can have negative weights, but no negative cycles
        #can be very slow for large inputs
        sp_mat = np.ones(dist_mat.shape) * np.inf
        non_zero_sel = dist_mat > 0
        sp_mat[non_zero_sel] = dist_mat[non_zero_sel]
        np.fill_diagonal(sp_mat,0.0)
        no_pts = dist_mat.shape[0]
        for k in range(no_pts):
            for i in range(no_pts):
                for j in range(no_pts):
                    if sp_mat[i,j] > sp_mat[i,k] + sp_mat[k,j]:
                        sp_mat[i,j] = sp_mat[i,k] + sp_mat[k,j]
        return sp_mat
                                    
    #@do_cprofile
    def graph_dsp(self,dist_mat,targets,adj_cut=6.01):
        """
        shortest path using Dijkstra's algorighm, faster for sparse graphs
        which should always be the case with supra-adjacency matricies
        """
        import heapq as HQ
        adj_sel = dist_mat < adj_cut
        n_nodes = dist_mat.shape[0]
        adj_mat = np.ones(dist_mat.shape,dtype=np.int16)*30000
        adj_mat[adj_sel] = 1
        print "      D-SP on %s targets using %s nodes" % (len(targets),n_nodes)
        t_dist,t_path,t_hops = {},{},{}
        for node in targets:
            queue = []
            #mod, single queue with dist,node,hops
            HQ.heappush(queue,(0.0,node,0))
            visited = {node:0.0}
            hops = {node:0}
            nodes = list(i for i in range(n_nodes))
            path = {}
            node_bool = np.ones(len(nodes),dtype=np.bool_)
            while queue and nodes:
                udist,unode,uhop = HQ.heappop(queue)
                try:
                    while unode not in nodes:
                        udist, unode, uhop = HQ.heappop(queue)
                except IndexError:
                    break
                #remove method is bottleneck here
                #np nonzero actually 2x as fast
                node_bool[unode] = False
                #nodes.remove(unode)
                nodes = list(np.nonzero(node_bool)[0])
                neighbors = list(np.nonzero(adj_sel[unode] > 0)[0])
                for vnode in neighbors:
                    alt = udist + dist_mat[unode,vnode]
                    alth = uhop + adj_mat[unode,vnode]
                    if vnode not in visited or alt < visited[vnode]:
                        visited[vnode] = alt
                        HQ.heappush(queue,(alt,vnode,alth))
                        path[vnode] = unode
                        hops[vnode] = alth 
            t_dist[node] = visited
            t_path[node] = path
            t_hops[node] = hops
        return t_dist,t_path,t_hops


    def get_mlat(self,cont_db,unal_list):
        """
        Attempt to handle symmetry related contacts with a multi-layer adjacency tensor (MLAT)
        as per: Domenico et. al, "Mathematical formulation of Multi-layer Networks" (2013).
        For a network with n nodes and l layers, the mlat is a rank-4 tensor
        of dimentions n*n*l*l.  Here, each l is a sym-mate.  This tensor is initialized
        with each self layer mlat[:,:,h,h] as the usual graph adjacency matrix.  Sym connections
        are added symmetrically (undirected graph) with mlat[i,j,h,k] = mlat[i,j,k,h].  Various
        tensor contractions are used to get a complete contact representation.  In this scheme
        layer 0 is "central", like a star graph of graphs as there are no connectsion between
        any two layers with l>0 (not found, anyway).  So, in terms of adjacency, the multi-layer
        graph can be condensed to two layers, 0 = ASU, 1 = SYM.  However, I suspect that other 
        metrics like similarity might behave differently (e.g. multiple connections to the same peak by
        different symmetries would be reduced to "1" = adjacent).  Authors state that degree centrality
        not equivalent between full-rank centrality and projections.  However, the PMN projection centrality
        seems to be the same as the full-rank centrality here.  Perhaps a result of the unique nature
        of crystallographic contacts.

        Thus, will work with full mlat.  Note that what is returned is mlat with elements as contact
        distances initialized to 9.99, which is "out of range".
        The rank-r MLAT is also returned as a flattened 2d array as supra-adjacency matrix
        """
        #get list of contact lists
        init_cont = {unal:clist for (unal,clist) in cont_db.iteritems() if unal in unal_list}
        #find number of contacted symmates and use dict for indexing
        sym_n_list = list(set(list(cont['sym'] for clist in init_cont.values() for cont in clist)))
        sym_n_list.sort()
        sdict = {sym:index for (index,sym) in enumerate(sym_n_list)}
        #dict for lookup from tensor index to unal
        idict = {}
        udict = {}
        for index,unal in enumerate(init_cont.keys()):
            idict[index] = unal
            udict[unal] = index
        print "      Constructing Multi-Layer Adjacency Tensor (%s symops)" % len(sym_n_list)
        # initialize mlat
        #initially use raw distances, above 6 is out of range
        n_nodes = len(unal_list)
        n_layer = len(sym_n_list)
        mlat_raw = np.ones((n_nodes,n_nodes,n_layer,n_layer)) * 9.99
        for unal,clist in init_cont.iteritems():
            s_ind = udict[unal]
            #source sym should be 0, can be other for special positions
            #only peak contacts
            pclist = list(cont for cont in clist if (cont['model'] == 4 and cont['unal'] in unal_list))
            for cont in pclist:
                c_unal = cont['unal']
                c_sym = cont['sym']
                c_ind = udict[c_unal]
                c_symi = sdict[c_sym]
                dist = cont['distance']
                #contacts within asu
                if c_sym == 0:
                    for i in range(n_layer):
                        mlat_raw[s_ind,c_ind,i,i] = dist
                        mlat_raw[c_ind,s_ind,i,i] = dist
                else:
                    mlat_raw[s_ind,c_ind,0,c_symi] = dist
                    mlat_raw[c_ind,s_ind,0,c_symi] = dist
                    mlat_raw[s_ind,c_ind,c_symi,0] = dist
                    mlat_raw[c_ind,s_ind,c_symi,0] = dist
        #generate supra_adj_matrix, start with supra_dist?
        sam = np.ones((n_nodes*n_layer,n_nodes*n_layer))
        for sai in range(n_layer):
            for saj in range(n_layer):
                sam[n_nodes*sai:n_nodes*(sai+1),n_nodes*saj:n_nodes*(saj+1)] = mlat_raw[:,:,sai,saj]
        return mlat_raw,idict,sam
    #@do_cprofile
    def div_cluster(self,sim_mat):
        #iteratively cut graph by DFS and vector min-cut until each peak is its own tree
        #path is misnomer, cluster?
        all_ind = list(x for x in range(sim_mat.shape[0]))
        paths = [all_ind,]
        for path in paths:
            if len(path) > 1:
                new_paths = self.graph_cut(sim_mat,path)
                for npath in new_paths:
                    paths.append(npath)
        paths.sort(key = lambda x: len(x),reverse=True)
        return paths


    def graph_dfs(self,input_mat,start):
        #will take a similarity matrix, laplacian, or standard adj matrix
        #convert to integer boolean matrix
        #set diagonal to zero (hollow), no self-loops in graph
        adj_mat = (np.abs(input_mat) < 1E-10).astype(np.int16)
        np.fill_diagonal(adj_mat,0)
        stack,path = [start],[]
        while stack:
            vertex = int(stack.pop())
            if vertex in path:
                continue
            path.append(vertex)
            connect = list(np.nonzero(adj_mat[vertex] == 0)[0])
            #connect = list(int(x) for x in np.argwhere(adj_mat[vertex] == 0))
            for cvert in connect:
                stack.append(cvert)
        return path




    def eigvect_cent(self,adj_mat):
        adj_eval,adj_evect = np.linalg.eigh(adj_mat)
        e1=np.sign(np.nansum(adj_evect[:,-1]))*adj_evect[:,-1]
        return np.dot(e1,adj_mat)


    def local_eigcent(self,adj_mat,sorted_clist,local_peaks = 10):
        #for every peak in adj mat, take closest number local_peaks
        #create subadj matrix, compute eig centrality
        #returns 1D array of eigval
        #peaks close to special positions may appear multiply
        #and skew results
        local_eigc = np.zeros(adj_mat.shape[0])
        for index,row in enumerate(sorted_clist):
            cluster_ind = [index,]
            n_peaks = np.amin((row.shape[0],local_peaks)) #if less than specified peaks
            cluster_ind.extend(list(row[0:n_peaks]))
            adj_smat = adj_mat.copy()
            adj_smat = adj_smat[cluster_ind,:]
            adj_smat = adj_smat[:,cluster_ind]
            eigc = self.eigvect_cent(adj_smat)
            local_eigc[index] = eigc[0]
        return local_eigc


    def simsq_rbf(self,data_col,dmask,calc_sim=True,norm=True,gamma=None):
        #calculate similarity matrices using radial basis function
        if calc_sim:
            #calculate similarity  stat as element-wise difference squared
            num_peaks = len(data_col)
            gx,gy = np.mgrid[0:num_peaks,0:num_peaks]
            dsq_mat = np.square(np.subtract(np.array(data_col)[gx],np.array(data_col)[gy]))
        else: #if input is already sq_sim metric as matrix
            dsq_mat = data_col
        if norm:
            # scale to 0:1
            norm_mat = (dsq_mat - np.amin(dsq_mat))/(np.amax(dsq_mat) - np.amin(dsq_mat))
        else:
            #take as is
            norm_mat = dsq_mat
        ma_input = norm_mat
        #if not given, estimate gamma parameter as inv of std_deviation of valid data
        if gamma is None:
            gamma = 1.0/np.nanstd(ma_input)
        rbf = np.exp(-ma_input*gamma)
        scores = np.multiply(rbf,np.invert(dmask.astype(np.bool_)))
        return scores
