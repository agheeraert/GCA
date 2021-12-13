import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance, self_capped_distance
from tqdm import tqdm
from scipy.sparse import coo_matrix
from scipy.cluster.hierarchy import dendrogram, linkage, ward
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from itertools import combinations
import pandas as pd
import seaborn as sns
import pickle as pkl

mpl.rc('figure', fc = 'white')
mpl.rcParams.update({'font.size': 12})
plt.style.use('classic')
plt.rcParams['axes.facecolor']='white'
plt.rcParams['figure.facecolor']='white'


class GCA():
    """Generalized Contact Analysis (GCA)
    
    Computes the contact matrix of one or several trajectories and performs
    ... analyses
    
    It uses... 

    Parameters
    ----------
    cutoff: float, default = 5.0

    Attributes
    ----------
    Cutoff

    References
    ----------

    Examples
    --------
    """

    def __init__(self, cutoff=5.0, begin=0, end=-1, stride=1):
        self.cutoff = cutoff
        self.slice = slice(begin, end, stride)

    def standard_analysis(self, output_list=None, max_compo=4, **kwargs):
        """Follows a series of standard instructions for GCAnalysis 
        """


        if output_list == None:
            output_list = ['df_trajnet.dfp', 
                            '12pca.dfp', 
                            'influences_12pc.dfp',
                            'scree_plot.svg',
                            'PC1vPC2.svg', 
                            'pca{}.dfp'.format(max_compo),
                            '{}PCxy_plot.svg'.format(max_compo), 
                            'time_evolution_{}PC.svg'.format(max_compo),
                            'ward_hierarchy_acceleration.svg',
                            'optimal_clustering.svg',
                            'simulations_cluster_evolution.svg',
                            'clustered_contact_networks.dfp',
                            'GCA.p']

        output_list = iter(output_list)
        self.compute_cmat(to_df=next(output_list), **kwargs)
        self.compute_pca(next(output_list), n_components=12)
        self.get_features_pca(next(output_list))
        self.scree_plot(next(output_list))
        self.plot2D_landscapes(next(output_list), combi=[(0, 1)])
        self.compute_pca(next(output_list), n_components=max_compo)
        self.plot2D_landscapes(next(output_list))
        self.plot_pc_vs_time(next(output_list))
        self.get_n_optimal_clusters(next(output_list))
        self.plot_optimal_clusters(next(output_list))
        self.plot_cluster_time_evolution(next(output_list))
        self.get_clusters_networks(next(output_list))
        self.to_pickle(next(output_list))


    def compute_cmat(self, trajs, topos, to_df=None, 
                    sele='protein and not name H*', sele2=None, 
                    remove_intra=True, remove_duplicates=True, labels=None):

        """Computes the contact matrix from one or more trajectories and 
        topologies.
        
        Parameters
        ----------
        trajs: list or str or None
            Path or list of paths of input trajectories. If None, topos must
            be one of PDB or GRO.

        topos: list or str
            Path or list of paths of input topologies.
        
        to_df: str or None
            Save an average for each simulation to a pickle DataFrame.
            Allows to draw Perturbation Networks with an external function

        sele: str, default = 'protein and not name H*' i.e heavy-atom contacts
            Atoms selected for contact analysis. Use MDAnalysis atom selection
            reference (close to CHARMM’s atom selection syntax). See notes for
            atom selection

        sele2: str or None, default = None
            Atoms selected in the case of an "asymetric" contact analysis
            between two selections. Please note this can produce duplicates
            and self-contacts

        remove self: bool, default = True
            Decides if intraresidual atom contacts are discarded

        remove duplicates: bool, default = True
            Decides if duplicates residual contacts are discarded. Please note
            those duplicates only happen in the case of an asymetric selection
            which overlaps. 


        Notes
        ----------
        See https://docs.mdanalysis.org/stable/documentation_pages/selections.html
        and https://www.charmm.org/charmm/documentation/by-version/c45b1/select.html
        for selection reference

               """
        self.trajs = trajs
        self.topos = topos
        self.sele = sele
        self.sele2 = sele2
        self.remove_duplicates = remove_duplicates
        self.contacts1 = list()
        self.contacts2 = list()
        self.t = 0
        self.times = list()
        self.counts = list()

        #Creates an iterator over labels to get the trajectory labels
        if type(labels) != type(None):
            if type(labels) != list:
                labels = [labels]
            assert len(labels) == len(trajs), print("""Length of labels and
            trajectories differ""")
            self.traj_labels = list()
            self.iter_traj_labels = iter(labels)
        else:
            self.traj_labels = None

        #If a single topology is in the input, we assume the same topology is 
        # used for all trajectories. Also works for single traj/topology.
        if type(topos) == str:
            u_iter = iter([mda.Universe(topos, trajs)])
        else:
            u_iter = iter(mda.Universe(topo, traj) 
                    for topo, traj in zip(topos, trajs))
        for u in u_iter:
            self._compute_contacts(u)
        
        #Transforms information in numpy 1-D array for faster computations
#        dtypes = np.ushort, np.ushort, np.uintc, np.ubyte
#        print([(arr[0].dtype if type(arr[0]) != list else 0) for arr in [self.contacts1, self.contacts2, self.times, self.counts]])
        self.contacts1 = np.concatenate(self.contacts1)
        self.contacts2 = np.concatenate(self.contacts2)
        self.times = np.concatenate(self.times)
        self.counts = np.concatenate(self.counts)

        #Removes intraresidual contacts
        if remove_intra:
            to_keep = np.where(self.contacts1 != self.contacts2)[0]
            self.contacts1 = self.contacts1[to_keep]
            self.contacts2 = self.contacts2[to_keep]
            self.times = self.times[to_keep]
            self.counts = self.counts[to_keep]

        #Gets the list of unique contacts
        contacts = np.stack([self.contacts1, self.contacts2], axis=-1)
        unique_contacts, inv = np.unique(contacts, axis=0, 
                                        return_inverse=True)
        #Saving unique_contacts to re-build contacts name
        self.id2contact = unique_contacts
        #Build contact matrix with coo matrix and transforms then to dense
        contact_matrix = coo_matrix((self.counts, (inv, self.times)), 
                            shape=(unique_contacts.shape[0], self.t),
                            dtype=np.uint8)
        self.contact_matrix = np.array(contact_matrix.todense())
        if type(self.traj_labels) == list:
            self.traj_labels = np.array(self.traj_labels)
            if type(to_df) == str:
                df = pd.DataFrame({'node1': self.id2contact[:, 0],
                                    'node2': self.id2contact[:, 1]})
                for label in pd.unique(self.traj_labels):
                    ix = np.where(self.traj_labels == label)[0]
                    df[label] = np.average(self.contact_matrix[:, ix], axis=1)
                df.to_pickle(to_df)


    def normalize(self):
        pass #to implement

    def compute_pca(self, output=None, label_name="Simulation", **kwargs):
        """Computes PCA on the contact matrix

        Parameters
        ----------
        output: str
        Path where to save the PCA DataFrame pickled object

        label_name: str
        Name of the label column in the DataFrame PCA

           **kwargs: arguments to pass to the sklearn PCA class"""
        #PCA initialization
        self.pca = PCA(**kwargs)
        assert hasattr(self, 'contact_matrix'), print(
            "Contact matrix not computed")
        #Computes PCA
        self.pca_mat = self.pca.fit_transform(self.contact_matrix.T)
        self.pca_df = pd.DataFrame({'PC{}'.format(i+1): self.pca_mat[:, i]
                                for i in range(self.pca.n_components)})
        if type(self.traj_labels) in [list, np.ndarray]:
            self.pca_df[label_name] = self.traj_labels
            self.traj_label_name = label_name
        if type(output) != type(None):
            self.pca_df.to_pickle(output)

    def get_features_pca(self, output_df):
        features = pd.DataFrame({'node1': self.id2contact[:, 0],
                                    'node2': self.id2contact[:, 1]})
        for i in range(self.pca.n_components):
            features['PC{}'.format(i+1)] = self.pca.components_[i]
        features.to_pickle(output_df)
    
    def scree_plot(self, output, **kwargs):
        """Draws the scree plot (explained variance vs number of components)
        of the PCA

        Parameters
        ----------
        output: str
        Path where to save the scree plot

        **kwargs: arguments to pass to the sklearn PCA class if PCA not 
        already computed"""

        if not hasattr(self, 'pca'):
            self.compute_pca(**kwargs)
            
        fig, ax = plt.subplots(1, 1)
        ax.plot(range(1, int(self.pca.n_components)+1), 
                self.pca.explained_variance_, color='k', marker='+')
        ax.set_xlabel('Component Number')
        ax.set_ylabel('Eigenvalue')
        ax.set_ylim(0)
        ax.set_xlim(1, self.pca.n_components+1)
        ax.set_xticks(range(1, self.pca.n_components+1))
        plt.tight_layout()
        plt.savefig(output)
        plt.close(fig)
    
    def _individual_plot2D(self, ij, ax, color_palette="bright", 
                            show_start_finish=True, show_trajectory=False):
        if type(ax) == type(None):
            ax = plt.gca()

        x = "PC{}".format(ij[0]+1)
        y = "PC{}".format(ij[1]+1)
        if hasattr(self, "traj_label_name"):
            if not hasattr(self, "unique_traj_labels"):
                self._generate_colors_sim()

            g = sns.kdeplot(data=self.pca_df, x=x, y=y, common_norm=False, 
            ax=ax, hue=self.traj_label_name, palette=color_palette,
            legend=False)

            if show_start_finish:
                for label, color in self.label_color_iter:
                    ix = np.where(self.traj_labels == label)[0]
                    #Show start 
                    ax.scatter(self.pca_mat[ix[0], ij[0]], 
                    self.pca_mat[ix[0], ij[1]], color=color, marker='>', 
                    zorder=100, edgecolor='k')
                    #Show finish
                    ax.scatter(self.pca_mat[ix[-1], ij[0]], 
                    self.pca_mat[ix[-1], ij[1]], color=color, marker='8', 
                    zorder=100, edgecolor='k')


            if show_trajectory:
                for label in self.unique_traj_labels:
                    ix = np.where(self.traj_labels == label)[0]
                ax.plot(self.pca_mat[ix, ij[0]], self.pca_mat[ix, ij[1]], 
                color='k', linewidth=0.5)
        else:
            g = sns.kdeplot(data=self.pca_df, x=x, y=y, ax=ax, legend=False)
            if show_start_finish:
                #Show start 
                ax.scatter(self.pca_mat[0, ij[0]], self.pca_mat[0, ij[1]], 
                color='k', marker='>', zorder=100, edgecolor='k')
                #Show finish
                ax.scatter(self.pca_mat[-1, ij[0]], self.pca_mat[-1, ij[1]], 
                color='k', marker='8', zorder=100, edgecolor='k')
            if show_trajectory:
                ax.plot(self.pca_mat[:, ij[0]], self.pca_mat[:, ij[1]], 
                color='k', linewidth=0.5)   
                
        ax.grid(linestyle='--')
        ax.set_aspect('equal')


    def plot2D_landscapes(self, output, show_start_finish=True, 
                         show_trajectory=False, color_palette="bright",
                         combi=None, **kwargs):
        
        """Draws all possible 2D landscape (PCx vs PCy) or a custom list
        

        Parameters
        ----------
        output: str
        Path where to save the 2D landscape

        show_start_finish: bool, default = True
        Plots a '>' flag at the beginning of each simulation and a '8' flag
        at the finish

        show_trajectory: bool, default = False
        Plots each trajectory, very cumbersome. Use only in very simple systems


        color_palette: str, default="bright"
        Seaborn color palette to use for the labels of each simulations. 
        See https://seaborn.pydata.org/tutorial/color_palettes.html

        combi: list of integers tuples or None, default = None
        Draws only the combinations passed. e.g. [(0, 1)] draws only
        PC1 vs PC2. If None, all combinations are drawn

        **kwargs: arguments to pass to the sklearn PCA class if PCA not 
        already computed"""

        m_list = []
        if not hasattr(self, 'pca'):
            self.compute_pca(**kwargs)
        if self.traj_labels != None:
            self._generate_colors_sim()
        
        if type(combi) == type(None):
            combi = list(combinations(range(0, self.pca.n_components), 2))
        fig, axes = self._get_nice_axes(len(combi))
        # if len(combi) <= n_horizontal:
        #     n_horizontal = len(combi)
        #     figsize = [12, 12]
        # else:
        #     figsize = [12, n_horizontal*len(combi)//n_horizontal]

        # fig, axes = plt.subplots(len(combi)//n_horizontal, n_horizontal, 
        #                         sharex=True, sharey=True, figsize=figsize)
#        axes = np.array(axes)
        for ij, ax in zip(combi, axes.flatten()):
            #Building our own legend because the one from seaborn is a 
            #pain to handle   
            if len(m_list) == 0 and hasattr(self, "label_color_iter"):
                m_list = [mlines.Line2D([], [], color=c, label=lab) 
                        for (lab, c) in self.label_color_iter] 
            self._individual_plot2D(ij, ax, show_trajectory=show_trajectory, 
                            show_start_finish=show_start_finish,
                            color_palette=color_palette)

            ax.grid(linestyle='--')
            ax.set_aspect('equal')
        #Drawing at once simulation-dependant legend
        if len(m_list) >= 1:
            fig.legend(handles=m_list, loc='lower right', 
            title=self.traj_label_name, ncol=2)
        
        #Building our own legend for start finish and draw it
        if show_start_finish:
            m1 = mlines.Line2D([], [], color='w', marker='>', 
            markeredgewidth=1.0, markeredgecolor='k', markersize=10, 
            label='Start of simulation')
            m2 = mlines.Line2D([], [], color='w', marker='8',  
            markeredgewidth=1.0, markeredgecolor='k', markersize=10, 
            label='End of simulation')

            fig.legend(handles=[m1, m2], loc='lower left', numpoints=1)
        plt.tight_layout()
        plt.savefig(output)
        plt.close(fig)

    def plot_pc_vs_time(self, output, frames_per_ns=10, **kwargs):

        """Draws all possible PCx vs time  

        Parameters
        ----------
        output: str
        Path where to save the time-plot

        color_palette: str, default="bright"
        Seaborn color palette to use for the labels of each simulations. 
        See https://seaborn.pydata.org/tutorial/color_palettes.html

        frames_per_ns: int
        Number of frames per ns

        **kwargs: arguments to pass to the sklearn PCA class if PCA not 
        already computed"""

        if not hasattr(self, 'pca'):
            self.compute_pca(**kwargs)

        fig, axes = self._get_nice_axes(self.pca.n_components)

        ticks, ticklabels = self._generate_timed_xticklabels()

        for ax, i in zip(axes.flatten(), range(self.pca_mat.shape[1])):
            ax.set_title('PC{}'.format(i+1))
            if hasattr(self, "traj_label_name"):
                if not hasattr(self, "label_color_iter"):
                    self._generate_colors_sim()
                for label, color in self.label_color_iter:
                    ix = np.where(self.traj_labels == label)[0]
                    ax.plot(self.pca_mat[ix, i], color=color, 
                            label=label if i==0 else None, alpha=0.8)
                    ax.set_xlim(0, self.t/frames_per_ns)
                    ax.set_ylabel('PC{}'.format(i+1))
                    ax.set_xlabel('Time in ns')
                    ax.set_ticks(ticks)
                    ax.set_xticklabels(ticklabels)
            else:
                ax.plot(self.pca_mat[:, i], color='k')
        # if len(axes.flatten() != self.pca_mat.shape[1]):
        #     for ax in axes.flatten()[i+1:]:
        #         ax.axis('off')

        if hasattr(self, "traj_label_name"):
            fig.legend(ncol=2, loc='lower center', fontsize=10)
        plt.tight_layout()
        plt.savefig(output)
        plt.close(fig)

    def get_n_optimal_clusters(self, output, pc=(0, 1), **kwargs):
        """Draws all possible PCx vs time
        

        Parameters
        ----------
        output: str
        Path where to save the time-plot

        pc: tuple of int, default = (0, 1)
        Principal components where to cluster

        color_palette: str, default="bright"
        Seaborn color palette to use for the labels of each simulations. 
        See https://seaborn.pydata.org/tutorial/color_palettes.html

        frames_per_ns: int
        Number of frames per ns

        **kwargs: arguments to pass to the sklearn PCA class if PCA not 
        already computed"""

        if not hasattr(self, 'pca'):
            self.compute_pca(**kwargs)

        mpl.rcParams.update({'font.size': 16})
        fig, axes = plt.subplots(1, 2, figsize=[12, 6])
        l = ward(self.pca_mat[:,pc[0]:pc[1]+1:pc[1]-pc[0]])
        dendrogram(l, no_labels=True, count_sort='descendent', ax=axes[0])
        # generate the linkage matrix
        Z = linkage(self.pca_mat[:,pc[0]:pc[1]+1:pc[1]-pc[0]], 'ward')
        last = Z[-10:, 2]
        idxs = np.arange(1, len(last) + 1)
        lns1 = axes[1].plot(idxs, last[::-1], label='height', marker='+')

        acceleration = np.diff(last, 2)  # 2nd derivative of the distances
        acceleration_rev = acceleration[::-1]
        ax2 = axes[1].twinx()
        lns2 = ax2.plot(idxs[:-2] + 1, acceleration_rev, label='acceleration',
                         marker='+', color='r')
        self.optimal_clusters = acceleration_rev.argmax() + 2

        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        axes[1].legend(lns, labs, loc='best')

        axes[1].set_xlabel('Number of clusters')
        axes[0].set_xlabel('Clusters')
        axes[0].set_ylabel('Height')
        axes[1].set_ylabel('Height')

        ax2.set_ylabel('Height acceleration')
        plt.tight_layout()
        plt.savefig(output)
        plt.close(fig)
        mpl.rcParams.update({'font.size': 12})
    
    def plot_optimal_clusters(self, output, color_clusters=None, pc=(0, 1), **kwargs):
        """Draws optimal clusters in a 2D projection
        

        Parameters
        ----------
        output: str
        Path where to save the plot of optimal clusters

        color_clusters: str or None, default = None
        Iterator over colors to plot clusters. If None, automatically assign
        some colors. It's difficult but possible to match with colors of 
        get_n_optimal_clusters

        pc: tuple of int, default = (0, 1)
        Principal components where to cluster


        **kwargs: arguments to pass to the _individual_plot2D method"""

        if not hasattr(self, "optimal_clusters"):
            self.get_n_optimal_clusters(None)

        if color_clusters == None:
            color_clusters = sns.color_palette("bright", 
                                             self.optimal_clusters)
        self.color_clusters = color_clusters

        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, 
                                figsize=[12, 6])

        self._individual_plot2D(pc, axes[0], **kwargs)

        cluster = AgglomerativeClustering(n_clusters=self.optimal_clusters)
        self.cluster_labels = cluster.fit_predict(
            self.pca_mat[:,pc[0]:pc[1]+1:pc[1]-pc[0]])

        m = []
        i = 0
        for l, c in zip(pd.unique(self.cluster_labels), color_clusters):
            ix = np.where(self.cluster_labels == l)[0]
            sns.scatterplot(x=self.pca_mat[ix, pc[0]], y=self.pca_mat[ix, pc[1]], 
                            color=c, marker='+', linewidth=1, ax=axes[1])
            m.append(mlines.Line2D([], [], color=c, marker='+',
                            markersize=10, label='Cluster {}'.format(i+1)))
            i+=1
        axes[1].set_aspect('equal')
        axes[1].set_xlabel('PC1')
        axes[1].set_ylabel('PC2')
        axes[1].grid('--')
        axes[1].legend(handles=m, loc='best', numpoints = 1, ncol=2, 
                    fontsize=10)
        plt.tight_layout()
        plt.savefig(output)
        plt.close(fig)

    def plot_cluster_time_evolution(self, output, frames_per_ns=10,
                                    ticks_step=1000, aspect=500):
        
        """Draws the time evolution of the different simulations in each 
        cluster
        

        Parameters
        ----------
        output: str
        Path where to save the plot of optimal clusters

        pc: tuple of int, default = (0, 1)
        Principal components where to cluster

        **kwargs: arguments to pass to the _individual_plot2D method"""

        #Determine optimal dimensions of the plot
        if type(self.topos) == list:
            traj_len = [len(mda.Universe(top, traj).trajectory) 
                        for top, traj in zip(self.topos, self.trajs)]
            max_x = max(traj_len)
        else:
            max_x = len(mda.Universe(self.topos, self.trajs).trajectory)
        ticks_step = max_x // 10
        aspect = max_x // 20


        if self.traj_labels != None:
            if not hasattr(self, "unique_traj_labels"):
                self._generate_colors_sim()
            n_graphs = len(self.unique_traj_labels)
        else:
            n_graphs = 1

        if not hasattr(self, "cluster_labels"):
            self.plot_optimal_clusters(output=None)
        fig, axes = plt.subplots(n_graphs, 1, sharex=True, 
                                figsize=[12, n_graphs])
        ix = np.argsort(pd.unique(self.cluster_labels))
        cmap = mpl.colors.ListedColormap(np.array(self.color_clusters)[ix])
        bounds=list(range(len(self.color_clusters)+1))
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        axes = np.array(axes)
        
        for i, ax in enumerate(axes.flatten()):
            if hasattr(self, "unique_traj_labels"):
                traj_lab = self.unique_traj_labels[i]
                ax.set_title(traj_lab)
                ix = np.where(self.traj_labels==traj_lab)[0]
            else:
                ix = slice(-1)

            ax.imshow([self.cluster_labels[ix]], cmap=cmap, aspect=aspect, 
                        norm=norm, interpolation='none')
        ticks, ticklabels = self._generate_timed_xticklabels()
            
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels)

        ax.set_xlabel('Time in ns')
        #plt.tight_layout()
        [ax.set_yticks([]) for ax in axes.flatten()]
        plt.tight_layout()
        plt.savefig(output)
        plt.close(fig)

    def get_clusters_networks(self, output):
        assert hasattr(self, "cluster_labels"), print("""Optimal clustering 
        hasn't been done.""")
        features = pd.DataFrame({'node1': self.id2contact[:, 0],
                                 'node2': self.id2contact[:, 1]})
        for traj_lab in pd.unique(self.cluster_labels):
            ix = np.where(self.cluster_labels==traj_lab)[0]
            features['cluster_{}'.format(traj_lab)] = np.mean(
                self.contact_matrix[:,ix], axis=1)
        features.to_pickle(output)

    def to_pickle(self, output):
        pkl.dump(self, open(output, 'wb'))
        
    
    def _compute_contacts(self, u):
        """Computes the contacts of a MDAnalysis universe and append the 
        result to lists storing the residues in contact, the number
        of contacts and the time of contacts (trajectory dependent) 

        Parameters
        ----------
        u: MDA Universe object
        """

        #Create dictionnary with atom to residue correspondance
        self.atom2res = {atom.index: atom.residue.resindex for atom in u.atoms}
        #Gets first selection and its indexing in the protein
        self.s1 = u.select_atoms(self.sele)
        self.ix1 = np.array(self.s1.atoms.indices)
        #If a second selection: gets it and indexes it
        if self.sele2 != None:
            self.s2 = u.select_atoms(self.sele2)
            self.ix2 = np.array(self.s2.atoms.indices)

        #Iterates over frames to compute contacts
        for _t, ts in enumerate(tqdm(u.trajectory[self.slice])):
            if type(self.sele2) == type(None):
                pairs = self._contacts_sym()
            else:
                pairs = self._contacts_asym()
            U, inv = np.unique(pairs, return_inverse=True)
            #Translates atomic contact information in residue contact info
            pairs_res = np.array(
                [self.atom2res[x] for x in U])[inv].reshape(pairs.shape)
            unique_pairs, counts = np.unique(pairs_res, axis=1, 
                                            return_counts=True)
            #Stores information in four different lists
            self.contacts1.append(unique_pairs[0].astype(np.ushort)) #res1 
            self.contacts2.append(unique_pairs[1].astype(np.ushort)) #res2 
            self.counts.append(counts.astype(np.ushort)) #number of contact
            self.times.append(np.array(
                [self.t+_t]*unique_pairs.shape[1]).astype(np.uintc)) #timestep
        if type(self.traj_labels) == list:
            self.traj_labels += [next(self.iter_traj_labels)]*(_t+1)
        self.t += (_t+1) #shifts the timestep for a series of trajectories


    def _generate_colors_sim(self, color_palette="bright"):
        """ Generates colors, labels and corresponding zipped iterator
        to be used for different methods"""
        self.unique_traj_labels = pd.unique(self.traj_labels)
        self.n_traj_labels = len(pd.unique(self.traj_labels))
        self.color_trajs = sns.color_palette(color_palette, 
                                                self.n_traj_labels)
        self.label_color_iter = list(zip(self.unique_traj_labels, 
                                        self.color_trajs))


    def _contacts_sym(self):
        """Uses built-in MDAnalysis function to get interacting pairs in 
        symmetric selection"""
        pairs = self_capped_distance(self.s1.positions, self.cutoff, 
                                    return_distances=False)
        return np.sort(self.ix1[pairs], axis=1).T

    def _contacts_asym(self):
        """Uses built-in MDAnalysis function to get interacting pairs in
        asymmetric selection"""
        pairs = capped_distance(self.s1.positions, self.cutoff,
                                    return_distances=False)
        pairs = np.stack([self.ix1[pairs[:,0]], self.ix2[pairs[:,1]]])
        if self.remove_duplicates:
            pairs = np.sort(pairs, axis=1)
            pairs = np.unique(pairs, axis=1)
        return pairs
    
    def _get_nice_axes(self, n_graphs, n_h=3):
        """Generate nice axes subplots for different cases"""
        if n_graphs == 1:
            fig, axes = plt.subplots(1, 1, figsize=[12, 6])
            axes = np.array(axes)
        if n_graphs == 2:
            fig, axes = plt.subplots(1, 2, figsize=[12, 6])
        if n_graphs == 3:
            fig, axes = plt.subplots(1, 3, figsize=[12, 6])
        if n_graphs == 4:
            fig, axes = plt.subplots(2, 2, figsize=[12, 12])
        elif n_graphs > 4:
            n_vert = (n_graphs//n_h)
            if n_graphs % n_h !=0: 
                n_vert += 1 
            fig, axes = plt.subplots(n_vert, 3, figsize=[12, n_vert*3])
            if n_graphs % n_h !=0:
                for ax in axes.flatten()[-n_graphs % n_h:]:
                    ax.axis('off')
        return fig, axes

    def _generate_timed_xticklabels(self):
        """Generate xticks"""

        if type(self.topos) == list:
            traj_len = [len(mda.Universe(top, traj).trajectory) 
                        for top, traj in zip(self.topos, self.trajs)]
            big_univ, max_x = min([(j, i) for i, j in enumerate(traj_len)])
            u = mda.Universe(self.topos[big_univ], self.trajs[big_univ])
       
        else:
            max_x = len(mda.Universe(self.topos, self.trajs).trajectory)
            u = mda.Universe(self.topos, self.trajs)

        u.trajectory[0]
        t0 = u.trajectory.time
        u.trajectory[-1]
        tf = u.trajectory.time
        step_app = (tf-t0)/10
        step = '{}{}'.format(str(step_app)[0], 
                    ''.join(['0']*(len(str(step_app).split('.')[0])-1)))
        if len(step) == 1:
            step = 10
        else:
            step = int(step)

        fs = step*(len(u.trajectory))/(tf-t0)
        ticks = list(np.arange(0, len(u.trajectory), fs))+[len(u.trajectory)]
        ticks = np.array(ticks).astype(np.int32)
        ticklabels = list(np.arange(t0, tf, step))+[tf]
        ticklabels = np.array(ticklabels).astype(np.int32)
        return ticks, ticklabels        


            


        
