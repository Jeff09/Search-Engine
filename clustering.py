# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import csv, operator

import sys
sys.setrecursionlimit(3500)

def get_top_docs(filename):
    with open(filename) as f:
        fr = f.read()
    docs_ranks = {}
    for ranks in fr.split('\n'):
        if len(ranks.split('\t')) == 2:
            doc = int(ranks.split('\t')[0])
            rank = ranks.split('\t')[1]
            docs_ranks[doc] = float(rank)
    sorted_ranks = sorted(docs_ranks.items(), key =operator.itemgetter(1), reverse=True)[:10000]
    #print len(sorted_ranks)
    sorted_docs = dict(sorted_ranks)
    top_docs = sorted(sorted_docs.items(), key = operator.itemgetter(0))
    docs = dict(top_docs)
    #print top_docs
    
    return docs.keys()

def tfidf_matrix_cluster(matrix):
        from sklearn.feature_extraction.text import TfidfTransformer
        
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(matrix)
        
        return tfidf.toarray()


class clustering:
    def __init__(self, filepath):
        self.filepath = filepath
        self.tfidf_dict = []
        
    def get_vac(self):
        filepath = self.filepath
        with open(filepath) as f:
            docs = f.read()
        i = 1
        tfidf_dict = []
        for doc in docs.split('\n'):
            if i%2 == 0:
               tfidf_dict.append(doc)    
            i+=1    
        
        doc_tfidf_dict = []
        vac_freq = {}
        for doc in tfidf_dict:      
            term_tfidf_dict ={}
            for term_tfidf in doc.split():
                if len(term_tfidf.split(":")) == 2:                
                    term = term_tfidf.split(":")[0]
                    tfidf = term_tfidf.split(":")[1]
                    term_tfidf_dict[term] = float(tfidf)
                    vac_freq[term] = vac_freq.get(term, 0) + float(tfidf)
            doc_tfidf_dict.append(term_tfidf_dict)
        
        vac = [word for word, freq in vac_freq.iteritems() if freq >= 10]
        self.vac = vac
        print 'vac size:', len(vac)
        
        with open('vac.txt', 'wb') as f:
            for word in vac:
                f.write(str(word)+ ' ')
           
        return doc_tfidf_dict
    
        
    def get_tfidf_dict(self):        
        
        top_docs = get_top_docs('pageranke_score.txt')                
        print 'top_docs size:', len(top_docs)
        self.top_docs = top_docs
        with open('top_docs.txt', 'wb') as f:
            for doc in top_docs:
                f.write(str(doc)+ ' ')
        
        doc_tfidf_dict = self.get_vac()
        vac = self.vac
        
        import copy
        new_docs_dict = []
        j=-1
        for doc in doc_tfidf_dict:
            j += 1
            if j not in top_docs:
                continue
            #new_doc_matrix = {}
            doc_copy = copy.deepcopy(doc)
            for word, freq in  doc_copy.iteritems():
                if word not in vac:
                    del doc[word]
            new_docs_dict.append(doc)
            
                    #new_doc_matrix[word] = freq
            #new_docs_dict.append(new_doc_matrix)
        print len(new_docs_dict), len(new_docs_dict[0])        
        
        self.tfidf_dict = new_docs_dict
    
    def dict2matrix(self):
        #matrix = []
        if self.tfidf_dict == None:
            self.get_tfidf_dict()
        tfidf_dict = self.tfidf_dict
        vac = self.vac
        for doc in tfidf_dict:            
            data = [doc.get(word, 0) for word in vac]
            yield data
            #matrix.append(data)
        #yield np.array(matrix), vac
        
    def write_csv(self, filename):
        f = csv.writer(open(filename, 'wb'))
        vac = self.vac
        f.writerow(vac)
        for row in self.dict2matrix():
            f.writerow(row)
            
    def fancy_dendrogram(self, *args, **kwargs):
        max_d = kwargs.pop('max_d', None)
        if max_d and 'color_threshold' not in kwargs:
            kwargs['color_threshold'] = max_d
        annotate_above = kwargs.pop('annotate_above', 0)
    
        ddata = dendrogram(*args, **kwargs)
    
        if not kwargs.get('no_plot', False):
            plt.title('Hierarchical Clustering Dendrogram (truncated)')
            plt.xlabel('sample index or (cluster size)')
            plt.ylabel('distance')
            for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
                x = 0.5 * sum(i[1:3])
                y = d[1]
                if y > annotate_above:
                    plt.plot(x, y, 'o', c=c)
                    plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                                 textcoords='offset points',
                                 va='top', ha='center')
            if max_d:
                plt.axhline(y=max_d, c='k') 
        #return ddata
    
    def get_average_centers(self, clusters, tfidf_matrix):
        cluster2docs = {}
        i = 0
        clusters_list =tuple(clusters.tolist())
        for cluster in clusters_list:
            if cluster not in cluster2docs:
                cluster2docs[cluster] = [i]
                #cluster2docs[cluster] = tfidf_matrix[i, :]
                #average = dict(zip(cluster, tfidf_matrix[i,:]))
            else:
                cluster2docs[cluster].append(i)
            i+=1
        
        all_centroids = []        
        for clusterid, docsid in cluster2docs.iteritems():
            centroid = np.zeros(len(tfidf_matrix[0]))
            i = 0
            for each in docsid:                    
                centroid += tfidf_matrix[each]
                i += 1
            all_centroids.append(centroid/i)
        return np.asarray(all_centroids)
    
    def hierarchical_clustering(self, k=0, max_d = 0.7, method='single'):
        matrix = []
        for row in self.dict2matrix():
            matrix.append(row)
        
        tfidf_matrix = tfidf_matrix_cluster(np.array(matrix))
        
        if method == 'single':
            Z = linkage(tfidf_matrix, 'single')
            filename1 = 'doc2cluster_single.csv'
            filename2 = 'centroids_single.csv'
        elif method == 'complete':
            Z = linkage(tfidf_matrix, 'complete')
            filename1 = 'doc2cluster_complete.csv'
            filename2 = 'centroids_complete.csv'
        elif method == 'average':
            Z = linkage(tfidf_matrix, 'average')  
            filename1 = 'doc2cluster_average.csv'
            filename2 = 'centroids_average.csv'
        
        #set cut-off
        #max_d = 1  # max_d as in max_distance    
        self.fancy_dendrogram(
            Z,
            truncate_mode='lastp',
            p=40,
            leaf_rotation=90.,
            leaf_font_size=12.,
            show_contracted=True,
            annotate_above=10,
            max_d=max_d,  # plot a horizontal cut-off line
            )
        
        clusters = fcluster(Z, k, criterion='maxclust') #cluster id start from 1
        clusters = clusters - 1
        #clusters = fcluster(Z, max_d, criterion='distance')
        centriods = self.get_average_centers(clusters, matrix)    
        new_clusters = zip(self.top_docs, clusters)
        
        np.savetxt(filename1, new_clusters, delimiter=',', fmt='%f')
        np.savetxt(filename2, centriods, delimiter=',', fmt='%f')
        if method == 'single':
            self.centroids_single = centriods
            self.doc2cluster_single = clusters
        elif method == 'complete':
            self.centroids_complete = centriods
            self.doc2cluster_complete = clusters
        elif method == 'average':
            self.centroids_average = centriods
            self.doc2cluster_average = clusters
    
    def kmeans(self, k):   
        print 'build kmeans'
        from sklearn.cluster import KMeans    
        matrix = []
        for row in self.dict2matrix():
            matrix.append(row)
            
        tfidf_matrix = tfidf_matrix_cluster(np.array(matrix))
        
        kmeans = KMeans(n_clusters = k, random_state=1).fit(tfidf_matrix)
        centers = kmeans.cluster_centers_    #cluster id start form 0
        #centers_mapping = {case: cluster for case, cluster in enumerate(kmeans.labels_)}# docID : clusterID
        centers_mapping = zip(self.top_docs, kmeans.labels_)
        np.savetxt('doc2cluster_kmeans.csv',centers_mapping, delimiter=',', fmt='%f')
        np.savetxt('centroids_kmeans.csv',centers, delimiter=',', fmt='%f')
        
        self.centroids_kmeans = centers
        self.doc2clusters_kmeans = centers_mapping      
        
    def centroid_matrix2dict(self, centriods, method='single'):
        
        if method == 'single':
            filename = 'centroids_single.txt'
        elif method == 'complete':
            filename = 'centroids_complete.txt'
        elif method == 'average':
            filename = 'centroids_average.txt'
            
        f = open(filename, 'w')
        
        vac = self.vac
        centriods_dict = []
        for doc in centriods:
            centriod_dict = {}
            i = 0
            for freq in doc:
                if freq > 0:
                    centriod_dict[vac[i]] = freq
                i += 1
            centriods_dict.append(centriod_dict)
        
        f.write(centriods_dict)
        print len(centriods_dict), len(centriods_dict[0])        
 

filepath = 'Document Vector.txt'  
cc = clustering(filepath)
cc.get_tfidf_dict() 
cc.dict2matrix() 
cc.write_csv('term_doc_matrix.csv')
          
#matrix_array, vac = count_matrix(filepath)
#tfidf_matrix = get_tfidf_matrix(matrix_array)

#print 'hierarchical clustering using single linkage.'
#cc.hierarchical_clustering(k=100, method='single')

#print 'hierarchical clustering using complete linkage.'
#cc.hierarchical_clustering(k=100, method='complete')

#print 'hierarchical clustering using average linkage.'
#cc.hierarchical_clustering(k=100, method='average')

print 'KMeans clustering.'
cc.kmeans(100)
#cc.centroid_matrix2dict(cc.centroids_kmeans)
"""
query_file = 'sport swiming'
que_vec = get_query_vector(query_file, vac)

best_cluster_kmeans = cosinie_similarity(que_vec, centroids_kmeans)
best_cluster_average = cosinie_similarity(que_vec, centroids_average)
best_cluster_complete = cosinie_similarity(que_vec, centroids_complete)

kmeans_docs = return_docsID(best_cluster_kmeans, doc2clusters_kmeans, 'kmeans')
average_docs = return_docsID(best_cluster_average, doc2clusters_average - 1, 'average')
complete_docs = return_docsID(best_cluster_complete, doc2clusters_complete-1, 'complete')
print 'docID from kmeans:'
print kmeans_docs
print 'docID from average:'
print average_docs
print 'docID from complete:'
print complete_docs
"""
