import numpy as np
import math as mt
import sys
from sklearn.cluster import KMeans
from sklearn import metrics

class XMeans:
    def loglikelihood(self, r, rn, var, m, k):
        l1 = - rn / 2.0 * mt.log(2 * mt.pi)
        l2 = - rn * m / 2.0 * mt.log(var)
        l3 = - (rn - k) / 2.0
        l4 = rn * mt.log(rn)
        l5 = - rn * mt.log(r)

        return l1 + l2 + l3 + l4 + l5

    def __init__(self, X, kmax = 20):
        self.X = X
        self.num = np.size(self.X, axis=0)
        self.dim = np.size(X, axis=1)
        self.KMax = kmax

    def fit(self):
        k = 1
        X = self.X
        M = self.dim
        num = self.num

        while(1):
            ok = k

            #Improve Params
            kmeans = KMeans(n_clusters=k).fit(X)
            labels = kmeans.labels_
            m = kmeans.cluster_centers_

            #Improve Structure
            #Calculate BIC
            p = M + 1

            obic = np.zeros(k)

            for i in range(k):
                rn = np.size(np.where(labels == i))
                var = np.sum((X[labels == i] - m[i])**2)/float(rn - 1)
                obic[i] = self.loglikelihood(rn, rn, var, M, 1) - p/2.0*mt.log(rn)

            #Split each cluster into two subclusters and calculate BIC of each splitted cluster
            sk = 2 #The number of subclusters
            nbic = np.zeros(k)
            addk = 0

            for i in range(k):
                ci = X[labels == i]
                r = np.size(np.where(labels == i))

                kmeans = KMeans(n_clusters=sk).fit(ci)
                ci_labels = kmeans.labels_
                sm = kmeans.cluster_centers_

                for l in range(sk):
                    rn = np.size(np.where(ci_labels == l))
                    var = np.sum((ci[ci_labels == l] - sm[l])**2)/float(rn - sk)
                    nbic[i] += self.loglikelihood(r, rn, var, M, sk)

                p = sk * (M + 1)
                nbic[i] -= p/2.0*mt.log(r)

                if obic[i] < nbic[i]:
                    addk += 1

            k += addk

            if ok == k or k >= self.KMax:
                break


        #Calculate labels and centroids
        kmeans = KMeans(n_clusters=k).fit(X)
        self.labels_ = kmeans.labels_
        self.k = k
        self.cluster_centers_ = kmeans.cluster_centers_


class ElbowPointDiscrimant:
    def __init__(self, X, kmax = 10):
        self.X = X
        self.num = np.size(self.X, axis=0)
        self.dim = np.size(X, axis=1)
        self.KMax = kmax

    def fit(self):
        elbow_points = []
        centers = []
        kmeans = []
        k_min = 1
        min_distortion = -1
        for k in range(k_min, self.KMax):
            km = KMeans(n_clusters=k).fit(self.X)
            self.labels_ = km.labels_
            self.cluster_centers_ = km.cluster_centers_
            distortion = self.compute_mean_distortion()
            if min_distortion == -1:
                min_distortion = distortion
            if distortion > min_distortion:
                break
            else:
                min_distortion = distortion
            kmeans.append(km)
            elbow_points.append(distortion)
            centers.append(km.cluster_centers_)
        # MinMaxScaler
        max_distortion = max(elbow_points)
        min_distortion = min(elbow_points)
        for i in range(len(elbow_points)):
            elbow_points[i] = (elbow_points[i] - min_distortion) / (max_distortion / min_distortion)
        print('mean_distortion:', elbow_points)
        # If mean_distortion increases while increasing K above a given value, remove all the following values.
        # min_distortion = max(elbow_points)
        # for i in range(len(elbow_points)):
        #     if elbow_points[i] <= min_distortion:
        #         min_distortion = elbow_points[i]
        #     else:
        #         print(elbow_points[i], min_distortion)
        #         print('Removing some values...')
        #         elbow_points = elbow_points[:i+1]
        #         break
        # Compute angles
        alphas = []
        for i in range(len(elbow_points) - 2):
            j = i + 1
            k = i + 2
            p_i, p_j, p_k = (elbow_points[i], k_min+i), (elbow_points[j], k_min+j), (elbow_points[k], k_min+k)
            print('p_i:', p_i, 'p_j:', p_j, 'p_k', p_k)
            a = np.sqrt(np.square(p_i[0] - p_j[0]) + np.square(p_i[1] - p_j[1]))
            b = np.sqrt(np.square(p_j[0] - p_k[0]) + np.square(p_j[1] - p_k[1]))
            c = np.sqrt(np.square(p_k[0] - p_i[0]) + np.square(p_k[1] - p_i[1]))
            alpha = np.arccos((np.square(a) + np.square(b) - np.square(c)) / (2 * a * b))
            alphas.append((alpha, k_min+j))
            print('a:', a, 'b:', b, 'c:', c)
        best_k = min(alphas)[1]
        print('alphas: ', alphas)
        print('best k: ', best_k)
        self.labels_ = kmeans[best_k - k_min].labels_
        self.cluster_centers_ = kmeans[best_k - k_min].cluster_centers_


    def compute_mean_distortion(self):
        total_dispersion = 0
        # sum of squared euclidian distance
        sse = 0
        for k in range(len(self.cluster_centers_)):
            data = self.X[np.where(self.labels_ == k)]
            for x1 in data:
                for x2 in data:
                    se = np.linalg.norm(x1-x2)
                    sse += se
                    # total_dispersion += sum(np.abs(x1-x2))
        mean_distortion = sse / len(self.X)
        return mean_distortion