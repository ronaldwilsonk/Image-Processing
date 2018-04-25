import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt


class DataAnalysis:
    def __init__(self, input_vectors, label_vector, n_cross_validation):
        """
        Program to analyze data and split it into n-fold cross validation dataset.
        The splitting of dataset is done conserving the distance distribution of data for each cluster to avoid bias
        incurred in randomly choosing data for training. Algorithm modelled after Histogram Equalization.
        Mahalanobis distance is used to ensure uniform scaling in each dimension.
        Author: Ronald Wilson

        :param input_vectors: n-dimensional vector, array like m-samples x n-dimensions
        :param label_vector: Assumes format [-1,-1,....,1,....-1]
                             -1:False class     1:True Class
                             array like m-samples x k-classes
        :param n_cross_validation: Number of divisions in dataset. Splits data into 'n + 1' sections.
                                   'n' for training, '1' for testing
        """
        self.input = input_vectors
        self.label = label_vector
        self.n_valid = n_cross_validation

        # Generates the covariance matrix for the input data
        _, self.covariance_matrix = self._generate_covariance_matrix()

        # Labels are condensed into a single number. The basis of projection is linear with unit step
        # i.e. [1 , -1, -1] => matrix_product([1 , -1, -1], [1, 2, 3]')
        self.label_dict = dict()
        # Modifies labels using dictionary generated above
        self._process_labels()

        # Dictionary used to store distance from assigned cluster centers
        self.cluster_distance_distribution_dict = dict()

        # Calculate unique labels
        cluster_num = np.unique(self.label)
        # Dictionary to store processed dataset
        self.cross_validation_dataset = self._generate_cluster_dict(self.n_valid - 1, 1)
        self.cross_validation_labels = self._generate_cluster_dict(self.n_valid - 1, 1)

        # Go through data and bin clusters based on the binsize
        for cluster in cluster_num:
            self._bin_cluster_members(cluster, bin_size=100)

    def _process_labels(self):
        """
        Labels are condensed into a single number. The basis of projection is linear with unit step
        i.e. [1 , -1, -1] => matrix_product([1 , -1, -1], [1, 2, 3]')
        A dictionary mapping original label to condensed form is also saved.
        :return: None
        """
        rows, cols = np.shape(self.label)
        processed_label_vec = np.zeros(rows)
        pos_tag = np.arange(1, cols + 1, 1)
        for i in range(0, rows):
            val = sum(np.multiply(pos_tag, self.label[i, :]))
            processed_label_vec[i] = val
            self.label_dict[val] = self.label[i, :]
        self.label = processed_label_vec

    def _generate_covariance_matrix(self):
        """
        Generates the covariance matrix for the input data
        :return: Mean Vector, Covariance Matrix
                 Mean Vector stores the centroid for each cluster label
        """
        mean_vector = np.mean(self.input, axis=0)
        zero_mean_input = np.subtract(self.input, mean_vector)
        covariance_matrix = np.matmul(np.transpose(zero_mean_input), zero_mean_input)
        return mean_vector, covariance_matrix

    def _get_mahalanobis_distance(self, vec1, vec2):
        """
        Rerturn the Mahalanobis distance between two vectors
        :param vec1: Vector 1
        :param vec2: Vector 2
        :return: Mahalanobis Distance (Measured in standard deviations. No S.I units of measurement)
        """
        unscaled_dist = np.subtract(vec1, vec2)
        buf = np.matmul(unscaled_dist, self.covariance_matrix)
        scaled_dist = np.matmul(buf, np.transpose(unscaled_dist))
        return scaled_dist

    def _generate_cluster_dict(self, max_limit, bin_size):
        """
        Generates a empty template dictionary
        Template dict[integer(range(0, max_limit, step=bin_size))] = []
        :param max_limit: Upper limit of keys in the dictionary
        :param bin_size: Separation between each key
        :return: Empty template dictionary
        """
        max_limit = round(max_limit/bin_size) + 1
        keys = np.arange(0, max_limit, 1)
        clust_dict = dict()
        for key in keys:
            clust_dict[key] = []
        return clust_dict

    def _bin_cluster_members(self, cluster_label, bin_size):
        """
        Uses the Histogram equalization algorithm with set bin_size to bin data
        Updates the cross_validation_data_dictionary with the input samples from the binned data
        :param cluster_label: Label of current cluster
        :param bin_size: Histogram bin size
        :return: None
        """
        labels, count = np.unique(self.label, return_counts=True)
        rows, cols = np.shape(self.input)
        cluster_membership_count = count[list(labels).index(cluster_label)]
        cluster_members = np.zeros((cluster_membership_count, cols))
        dist_vec, pos = [], 0

        for i in range(0, rows):
            if self.label[i] == cluster_label:
                cluster_members[pos, :] = np.matmul(self.input[i], self.covariance_matrix)
                pos += 1
        cluster_centroid = np.mean(cluster_members, axis=0)

        for i in range(0, len(cluster_members)):
            dist_vec.append(np.linalg.norm(np.subtract(cluster_centroid, cluster_members[i])))

        cluster_dict = self._generate_cluster_dict(max(dist_vec), bin_size)
        for i in range(0, len(dist_vec)):
            buf = cluster_dict[round(dist_vec[i] / bin_size)]
            buf.append(cluster_members[i])
            cluster_dict[round(dist_vec[i] / bin_size)] = buf
        # Split the binned data for cross validation and populate it into the dataset dictionary
        self._populate_cross_validation_dict(cluster_dict, cluster_label)

    def _populate_cross_validation_dict(self, cluster_dict, cluster_label):
        """
        Updates the cross-validation dictionary with the binned data
        :param cluster_dict: Binned data dictionary for the current cluster
        :param cluster_label: Name of the current cluster
        :return: None
        """
        membership_count = []
        for key in list(cluster_dict.keys()):
            membership_count.append(len(cluster_dict[key]))
        self.cluster_distance_distribution_dict[cluster_label] = membership_count

        training_count = np.floor(np.divide(membership_count, self.n_valid))
        for item in list(self.cross_validation_labels.keys()):
            buf = self.cross_validation_labels[item]
            if item != self.n_valid - 1:
                for i in range(0, np.int(sum(training_count))):
                    buf.append(self.label_dict[cluster_label])
                self.cross_validation_labels[item] = buf
            else:
                temp = np.int(sum(membership_count) - (self.n_valid - 1) * sum(training_count))
                for i in range(0, temp):
                    buf.append(self.label_dict[cluster_label])
                self.cross_validation_labels[item] = buf

        for idx, key in enumerate(list(cluster_dict.keys())):
            count = len(cluster_dict[key])
            base = np.arange(0, count, 1, dtype=np.int)
            np.random.shuffle(base)
            val = cluster_dict[key]
            pos = 0

            for key2 in range(0, len(self.cross_validation_dataset.keys())):
                buf = self.cross_validation_dataset[key2]
                if key2 != self.n_valid - 1:
                    for i in range(pos, np.int(pos + training_count[idx])):
                        buf.append(val[base[i]])
                    self.cross_validation_dataset[key2] = buf
                    pos += np.int(training_count[idx])
                else:
                    for i in range(pos, count):
                        buf.append(val[base[i]])
                    self.cross_validation_dataset[key2] = buf

    def get_dataset(self):
        """
        Fetch the processed dataset and labels
        :return: input_data_dict, label_data_dict
                 input_data_dict has keys: 0:n_cross_validation + 1
                 0:n_cross_validation for training and the next key for testing
                 label_data_dict has the same format and dimensions
        """
        return self.cross_validation_dataset, self.cross_validation_labels

    def plot_cluster_distance_plot(self):
        """
        Plots the spread of datapoints of all clusters
        :return: None
        """
        for item in list(self.cluster_distance_distribution_dict.keys()):
            plt.plot(self.cluster_distance_distribution_dict[item], label=str(self.label_dict[item]))
            plt.hold(True)
        plt.legend()
        plt.ylabel("Memebership count")
        plt.xlabel("Distance(scale: 10e+02 units)")
        plt.show()


if __name__ == "__main__":

    input_dict = loadmat("D:/Project 2/Proj2FeatVecsSet1.mat")
    input_vec = input_dict['Proj2FeatVecsSet1']

    label_dict = loadmat("D:/Project 2/Proj2TargetOutputsSet1.mat")
    label_vec = label_dict['Proj2TargetOutputsSet1']

    DP = DataAnalysis(input_vec, label_vec, n_cross_validation=5)

    # Get processed data
    data, labels = DP.get_dataset()

    # Plot distance distribution
    # DP.plot_cluster_distance_plot()
