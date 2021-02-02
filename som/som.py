"""
Script to implement simple self organizing map using PyTorch, with methods
similar to clustering method in sklearn.

@author: Riley Smith
Created: 1-27-21
"""

import numpy as np

class SOM():
    """
    An implementation of a self organizing map using numpy.
    """
    def __init__(self, m=3, n=3, dim=3, lr=1, sigma=1, max_iter=3000, **kwargs):
        # Initialize descriptive features of SOM
        self.m = m
        self.n = n
        self.dim = dim
        self.shape = (m, n)
        self.initial_lr = lr
        self.lr = lr
        self.sigma = sigma
        self.max_iter = max_iter

        # Initialize weights
        self.weights = np.random.normal(size=(m * n, dim))
        self._locations = self._get_locations(m, n)

        # Set after fitting
        self._inertia = None
        self._n_iter_ = None
        self._trained = False

    def _get_locations(self, m, n):
        """
        Return the indices of an m by n array.
        """
        return np.argwhere(np.ones(shape=(m, n))).astype(np.int64)

    def _find_bmu(self, x):
        """
        Find the index of the best matching unit for the input vector x.
        """
        # Stack x to have one row per weight
        x_stack = np.stack([x]*(self.m*self.n), axis=0)
        # Calculate distance between x and each weight
        distance = np.linalg.norm(x_stack - self.weights, axis=1)
        # Find index of best matching unit
        return np.argmin(distance)

    def step(self, x):
        """
        Do one step of training on the given input vector.
        """
        # Stack x to have one row per weight
        x_stack = np.stack([x]*(self.m*self.n), axis=0)

        # Get index of best matching unit
        bmu_index = self._find_bmu(x)

        # Find location of best matching unit
        bmu_location = self._locations[bmu_index,:]

        # Find square distance from each weight to the BMU
        stacked_bmu = np.stack([bmu_location]*(self.m*self.n), axis=0)
        bmu_distance = np.sum(np.power(self._locations.astype(np.float64) - stacked_bmu.astype(np.float64), 2), axis=1)

        # Compute update neighborhood
        neighborhood = np.exp((bmu_distance / (self.sigma ** 2)) * -1)
        local_step = self.lr * neighborhood

        # Stack local step to be proper shape for update
        local_multiplier = np.stack([local_step]*(self.dim), axis=1)

        # Multiply by difference between input and weights
        delta = local_multiplier * (x_stack - self.weights)

        # Update weights
        self.weights += delta

    def _compute_point_intertia(self, x):
        """
        Compute the inertia of a single point. Inertia defined as squared distance
        from point to closest cluster center (BMU)
        """
        # Find BMU
        bmu_index = self._find_bmu(x)
        bmu = self.weights[bmu_index]
        # Compute sum of squared distance (just euclidean distance) from x to bmu
        return np.sum(np.square(x - bmu))

    def fit(self, data, epochs=1, shuffle=True):
        """
        Take data (a tensor of type float64) as input and fit the SOM to that
        data for the specified number of epochs.
        """
        # Count total number of iterations
        global_iter_counter = 0
        n_samples = data.shape[0]
        total_iterations = np.minimum(epochs * n_samples, self.max_iter)

        for epoch in range(epochs):
            # Break if past max number of iterations
            if global_iter_counter > self.max_iter:
                break

            if shuffle:
                indices = np.random.permutation(n_samples)
            else:
                indices = np.arange(n_samples)

            # Train
            for idx in indices:
                # Break if past max number of iterations
                if global_iter_counter > self.max_iter:
                    break
                input = data[idx]
                # Do one step of training
                self.step(input)
                # Update learning rate
                global_iter_counter += 1
                self.lr = (1 - (global_iter_counter / total_iterations)) * self.initial_lr

        # Compute inertia
        inertia = np.sum(np.array([float(self._compute_point_intertia(x)) for x in data]))
        self._inertia_ = inertia

        # Set n_iter_ attribute
        self._n_iter_ = global_iter_counter

        # Set trained flag
        self._trained = True

        return

    def predict(self, X):
        """
        For each line in x, predict the cluster center that is closest to it
        (return that cluster center's index).
        """
        # Check to make sure SOM has been fit
        if not self._trained:
            raise NotImplementedError('SOM object has no predict() method until after calling fit().')

        # Make sure X has proper shape
        assert len(X.shape) == 2, f'X should have two dimensions, not {len(X.shape)}'
        assert X.shape[1] == self.dim, f'This SOM has dimesnion {self.dim}. Received input with dimension {X.shape[1]}'

        labels = np.array([self._find_bmu(x) for x in X])
        return labels

    def transform(self, X):
        """
        Transform the data X into cluster distance space.

        Args:
        --X: <ndarray> Data of shape (num_samples, self.dim)

        Returns:
        Ndarray of size (num_samples, n_clusters) (the distance to each cluster
        center from each sample).
        """
        # Stack data and cluster centers
        X_stack = np.stack([X]*(self.m*self.n), axis=1)
        cluster_stack = np.stack([self.weights]*X.shape[0], axis=0)

        # Compute difference
        diff = X_stack - cluster_stack

        # Take and return norm
        return np.linalg.norm(diff, axis=2)

    def fit_predict(self, X, **kwargs):
        """
        Convenience method for calling fit(X) followed by predict(X).

        Args:
        --X: <ndarray> Data of shape (num_samples, self.dim)
        --**kwargs: Keyword arguments for the .fit() method

        Returns:
        Index of predicted cluster for each element in X.
        """
        # Fit to data
        self.fit(X, **kwargs)

        # Return predictions
        return self.predict(X)

    def fit_transform(self, X, **kwargs):
        """
        Convenience method for calling fit(X) followed by transform(X). Unlike
        in sklearn, this is not implemented more efficiently (the efficiency is
        the same as calling fit(X) directly followed by transform(X)).

        Args:
        --X: <ndarray> Data of shape (num_samples, self.dim)
        --**kwargs: Keyword arguments for the .fit() method

        Returns:
        Ndarray of size (num_samples, n_clusters) (the distance to each cluster
        center from each sample).
        """
        # Fit to data
        self.fit(X, **kwargs)

        # Return points in cluster distance space
        return self.transform(X)

    @property
    def cluster_centers_(self):
        return self.weights.reshape(self.m, self.n, self.dim)

    @property
    def inertia_(self):
        if self._inertia_ is None:
            raise AttributeError('SOM does not have inertia until after calling fit()')
        return self._inertia_

    @property
    def n_iter_(self):
        if self._n_iter_ is None:
            raise AttributeError('SOM does not have n_iter_ attribute until after calling fit()')
        return self._n_iter_

import sklearn
from sklearn.cluster import KMeans
from sklearn import datasets
iris = datasets.load_iris()
iris_data = iris.data

som = SOM(m=3, n=1, dim=4)

som.fit(iris_data)

centers = som.transform(iris_data)

som.cluster_centers_

t = iris_data[0]
centers[0]
c1, c2, c3 = som.cluster_centers_.squeeze(1)
c1
np.linalg.norm(t - c1)
np.linalg.norm(t - c2)
np.linalg.norm(t - c3)

som = SOM(m=3, n=1, dim=4)
preds = som.fit_predict(iris_data)
preds
preds.shape
irirs_label = iris.target
diff = np.where(irirs_label == preds, np.ones_like(preds), np.zeros_like(preds))
diff.mean()

som = SOM(m=3, n=1, dim=4)
transformed = som.fit_transform(iris_data)
transformed.shape
transformed[0]
som.cluster_centers_[0]
