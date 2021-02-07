sklearn-som v. 1.0.0
====================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Master Documentation
====================


.. class:: SOM(m=3, n=3, dim=3, lr=1, sigma=1, max_iter=3000, **kwargs)
    :module: sklearn_som.som

    The 2-D, rectangular grid self-organizing map class using Numpy.

    |

    **Parameters**

    **m : int**, default=3
        The shape along dimension 0 (vertical) of the SOM.
    **n : int**, default=3
        The shape along dimesnion 1 (horizontal) of the SOM.
    **dim : int**, default=3
        The dimensionality (number of features) of the input space.
    **lr : float**, default=1
        The initial step size for updating the SOM weights.
    **sigma : float**, optional
        Optional parameter for magnitude of change to each weight. Does not
        update over training (as does learning rate). Higher values mean
        more aggressive updates to weights.
    **max_iter : int**, optional
        Optional parameter to stop training if you reach this many
        interation.

    |

**Methods**

----

.. method:: fit(X, epochs=1, shuffle=True)

    Fit the self organizing-map to the given data.

    |

    **Parameters**

    **X : ndarray**
        Training data. Must have shape (n, self.dim) where n is the number
        of training samples.
    **epochs : int**, default=1
        The number of times to loop through the training data when fitting.
    **shuffle : bool**, default True
        Whether or not to randomize the order of train data when fitting.
        Can be seeded with np.random.seed() prior to calling fit.

    |

    **Returns**

    **None**
        Fits the SOM to the given data but does not return anything.

    |

----

.. method:: predict(X)

    Predict cluster for each element in X.

    |

    **Parameters**

    **X : ndarray**
        An ndarray of shape (n, self.dim) where n is the number of samples.
        The data to predict clusters for.

    |

    **Returns**

    **labels : ndarray**
        An ndarray of shape (n,). The predicted cluster index for each item
        in X.

    |

----

.. method:: transform(X)

    Transform the data X into cluster distance space.

    |

    **Parameters**

    **X : ndarray**
        Data of shape (n, self.dim) where n is the number of samples. The
        data to transform.

    |

    **Returns**

    **transformed : ndarray**
        Transformed data of shape (n, self.n*self.m). The Euclidean distance
        from each item in X to each cluster center.

    |

----

.. method:: fit_predict(X, **kwargs)

    Convenience method for calling fit(X) followed by predict(X).

    |

    **Parameters**

    **X : ndarray**
        Data of shape (n, self.dim). The data to fit and then predict.
    **\*\*kwargs**
        Optional keyword arguments for the .fit() method.

    |

    **Returns**

    **labels : ndarray**
        ndarray of shape (n,). The index of the predicted cluster for each
        item in X (after fitting the SOM to the data in X).

    |

----

.. method:: fit_transform(X, **kwargs)

    Convenience method for calling fit(X) followed by transform(X).

    |

    Unlike in sklearn, this is not implemented more efficiently (the efficiency is
    the same as calling fit(X) directly followed by transform(X)).

    |

    **Parameters**

    **X : ndarray**
        Data of shape (n, self.dim) where n is the number of samples.
    **\*\*kwargs**
        Optional keyword arguments for the .fit() method.

    |

    **Returns**

    **transformed : ndarray**
        ndarray of shape (n, self.m*self.n). The Euclidean distance
        from each item in X to each cluster center.
