# sklearn-som
A simple, planar self-organizing map with methods similar to clustering methods in Scikit Learn.

sklearn-som is a minimalist, simple implementation of a Kohonen self organizing map with a planar (rectangular) topology. It is used for clustering data and performing dimensionality reduction. For a brief, all-around introduction to self organizing maps, check out [this helpful article](https://rubikscode.net/2018/08/20/introduction-to-self-organizing-maps/) from Rubik's Code.

### Why another SOM package?
There are already a handful of useful SOM packages available in your machine learning framework of choice. So why make another one? Well, sklearn-som, as the name suggests, is written to interface just like a clustering method you would find in Scikit Learn. It has the advantage of only having one dependency (numpy) and if you are already familiar with Scikit Learn's machine learning API, you will find it easy to get right up to speed with sklearn-som.

### How to use
Using sklearn-som couldn't be easier. First, import the SOM class from the som.som module:
```python
from som.som import SOM
```
Now you will have to create an instance of SOM to cluster data, but first let's get some data. For this part we will use sklearn's Iris Dataset, but you do not need sklearn to use SOM. If you have data from another source, you will not need it. But we are going to use it, so let's grab it. We will also use only the first two features so our results are easier to visualize:
```python
from sklearn import datasets

iris = datasets.load_iris()
iris_data = iris.data[:, :2]
iris_label = iris.target
```
Now, just like with any classifier right from sklearn, we will have to build an SOM instance and call `.fit()` on our data to fit the SOM. We already know that there are 3 classes in the Iris Dataset, so we will use a 3 by 1 structure for our self organizing map, but in practice you may have to try different structures to find what works best for your data. Let's build and fit the som:
```python
iris_som = SOM(m=3, n=1, dim=2)
iris_som.fit(iris_data)
```
Note that when building the instance of SOM, we specify `m` and `n` to get an `m` by `n` matrix of neurons in the self organizing map.

Now also like in sklearn, let's assign each datapoint to a predicted cluster using the `.predict()` method:
```python
predictions = iris_som.predict(iris_data)
```
And let's take a look at how we did:
![Iris Data Results](https://github.com/rileypsmith/sklearn-som/blob/main/example/iris_example.png)

Not bad! For the full example code, including the code to reproduce that plot, see `example/example.py`.
