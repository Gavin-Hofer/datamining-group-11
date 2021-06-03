# datamining-group-11

Before running the code, retrieve the data from https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD
and paste 'YearPredictionMSD.txt' in the data folder. If running on google colab, you will also have to upload
'utils.py' to the runtime.

For this project we tried a number of different approaches to analyze the YearPredictionMSD dataset.
Our code for each of these approaches in contained in the following python notebooks:

clustering_svd.ipynb - K-Means Clustering and DBSCAN using TruncatedSVD for dimensionality reduction

dnn_classification.ipynb - Contains two Dense Neural Network models to attempt to classify the data based on release year. 
Inputs to the DNN models are preprocessed using PCA for dimensionality reduction. Also contains a K-Means clustering
implementation.

dnn_regression.ipynb - Contains a DNN model with a single output neuron that predicts the release year.