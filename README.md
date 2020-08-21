# MovieLens-Recommender-System

For the Massive Data mining project, we have compared various Recommendation System Algorithms on the [MovieLens dataset](http://files.grouplens.org/datasets/movielens/ml-latest-small-README.html).

The steps involved are as follows : 

![](images/flow_dia.png?raw=true)  

Comparison of performance of the following algorithms is done :    
1. Baseline
2. Singular Value Decomposition (SVD)
3. k Nearest Neighbors (kNN)
4. Slope One
5. Co-Clustering
6. Matrix Factorization

Since each of the algorithm has its own pros and cons, we employ a hybrid approach combining different algorithms. The least RMSE of 0.8558 is achieved by a hybrid model that combines SVD, SVD++ and SlopeOne, with most weights 0.19, 0.47, 0.19 respectively. The result of the experiments is given in the table below : 
![](images/result_1.png?raw=true)  

Next, we implement a novel interest sequence based CF from the paper [Collaborative Filtering Recommendation on Users’ Interest Sequences](https://doi.org/10.1371/journal.pone.0155739). The results of this approach are:

![](images/result_2.png?raw=true)  

The approaches and the results are described in detail in the attached report. 
