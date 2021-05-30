# Spatial_Mbiome

A class of Directional Guassian Mixture Models (GMMs) for recovering microbial spatial structre from MaPS-seq data. Data is from [Sheth et al](https://www.nature.com/articles/s41587-019-0183-2?proof=t).

# Dependencies

```environment.yml``` contains the virtual environment that was used to generate our results. 

# Input

Data from the sm1 experiment in [Sheth et al](https://www.nature.com/articles/s41587-019-0183-2?proof=t) is filtered as described in the manuscript. ```sml_data.p``` contains the filtered data. 

# EM algorithms
```basic_GMM_EM.py``` contains an implemention of a standard Expectation Maximization (EM) algorithm for parameter inference in a naive GMM. ```EM_intercept_inference.py``` and ```EM_twodirection_inference.py``` contains the EM algorithms for the single- and two-directional GMMs respectively. Mathematical proofs of the accuracy of our algorithms can be provided upon request. 

# Model selection
```model_selection_sm1.py``` provides an implemention for loading the filtered sm1 data and finding the best fitted model. We have provided the model selection results for the Cecum dataset. 
