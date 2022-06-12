# GCN_Image_Annotation
Implementation of Graph Convolutional Networks (with dynamic or static adjacency matrix) for Image Annotation on Corel-5k dataset with PyTorch library

## Dataset
There is a 'Corel-5k' folder that contains the (Corel-5k) dataset with 5000 real images (and 13,500 fake images), which has 260 labels in the vocabulary. <br >
(for more information see [CNN_Image_Annotation_dataset](https://github.com/parham1998/CNN_Image_Annotaion#dataset))

## Data Augmentation
I used a multi-label data augmentation method based on Wasserstein-GAN which is fully described here: [CNN_Image_Annotation_data_augmentation](https://github.com/parham1998/CNN_Image_Annotaion#data-augmentation)

## Convolutional models
I chose three convolutional neural networks, including ResNet, ResNeXt, and Xception. As all of these models have been pre-trained, weights do not need to be initialized. I will compare the results obtained in the three mentioned models and choose the best one. <br > 
(more information can be found at  [CNN_Image_Annotation_convolutional_models](https://github.com/parham1998/CNN_Image_Annotaion#convolutional-models))

## Graph Convolutional Networks
As objects normally co-occur in an image, it is desirable to model the label dependencies to improve the recognition performance.
For capturing and exploring such important dependencies, In this study, I employ a model based on graph convolutional networks (GCN). <br >
There are two types of GCNs based on their correlation (adjacency) matrices: 

### 1. Static GCN
As mentioned in the paper of Z-M. Chen, et al, in order to build the static GCN, we need to make a static correlation matrix as *an explicit relationship* between labels and try to enhance the relationship of the embedded words, which indicates *the implicit relation* between labels, with this correlation matrix. <br >
The structures of CNN-GCN & GCN are shown in the images below:
![gcn](https://user-images.githubusercontent.com/85555218/173224754-5c02046d-133b-4c75-8162-261f911fba05.png)
![gcn](https://user-images.githubusercontent.com/85555218/173225823-6846cc2e-1579-43a2-9cbe-cb2e926a2955.png)

**Static Correlation (Adjacency) Matrix:** <br >
Z-M. Chen, et al, model the label correlation dependency as a conditional probability. for example, P(L(person) | L(surfboard)) denotes the probability of occurrence of label L(person) when label L(surfboard) appears. As shown in the image below P(L(person) | L(surfboard))  is not equal to P(L(surfboard) | L(person)).
Thus, the correlation matrix is **asymmetrical**.
![correlatin](https://user-images.githubusercontent.com/85555218/173229994-003412ec-94a8-42be-9df9-5f5798817efc.png)

**Word Embedding:** <br >
As shown in the images above, the labels need to be vectorized before they are sent to GCN. There are many word embedding techniques, but for static GCN we use GloVe embeddings, which have shown better results than the other methods. <br >
*(different word embeddings will hardly affect the accuracy, which reveals improvements do not absolutely come from the semantic meanings derived from word embeddings, rather than our GCN)* <br>
The image below shows the relations between labels after using the word embedding technique (t-sne: 300d -> 2d):
![t-sne_glove](https://user-images.githubusercontent.com/85555218/173232298-77d5d410-16c8-4439-b649-36db158cacb7.png)

The image below shows the relations between labels after training by GCN (t-sne: 2048d -> 2d):
![t-sne_xception](https://user-images.githubusercontent.com/85555218/173232165-f19ad20b-2f11-43ef-b13e-d33fa52222f3.png)

### 2. Dynamic GCN
coming soon ...

## Evaluation Metrics
Precision, Recall, F1-score, and N+ are the most popular metrics for evaluating different models in image annotation tasks.
I've used per-class (per-label) and per-image (overall) precision, recall, and f1-score, and also N+ which are common in image annotation papers. <br >
(check out [CNN_Image_Annotation_evaluation_metrics](https://github.com/parham1998/CNN_Image_Annotaion#evaluation-metrics) for more information)

## Reults
### 1: binary-cross-entropy loss + (threshold optimization with matthews correlation coefficient (more information at [MCC](https://github.com/parham1998/CNN_Image_Annotaion#2-binary-cross-entropy-loss--threshold-optimization-with-matthews-correlation-coefficient))
The binary-cross-entropy loss function is one the most popular loss functions in image annotation tasks. PyTorch has this loss function in several types, but I used *MultiLabelSoftMarginLoss*. <br />
![BCE](https://user-images.githubusercontent.com/85555218/130955151-213a3c51-dc66-4888-b842-f1968ee2492f.jpg)
  
model: ResNeXt50 <br />

global-pooling | batch-size | num of training images | image-size | optimizer | epoch time
------------ | ------------- | ------------- | ------------- | ------------- | -------------
avg | 32 | 4500 | 224 * 224 | Adam | 45s
  
data | precision | recall | f1-score 
------------ | ------------- | ------------- | -------------
*trainset* per-image metrics | 0.974 | 0.954 | 0.964 
*testset* per-image metrics | 0.677  | 0.582 | 0.626
*trainset* per-class metrics | 0.963 | 0.913 | 0.937
*testset* per-class metrics | 0.431 | 0.374 | 0.400
*testset* per-class metrics + MCC | 0.454 | 0.391 | **0.420**

data | N+ 
------------ | ------------- 
*trainset* | 259
*testset* | 146
*testset + MCC* | 151

<hr >

model: Xception <br />

global-pooling | batch-size | num of training images | image-size | optimizer | epoch time
------------ | ------------- | ------------- | ------------- | ------------- | -------------
avg | 24 | 4500 | 448 * 448 | Adam | 163s
  
data | precision | recall | f1-score 
------------ | ------------- | ------------- | -------------
*trainset* per-image metrics | 0.986 | 0.978 | 0.982 
*testset* per-image metrics | 0.695  | 0.595 | 0.641
*trainset* per-class metrics | 0.980 | 0.958 | 0.969
*testset* per-class metrics | 0.429 | 0.383 | 0.405
*testset* per-class metrics + MCC | 0.423 | 0.407 | **0.415**

data | N+ 
------------ | ------------- 
*trainset* | 259
*testset* | 145
*testset + MCC* | 150

### 2: focal loss (more information at [focal_loss](https://github.com/parham1998/CNN_Image_Annotaion#3-focal-loss--threshold-optimization-with-matthews-correlation-coefficient))

model: Xception <br />

global-pooling | batch-size | num of training images | image-size | optimizer | epoch time | lambda
------------ | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
avg | 24 | 4500 | 448 * 448 | Adam | 163s | 3
  
data | precision | recall | f1-score 
------------ | ------------- | ------------- | -------------
*trainset* per-image metrics | 0.959 | 0.921 | 0.939 
*testset* per-image metrics | 0.681  | 0.588 | 0.631
*trainset* per-class metrics | 0.938  | 0.853 | 0.894
*testset* per-class metrics | 0.445 | 0.419 | 0.431
*testset* per-class metrics + Th = 0.4 | 0.405 | 0.482 | **0.440**

data | N+ 
------------ | ------------- 
*trainset* | 258
*testset* | 154
*testset + Th = 0.4* | 168

result on test dataset (focal loss)
![test-data](https://user-images.githubusercontent.com/85555218/173231853-f4d818cc-d311-4888-bfa0-0c77c6bdadce.png)

## Conclusions
coming soon ...

## References
Z-M. Chen, X-S. Wei, P. Wang, and Y. Guo. <br />
*"Multi-Label Image Recognition with Graph Convolutional Networks"* (CVPR - 2019)

J. Ye, J. He, X. Peng, W. Wu, and Y. Qiao. <br />
*"Attention-Driven Dynamic Graph Convolutional Network for Multi-Label Image Recognition"* (ECCV - 2020)
