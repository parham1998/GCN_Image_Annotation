# GCN_Image_Annotation
Implementation of Graph Convolutional Network to Annotate Corel-5k images with PyTorch library

## Dataset
<div align="justify"> There is a 'Corel-5k' folder that contains the (Corel-5k) dataset with 5000 real images (and 13,500 fake images), which has 260 labels in the vocabulary. </div>

(for more information see [CNN_Image_Annotation_dataset](https://github.com/parham1998/CNN_Image_Annotaion#dataset))

## Data Augmentation
I used a multi-label data augmentation method based on Wasserstein-GAN which is fully described here: [CNN_Image_Annotation_data_augmentation](https://github.com/parham1998/CNN_Image_Annotaion#data-augmentation)

## Convolutional model
<div align="justify"> As compared to other CNNs in my experiments, TResNet produced the best results for extracting features of images, so it has been chosen as the feature extractor. </div>

(more information can be found at  [CNN_Image_Annotation_convolutional_models](https://github.com/parham1998/CNN_Image_Annotaion#convolutional-models))

## Graph Convolutional Network
<div align="justify"> As objects normally co-occur in an image, it is desirable to model the label dependencies to improve the annotation performance.
For capturing and exploring such important dependencies, In this study, I employ a model based on graph convolutional networks (GCN), which has been described below: </div>

<div align="justify"> As mentioned in the paper of Z-M. Chen, et al, in order to build the static GCN, we need to make a static correlation matrix as <strong> an explicit relationship </strong> between labels and try to enhance the relationship of the embedded words, which indicates <strong> the implicit relation </strong> between labels, by this correlation matrix. </div>

The structures of CNN-GCN & GCN are shown in the images below:
![gcn](https://user-images.githubusercontent.com/85555218/173224754-5c02046d-133b-4c75-8162-261f911fba05.png)
![gcn](https://user-images.githubusercontent.com/85555218/173225823-6846cc2e-1579-43a2-9cbe-cb2e926a2955.png)

**Static Correlation (Adjacency) Matrix:** <br >
<div align="justify">  Z-M. Chen, et al, model the label correlation dependency as a conditional probability. for example, P(L(person) | L(surfboard)) denotes the probability of occurrence of label L(person) when label L(surfboard) appears. As shown in the image below P(L(person) | L(surfboard))  is not equal to P(L(surfboard) | L(person)), Thus, the correlation matrix is an <strong> asymmetrical matrix</strong>. </div>

![correlatin](https://user-images.githubusercontent.com/85555218/173229994-003412ec-94a8-42be-9df9-5f5798817efc.png)

**Word Embedding:** <br >
<div align="justify"> As shown in the images above, the labels need to be vectorized before they are sent to GCN. There are many word embedding techniques, but for static GCN we use GloVe embeddings, which have shown better results than the other methods. </div>

> Different word embeddings will hardly affect the accuracy, which reveals improvements do not absolutely come from the semantic meanings derived from word embeddings, rather than GCN.
  
The image below shows the relations between labels after using the word embedding technique (t-sne: 300d -> 2d):
![t-sne_glove](https://user-images.githubusercontent.com/85555218/173232298-77d5d410-16c8-4439-b649-36db158cacb7.png)

The image below shows the relations between labels after training by GCN (t-sne: 2048d -> 2d):
![t-sne_tresnet](https://user-images.githubusercontent.com/85555218/183426014-88af0b6c-5523-468f-83e2-dd158f0f5e14.png)

## Evaluation Metrics
<div align="justify"> Precision, Recall, F1-score, and N+ are the most popular metrics for evaluating different models in image annotation tasks.
I've used per-class (per-label) and per-image (overall) precision, recall, f1-score, and also N+ which are common in image annotation papers. </div>

(check out [CNN_Image_Annotation_evaluation_metrics](https://github.com/parham1998/CNN_Image_Annotaion#evaluation-metrics) for more information)

## Train and Evaluation
To train the model in Spyder IDE use the code below:
```python
run main.py --loss-function {select loss function}
```
Please note that:
1) You should put **BCELoss**, **FocalLoss** or **AsymmetricLoss** in {select loss function}.
  
Using augmented data, you can train the model as follows:
```python
run main.py --loss-function {select loss function} --augmentation
```
  
To evaluate the model in Spyder IDE use the code below:
```python
run main.py --loss-function {select loss function} --evaluate
```

## Results
### asymmetric loss (more information at [asymmetric loss](https://github.com/parham1998/CNN_Image_Annotaion#3-asymmetric-loss))

| global-pooling | batch-size | num of training images | image-size | epoch time | 𝛾+ | 𝛾- | m 
| :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |
| avg | 32 | 4500 | 448 * 448 | 135s | 0 | 4 | 0.05 |
  
| data | precision | recall | f1-score |
| :------------: | :------------: | :------------: | :------------: |
| *testset* per-image metrics | 0.594  | 0.670 | 0.630 | 
| *testset* per-class metrics | 0.453 | 0.495 | **0.473** |

| data | N+ |
| :------------: | :------------: |
| *testset* | 175 |

## References
Z-M. Chen, X-S. Wei, P. Wang, and Y. Guo. <br />
*"Multi-Label Image Recognition with Graph Convolutional Networks"* (CVPR - 2019)
