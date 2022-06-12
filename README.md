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
model the label correlation dependency as a conditional probability

**Word Embedding:** <br >
As shown in the images above, the labels need to be vectorized before they are sent to GCN. There are many word embedding techniques, but for static GCN we use GloVe embeddings, which have shown better results than the other methods. <br >
*(different word embeddings will hardly affect the accuracy, which reveals improvements do not absolutely come from the semantic meanings derived from word embeddings, rather than our GCN)* <br>
The image below shows the relations between labels after using the word embedding technique (t-sne: 300d -> 2d):
![t-sne_glove](https://user-images.githubusercontent.com/85555218/173227980-947b5dce-4352-4ee5-8863-004aa06db1ba.png)

The image below shows the relations between labels after training by GCN (t-sne: 2048d -> 2d):
![t-sne_xception](https://user-images.githubusercontent.com/85555218/173228140-28f372a3-ab26-452c-9f6e-425f3710fc84.png)

### 2. Dynamic GCN
coming soon ...

## Evaluation Metrics
Precision, Recall, F1-score, and N+ are the most popular metrics for evaluating different models in image annotation tasks.
I've used per-class (per-label) and per-image (overall) precision, recall, and f1-score, and also N+ which are common in image annotation papers. <br >
(check out [CNN_Image_Annotation_evaluation_metrics](https://github.com/parham1998/CNN_Image_Annotaion#evaluation-metrics) for more information)

## Reults

## Conclusions
coming soon ...

## References
Z-M. Chen, X-S. Wei, P. Wang, and Y. Guo. <br />
*"Multi-Label Image Recognition with Graph Convolutional Networks"* (CVPR - 2019)

J. Ye, J. He, X. Peng, W. Wu, and Y. Qiao. <br />
*"Attention-Driven Dynamic Graph Convolutional Network for Multi-Label Image Recognition"* (ECCV - 2020)
