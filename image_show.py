# =============================================================================
# Import required libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

from dataset import get_mean_std


def imshow(args, tensor):
    mean, std = get_mean_std(args)
    #
    tensor = tensor.numpy()
    # img shape => (3, h, w), img shape after transpose => (h, w, 3)
    tensor = tensor.transpose(1, 2, 0)
    image = tensor * np.array(std) + np.array(mean)
    image = image.clip(0, 1)
    plt.imshow(image)


def convertBinaryAnnotationsToClasses(annotations, classes):
    labels = []
    annotations = np.array(annotations, dtype='int').tolist()
    for i in range(len(classes)):
        if annotations[i] == 1:
            labels.append(classes[i])
    return labels


# plot one batch of images with grand-truth and predicted annotations
def predicted_batch_plot(args,
                         classes,
                         model,
                         emb,
                         adj,
                         images,
                         annotations,
                         best_thresholds=None):
    model.eval()
    with torch.no_grad():
        outputs = torch.sigmoid(model(images, emb, adj))

    fig = plt.figure(figsize=(80, 30))
    for i in np.arange(args.batch_size):
        ax = fig.add_subplot(4, 8, i+1)
        imshow(args, images[i].cpu())
        #
        gt_anno = convertBinaryAnnotationsToClasses(annotations[i], classes)
        #
        if best_thresholds is None:
            o = np.array(outputs.cpu() > 0.5, dtype='int')
        else:
            for j in range(len(classes)):
                outputs[:, j] = torch.gt(
                    outputs[:, j], best_thresholds[j]).float()
            o = np.array(outputs.cpu(), dtype='int')
        pre_anno = convertBinaryAnnotationsToClasses(o[i], classes)
        #
        string_gt = 'GT: '
        string_pre = 'Pre: '
        if len(gt_anno) != 0:
            for ele in gt_anno:
                string_gt += ele if ele == gt_anno[-1] else ele + ' - '
        #
        if len(pre_anno) != 0:
            for ele in pre_anno:
                string_pre += ele if ele == pre_anno[-1] else ele + ' - '

        ax.set_title(string_gt + '\n' + string_pre)
        if best_thresholds is None:
            plt.savefig(args.data_root_dir + 'batch_plot.jpg')
        else:
            plt.savefig(args.data_root_dir + 'batch_plot_best_thresholds.jpg')


def tsne_plot(args, emb, classes, after_train=False):
    word_emb = np.array(emb)
    words = np.array(classes)
    tsne_model = TSNE(perplexity=40, 
                      n_components=2,
                      init='pca', 
                      n_iter=2500, 
                      random_state=1)
    new_values = tsne_model.fit_transform(word_emb)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(32, 32))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(words[i],
                     xy=(x[i], y[i]),
                     xytext=(6, 4),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    if not after_train:
        plt.savefig(args.data_root_dir + 'before_train_T-SNE.jpg')
    else:
        plt.savefig(args.data_root_dir + 'after_train_T-SNE.jpg')
