# =============================================================================
# Import required libraries
# =============================================================================
import os
import timeit
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn, optim

from sklearn.manifold import TSNE
from sklearn.metrics import matthews_corrcoef
from keras import backend as K
from tqdm import tqdm

from dataset import AnnotationDataset, corel_5k
from models import *
from loss_function import *

torch.manual_seed(1)
np.random.seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =============================================================================
# Check if CUDA is available
# =============================================================================
train_on_GPU = torch.cuda.is_available()
if not train_on_GPU:
    print('CUDA is not available. Training on CPU ...')
else:
    print('CUDA is available! Training on GPU ...')
    print(torch.cuda.get_device_properties('cuda'))


# =============================================================================
# Load data & data preprocessing
# =============================================================================
(batch_size, worker, input_size, root,
 mean, std, transform_train, transform_validation) = corel_5k()

trainset = AnnotationDataset(root=os.path.join(root, 'images'),
                             annotation_path=os.path.join(root, 'train.json'),
                             # aug_path=os.path.join(root, 'train_aug.json'),
                             transforms=transform_train)

testset = AnnotationDataset(root=os.path.join(root, 'images'),
                            annotation_path=os.path.join(root, 'test.json'),
                            transforms=transform_validation)

classes = trainset.classes


def imshow(tensor):
    tensor = tensor.numpy()
    # img shape => (3, h, w), img shape after transpose => (h, w, 3)
    tensor = tensor.transpose(1, 2, 0)
    image = tensor * np.array(std) + np.array(mean)
    image = image.clip(0, 1)
    plt.imshow(image)


def convertBinaryAnnotationsToClasses(annotations):
    labels = []
    annotations = np.array(annotations, dtype='int').tolist()
    for i in range(len(classes)):
        if annotations[i] == 1:
            labels.append(classes[i])
    return labels


# show one sample image
img, annotations = trainset[4000]
imshow(img)
print(convertBinaryAnnotationsToClasses(annotations))

# data loader
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          num_workers=worker,
                                          shuffle=True)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=batch_size,
                                         num_workers=worker,
                                         shuffle=False)


# =============================================================================
# Show one batch of images
# =============================================================================
# get one batch of images
images, annotations = iter(testloader).next()


# plot one batch of images with corresponding annotations
def batch_plot(classes, images, annotations):
    fig = plt.figure(figsize=(50, 25))
    for i in np.arange(batch_size):
        ax = fig.add_subplot(4, 8, i+1)
        imshow(images[i])
        gt_anno = convertBinaryAnnotationsToClasses(annotations[i])
        string_gt = 'GT: '

        if len(gt_anno) != 0:
            for ele in gt_anno:
                string_gt += ele if ele == gt_anno[-1] else ele + ' - '

        ax.set_title(string_gt, color='g')


batch_plot(classes, images, annotations)


# =============================================================================
# Calculate label distribution for the entire dataset
# =============================================================================
samples = testset.annotations + trainset.annotations
samples = np.array(samples)
class_counts = np.sum(samples, axis=0)
# Sort labels according to their frequency in the dataset
sorted_ids = np.array([i[0] for i in sorted(
    enumerate(class_counts), key=lambda x: x[1])], dtype=int)
print('Label distribution (count, class name):', list(
    zip(class_counts[sorted_ids].astype(int), np.array(classes)[sorted_ids])))


# =============================================================================
# Define adjacency matrix
# =============================================================================
def adjacency(num_classes, t=0.1, p=0.2):
    adj = np.zeros((num_classes, num_classes))
    anno = np.array(trainset.annotations)
    sum_anno = np.sum(anno, axis=0)
    for i in range(0, num_classes):
        N = sum_anno[i]
        for j in range(0, num_classes):
            if i != j:
                M = np.sum(anno[:, i] * anno[:, j])
                adj[i, j] = M/N
    # binary
    adj[adj < t] = 0
    adj[adj >= t] = 1
    #
    adj = adj * p / (adj.sum(0, keepdims=True) + K.epsilon())
    adj = adj + (1-p) * np.identity(num_classes, np.int32)
    return torch.Tensor(adj)


adj = adjacency(len(classes))


# =============================================================================
# Word embedding
# =============================================================================
'''
def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding='UTF-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float32)
    return word_to_vec_map

# download "glove.6B.300d.txt" and place it in glove folder
word_to_vec_map = read_glove_vecs('./glove/glove.6B.300d.txt')
emb = []
for word in classes:
    emb.append(word_to_vec_map[word])
emb = torch.from_numpy(np.array(emb))
'''
emb = torch.load('./glove/corel-5k_glove.pkl')


# =============================================================================
# t-sne plot show before training
# =============================================================================
def tsne_plot(emb, classes):
    word_emb = np.array(emb)
    words = np.array(classes)

    tsne_model = TSNE(perplexity=40, n_components=2,
                      init='pca', n_iter=2500, random_state=1)
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
    plt.show()


tsne_plot(emb, classes)


# =============================================================================
# Model
# =============================================================================
cnn = TResNet(pretrained=True)
GCN_CNN = GCNCNN(cnn)
print(GCN_CNN)


# =============================================================================
# Specify loss function and optimizer
# =============================================================================
lossFunction = 'AsymmetricLoss'
if lossFunction == 'BCELoss':
    criterion = nn.MultiLabelSoftMarginLoss()
elif lossFunction == 'FocalLoss':
    criterion = MultiLabelLoss(gamma_neg=3, 
							   gamma_pos=3,
							   neg_margin=0)
elif lossFunction == 'AsymmetricLoss':
    criterion = MultiLabelLoss(gamma_neg=4,
							   gamma_pos=0,
							   neg_margin=0.05)


opt = "Adam"
if opt == "SGD":
    epochs = 200
    lr = 0.5
    optimizer = optim.SGD(GCN_CNN.get_config_optim(lr=lr), 
					      lr=lr, 
						  momentum=0.9, 
						  weight_decay=1e-4)
elif opt == "Adam":
    epochs = 80
    lr = 0.0001
    optimizer = optim.Adam(GCN_CNN.get_config_optim(lr=lr),
						   lr=lr)
    steps_per_epoch = len(trainloader)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 
											  max_lr=lr, 
											  steps_per_epoch=steps_per_epoch, 
											  epochs=epochs, 
                                              pct_start=0.2)


def adjust_learning_rate():
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.1


def count_learnabel_parameters(params):
    return sum(p.numel() for p in params)


count_learnabel_parameters(GCN_CNN.cnn.parameters())
count_learnabel_parameters(GCN_CNN.gc1.parameters())
count_learnabel_parameters(GCN_CNN.gc2.parameters())
count_learnabel_parameters(GCN_CNN.parameters())


# =============================================================================
# Define precision, recall, f1-score, and N+ as the evaluation metrics
# =============================================================================
def per_class_precision(targets, outputs):
    tp = torch.sum(targets * outputs, 0)
    predicted = torch.sum(outputs, 0)
    return torch.mean(tp / (predicted + K.epsilon()))


def per_class_recall(targets, outputs):
    tp = torch.sum(targets * outputs, 0)
    grand_truth = torch.sum(targets, 0)
    return torch.mean(tp / (grand_truth + K.epsilon()))


def per_image_precision(targets, outputs):
    tp = torch.sum(targets * outputs)
    predicted = torch.sum(outputs)
    return tp / (predicted + K.epsilon())


def per_image_recall(targets, outputs):
    tp = torch.sum(targets * outputs)
    grand_truth = torch.sum(targets)
    return tp / (grand_truth + K.epsilon())


def f1_score(precision, recall):
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def N_plus(targets, outputs):
    tp = torch.sum(targets * outputs, 0)
    return torch.sum(torch.gt(tp, 0).int())


def calculate_metrics(targets, outputs, threshold):
    if threshold == 0.5:
        outputs = torch.gt(outputs, threshold).float()
    else:
        for i in range(len(classes)):
            outputs[:, i] = torch.gt(outputs[:, i], threshold[i]).float()
    pcp = per_class_precision(targets, outputs)
    pcr = per_class_recall(targets, outputs)
    pip = per_image_precision(targets, outputs)
    pir = per_image_recall(targets, outputs)
    pcf = f1_score(pcp, pcr)
    pif = f1_score(pip, pir)
    n_plus = N_plus(targets, outputs)
    return {'per_class/precision': pcp,
            'per_class/recall': pcr,
            'per_class/f1': pcf,
            'per_image/precision': pip,
            'per_image/recall': pir,
            'per_image/f1': pif,
            'N+': n_plus,
            }


# =============================================================================
# Training
# =============================================================================
best_per_class_f1 = 0

# losses per epoch
train_losses = []
valid_losses = []


# ===========
# train part
# ===========
def train(epoch, dataloader, thresholds=0.5):
    GCN_CNN.train()
    train_loss = 0
    total_outputs = []
    total_targets = []
    for batch_idx, (images, targets) in enumerate(tqdm(dataloader)):

        if train_on_GPU:
            images, targets = images.cuda(), targets.cuda()

        # zero the gradients parameter
        optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to
        # the model
        outputs = GCN_CNN(images, emb, adj)

        # calculate the batch loss
        loss = criterion(outputs, targets)

        # backward pass: compute gradient of the loss with respect to
        # the model parameters
        loss.backward()

        if opt == "SGD":
            nn.utils.clip_grad_norm_(GCN_CNN.parameters(), max_norm=10.0)

        # parameters update
        optimizer.step()
        
        if opt == 'Adam':
            scheduler.step()

        train_loss += loss.item()
        total_outputs.append(torch.sigmoid(outputs))
        total_targets.append(targets)

    train_losses.append(train_loss/(batch_idx+1))
    result = calculate_metrics(
        torch.cat(total_targets), torch.cat(total_outputs), thresholds)
    print('Epoch: {}'.format(epoch+1))
    print('Train Loss: {:.5f}'.format(train_loss/(batch_idx+1)))
    print('N+: {:.0f}'.format(result['N+']))
    print('per-class precision: {:.4f} \t per-class recall: {:.4f} \t per-class f1: {:.4f}'.format(
        result['per_class/precision'], result['per_class/recall'], result['per_class/f1']))
    print('per-image precision: {:.4f} \t per-image recall: {:.4f} \t per-image f1: {:.4f}'.format(
        result['per_image/precision'], result['per_image/recall'], result['per_image/f1']))


# ==============
# validation part
# ==============
def validation(dataloader, mcc=False, thresholds=0.5):
    GCN_CNN.eval()
    valid_loss = 0
    total_outputs = []
    total_targets = []
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader)):

            if train_on_GPU:
                images, targets = images.cuda(), targets.cuda()

            outputs = GCN_CNN(images, emb, adj)
            
            if not mcc:
                loss = criterion(outputs, targets)
                valid_loss += loss.item()

            total_outputs.append(torch.sigmoid(outputs))
            total_targets.append(targets)

    valid_losses.append(valid_loss/(batch_idx+1))
    result = calculate_metrics(
        torch.cat(total_targets), torch.cat(total_outputs), thresholds)
    if not mcc:
        print('Validation Loss: {:.5f}'.format(valid_loss/(batch_idx+1)))
    print('N+: {:.0f}'.format(result['N+']))
    print('per-class precision: {:.4f} \t per-class recall: {:.4f} \t per-class f1: {:.4f}'.format(
        result['per_class/precision'], result['per_class/recall'], result['per_class/f1']))
    print('per-image precision: {:.4f} \t per-image recall: {:.4f} \t per-image f1: {:.4f}'.format(
        result['per_image/precision'], result['per_image/recall'], result['per_image/f1']))

    # save model if test accuracy has increased
    global best_per_class_f1
    if result['per_class/f1'] > best_per_class_f1:
        print('Test per-class f1 increased ({:.4f} --> {:.4f}). saving model ...'.format(
            best_per_class_f1, result['per_class/f1']))
        GCN_CNN.save(cnn.path)
        best_per_class_f1 = result['per_class/f1']


print('==> Start Training ...')
for epoch in range(epochs):
    start = timeit.default_timer()
    train(epoch, trainloader)
    validation(testloader)
    if opt == "SGD" and (epoch == 49 or epoch == 99):
        adjust_learning_rate()
        print(optimizer)
    elif opt == 'Adam':
        print('LR {:.1e}'.format(scheduler.get_last_lr()[0]))
    stop = timeit.default_timer()
    print('time: {:.3f}'.format(stop - start))
    # early stop
    if opt == "Adam" and epoch == 59:
        break
print('==> End of training ...')


# =============================================================================
# Figure loss
# =============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 10))
ax1.plot(train_losses, marker="o", markersize=5)
ax1.set_title("Train Loss")
ax2.plot(test_losses, marker="o", markersize=5)
ax2.set_title("Test Loss")
plt.show()


# =============================================================================
# Load model
# =============================================================================
GCN_CNN.cnn.load_state_dict(torch.load(cnn.path))
GCN_CNN.gc1.load_state_dict(torch.load(GCN_CNN.gcn1_path))
GCN_CNN.gc2.load_state_dict(torch.load(GCN_CNN.gcn2_path))


# =============================================================================
# t-sne plot show after training
# =============================================================================
new_emb = GCN_CNN.get_emb(emb, adj)
tsne_plot(np.array(new_emb.transpose(0, 1).detach().cpu()), classes)


# =============================================================================
# Matthew correlation coefficient (threshold optimization)
# =============================================================================
def matthew_corrcoef(dataloader):
    o = []
    t = []

    GCN_CNN.eval()
    total_outputs = []
    total_targets = []
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):

            if train_on_GPU:
                images, targets = images.cuda(), targets.cuda()

            outputs = torch.sigmoid(GCN_CNN(images, emb, adj))

            total_outputs.append(outputs)
            total_targets.append(targets)

    o.append(torch.cat(total_outputs))
    t.append(torch.cat(total_targets))
    t = np.array(t[0].cpu())
    o = np.array(o[0].cpu())

    best_thresholds = []
    threshold = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

    for i in range(len(classes)):
        mcc = []
        for j in threshold:
            hold = o[:, i].copy()
            hold[hold >= j] = 1
            hold[hold < j] = 0
            mcc.append(matthews_corrcoef(t[:, i], hold))
        best_thresholds.append(threshold[np.argmax(mcc)])

    return best_thresholds


best_thresholds = matthew_corrcoef(traindataloader)
print(best_thresholds)
validation(testloader, mcc=True, thresholds=best_thresholds)


# =============================================================================
# Evaluate the model on a random test batch
# =============================================================================
def evaluation_testset(images, annotations, thresholds='none'):
    if train_on_GPU:
        images = images.cuda()

    GCN_CNN.eval()
    with torch.no_grad():
        outputs = torch.sigmoid(GCN_CNN(images, emb, adj))

    fig = plt.figure(figsize=(60, 25))
    for i in np.arange(32):
        ax = fig.add_subplot(4, 8, i+1)
        imshow(images[i].cpu())
        gt_anno = convertBinaryAnnotationsToClasses(annotations[i])
        if thresholds == 'none':
            o = np.array(outputs.cpu() > 0.5, dtype='int')
        elif thresholds == 'mcc':
            for j in range(len(classes)):
                outputs[:, j] = torch.gt(
                    outputs[:, j], best_thresholds[j]).float()
            o = np.array(outputs.cpu(), dtype='int')

        pre_anno = convertBinaryAnnotationsToClasses(o[i])

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
        plt.savefig('./img.jpg')


images, annotations = iter(testloader).next()
evaluation_testset(images, annotations, thresholds='none')


# =============================================================================
# Evaluate the model on some random images
# =============================================================================
test_image_path = './Corel-5k/unlabeled images/'


def evaluation_random_imgs(img, threshold='none'):
    img = transform_validation(img)
    img = img.unsqueeze(0)
    if train_on_GPU:
        img = img.cuda()

    GCN_CNN.eval()
    with torch.no_grad():
        output = torch.sigmoid(GCN_CNN(img, emb, adj))

    if threshold == 'none':
        o = np.array(output.cpu() > 0, dtype='int')
    elif threshold == 'mcc':
        for j in range(len(classes)):
            output[:, j] = torch.gt(
                output[:, j], best_thresholds[j]).float()
        o = np.array(output.cpu(), dtype='int')

    pre_anno = convertBinaryAnnotationsToClasses(o[0])
    imshow(img.squeeze(0).cpu())
    string_pre = 'Pre: '
    if len(pre_anno) != 0:
        for ele in pre_anno:
            string_pre += ele if ele == pre_anno[-1] else ele + ' - '
    plt.title(string_pre)
    plt.show()


for i in range(1, 11):
    img = Image.open(test_image_path + str(i) + '.jpg')
    evaluation_random_imgs(img, threshold='none')
