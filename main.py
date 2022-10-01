# =============================================================================
# Import required libraries
# =============================================================================
import os
import argparse
import numpy as np
import torch
from torch import nn

from dataset import *
from image_show import *
from models import *
from loss_function import MultiLabelLoss
from engine import Engine


# =============================================================================
# Define hyperparameters
# =============================================================================
parser = argparse.ArgumentParser(
    description='PyTorch Training for Automatic Image Annotation')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training')
parser.add_argument('--data_root_dir', default='./Corel-5k/', type=str)
parser.add_argument('--image-size', default=448, type=int)
parser.add_argument('--epochs', default=80, type=int)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--num_workers', default=2, type=int,
                    help='number of data loading workers (default: 2)')
parser.add_argument('--learning-rate', default=0.0001, type=float)
parser.add_argument('--loss-function', metavar='NAME',
                    help='loss function (e.g. BCELoss)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    help='evaluation of the model on the validation set')
parser.add_argument('--augmentation', dest='augmentation', action='store_true',
                    help='using more data for training')
parser.add_argument(
    '--save_dir', default='./checkpoints/', type=str, help='save path')


def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    is_train = True if not args.evaluate else False

    train_loader, validation_loader, annotations, classes = make_data_loader(
        args)

    cnn = TResNet(args, pretrained=is_train)
    model = GCNCNN(args,
                   cnn,
                   input_d=300,
                   middle_d=1024,
                   output_d=2048)

    if args.loss_function == 'BCELoss':
        criterion = nn.MultiLabelSoftMarginLoss()
    elif args.loss_function == 'FocalLoss':
        criterion = MultiLabelLoss(gamma_neg=3,
                                   gamma_pos=3,
                                   neg_margin=0)
    elif args.loss_function == 'AsymmetricLoss':
        criterion = MultiLabelLoss(gamma_neg=4,
                                   gamma_pos=0,
                                   neg_margin=0.05)

    if os.path.exists('./glove/Corel-5k_glove.pkl'):
        emb = torch.load('./glove/Corel-5k_glove.pkl')
    else:
        # download "glove.6B.300d.txt" and place it in glove folder
        emb = word_embedding('./glove/glove.6B.300d.txt', classes)

    adj = adjacency_matrix(annotations,
                           len(classes),
                           th=0.1,
                           p=0.2)

    engine = Engine(args,
                    model,
                    emb,
                    adj,
                    criterion,
                    train_loader,
                    validation_loader,
                    len(classes))

    # T-SNE plot before training the model
    tsne_plot(args, emb, classes)

    if is_train:
        engine.initialization(is_train)
        engine.train_iteration()
    else:
        engine.initialization(is_train)
        engine.load_model()
        print('Computing best thresholds: ')
        best_thresholds = engine.matthew_corrcoef(train_loader)
        print(best_thresholds)
        engine.validation(dataloader=validation_loader,
                          mcc=True,
                          thresholds=best_thresholds)
        # T-SNE plot after training the model
        new_emb = model.get_emb(emb, adj)
        tsne_plot(args,
                  np.array(new_emb.detach().cpu()),
                  classes,
                  after_train=True)
        # show images and predicted labels
        images, annotations = iter(validation_loader).next()
        if engine.train_on_GPU():
            images = images.cuda()
        predicted_batch_plot(args,
                             classes,
                             model,
                             emb,
                             adj,
                             images,
                             annotations,
                             best_thresholds=None)
        #
        predicted_batch_plot(args,
                             classes,
                             model,
                             emb,
                             adj,
                             images,
                             annotations,
                             best_thresholds=best_thresholds)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
