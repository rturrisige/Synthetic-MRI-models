from logging import raiseExceptions

import numpy as np
import os
import argparse
from data_utilities import Loader
from utilities import load_pretrained_model, Trainer, create_next_exp_folder
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import ConfusionMatrixDisplay
from torch import nn
from torch.utils.data import DataLoader


def finetuning(config, model, w_path, data_path, saver_path, all_results=None):

    # SAVE TRAINING INFORMATION
    saving_log = open(saver_path + "/logfile.txt", "w")
    saving_log.write('Model {}\n'.format(model))
    saving_log.write('Pretrained weights from {}\n'.format(w_path))

    saving_log.write('\nTraining parameters\n')
    for (k, v) in config.__dict__.items():
        if 'input_dim' not in k:
            saving_log.write(k + ': ' + str(v) + '\n')

    saving_log.write('\n')

    tfwriter = SummaryWriter(log_dir=saver_path)

    # LOAD TRAINING AND VALIDATION DATA
    train_files, train_labels = np.load(data_path + '/train_images.npy'), np.load(data_path + '/train_labels.npy').squeeze(1)
    val_files, val_labels = np.load(data_path + '/val_images.npy'), np.load(data_path + '/val_labels.npy').squeeze(1)

    if len(train_files.shape) == 3:
        train_files = train_files[:, :, :, None]
        val_files = val_files[:, :, :, None]


    train_data = Loader(train_files, train_labels, nchannels=config.nchannels, size=config.input_dim,
                        normalization=config.normalization)

    train_loader = DataLoader(train_data, sampler=None, batch_size=config.train_batch_size, num_workers=1,drop_last=False, pin_memory=True,
                              prefetch_factor=2)

    val_data = Loader(val_files, val_labels,  nchannels=config.nchannels, size=config.input_dim,
                      normalization=config.normalization)
    val_loader = DataLoader(val_data, batch_size=config.val_batch_size, shuffle=False, drop_last=False, pin_memory=True)


    print('')
    print('Total number of training samples {}'.format(len(train_files)))
    print('Total number of validation samples {}'.format(len(val_files)))
    print('')
    saving_log.write('Total number of training files {}\n'.format(len(train_files)))
    saving_log.write('Total number of validation files {}\n'.format(len(val_files)))

    net = load_pretrained_model(model, w_path, config.n_classes, config.input_dim)

    # freeze first layer
    next(net.parameters()).requires_grad = False


    # Create Trainer and start training
    trainer = Trainer(
        net=net,
        config=config,
        saver_path=saver_path,
        logfile=saving_log,
        tfwriter=tfwriter,
        eval_metric=config.es_metric,
    )

    train_accuracy, val_accuracy = trainer.train(train_loader, val_loader)
    del train_files, train_labels, val_files, val_labels

    # LOAD TEST DATA
    test_files, test_labels = np.load(data_path + '/test_images.npy'), np.load(data_path + '/test_labels.npy').squeeze(1)
    if len(test_files.shape) == 3:
        test_files = test_files[:, :, :, None]

    test_data = Loader(test_files, test_labels, nchannels=config.nchannels, size=config.input_dim,
                       normalization=config.normalization)
    test_loader = DataLoader(test_data, batch_size=config.test_batch_size, shuffle=False,
                             drop_last=False, pin_memory=True)
    print('')
    print('Total number of testing samples {}'.format(len(test_files)))
    print('')
    saving_log.write('Total number of testing files {}\n'.format(len(test_files)))

    # EVALUATE TEST DATA
    test_accuracy, test_f1, test_cm, test_auc = trainer.eval(test_loader, test=True)


    tfwriter.add_scalar("Test/acc", test_accuracy)
    tfwriter.add_scalar("Test/f1", test_f1)
    tfwriter.add_scalar("Test/AUC", test_auc)

    saving_log.write('Testing Accuracy: {:.4f} - F1 score: {:.4f} - AUC  {:.4f}\n'.format(test_accuracy, test_f1, test_auc))
    print('Testing Accuracy: {:.4f} - F1 score: {:.4f} - AUC  {:.4f}\n'.format(test_accuracy, test_f1, test_auc))
    np.save(saver_path + '/test_cmatrix.npy', test_cm)

    plt.figure()
    plt.plot(train_accuracy, label='train')
    plt.plot(val_accuracy, color='orange', label='val')
    plt.plot([np.argmax(val_accuracy)], [np.max(val_accuracy)], 'o', color='orange')
    plt.plot([np.argmax(val_accuracy)], [test_accuracy], 'o', color='green', label='test')
    plt.plot([np.argmax(val_accuracy)] * 2, plt.gca().get_ylim(), '--', color='black')
    plt.legend()
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.savefig(saver_path + '/accuracy.png')
    plt.close()

    saving_log.flush()
    saving_log.close()

    if all_results:
        nexp = saver_path.split('exp')[1]
        all_results.write('Exp {}. LR {} L1 {} L2 {}. Val acc: {:.4f}. Testing acc: {:.4f}, F1-score: {:.4f}, AUC: {:.4f}\n'.format(nexp,
                                                                                                      config.learning_rate,
                                                                                                      config.l1_lambda,
                                                                                                      config.l2_lambda,
                                                                                                      np.max(val_accuracy),
                                                                                                      test_accuracy, test_f1, test_auc))
        all_results.flush()
        all_results.close()

    disp = ConfusionMatrixDisplay(confusion_matrix=test_cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(saver_path + '/cmatrix.png')
    plt.close()

##


class TrainingConfiguration(object):
    def __init__(self, lr=0.0001, l1=0.0, l2=0.0, nc=1, nclasses=2):
        self.num_epochs = 100  # maximum number of iterations
        self.patience = 30 #patience for early stopping
        self.learning_rate = lr  # learning rate
        self.l1_lambda = l1  # l1 penalty
        self.l2_lambda = l2 # l2 penalty
        self.es_metric = 'accuracy' # main evaluation metrics for early stopping
        self.nchannels = nc # number of input channels
        self.n_classes = nclasses # number of classes
        self.criterion = nn.CrossEntropyLoss() # loss
        self.input_dim = 128 # input dimension
        self.normalization = 'minmax' # normalization type
        self.train_batch_size = 100 # training batch size
        self.val_batch_size = 100 # validation batch size
        self.test_batch_size = 100 # testing batch size



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Fine tuning of <model>.
    File from <data_dir> are loaded and used as input for the <model>. 
    Training and evaluation results are saved in
    <saver_dir>.""")
    parser.add_argument('--model', default='resnet18', type=str,
                        help='The pre-trained model adopted for transfer learning: '
                             'ADnet, Resnet18, or Resnet50',
                        choices = ['ADnet', 'resnet18', 'resnet50'])
    parser.add_argument('--data_dir', required=True, type=str,
                        help='The directory that contains npy files to be processed')
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='Learning rate')
    parser.add_argument('--l1', default=0.0, type=float,
                        help='L1 penalty')
    parser.add_argument('--l2', default=0.0, type=float,
                        help='L2 penalty')
    parser.add_argument('--nclasses', default=2, type=int,
                        help='Number of classes')
    parser.add_argument('--saver_dir', default=os.getcwd() + '/Results' )
    args = parser.parse_args()

    print('')
    print('MODEL:', args.model)
    if not os.path.exists(args.saver_dir):
        os.makedirs(args.saver_dir)

    if args.model == 'ADnet':
        nc = 1
    else:
        nc = 3

    all_results = open(args.saver_dir + "/all_results.txt", "a")
    config = TrainingConfiguration(args.lr, args.l1, args.l2, nc, args.nclasses)
    w_dir = './weights/' + args.model + '_' + str(config.input_dim) + 'x' +  str(config.input_dim) + '.pt'
    saver_path = create_next_exp_folder(args.saver_dir)
    finetuning(config, args.model, w_dir, args.data_dir, saver_path, all_results)


