from copy import deepcopy as dcopy
import os
import fnmatch
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import torch
import sys
import warnings
import matplotlib
warnings.filterwarnings("ignore")
matplotlib.use('Agg')


def create_next_exp_folder(base_directory):
    # List all folders in the base directory that match the 'exp*' pattern
    exp_folders = [folder for folder in os.listdir(base_directory)
                   if os.path.isdir(os.path.join(base_directory, folder))
                   and fnmatch.fnmatch(folder, 'exp*')]

    if not exp_folders:
        # If no 'exp*' folder exists, create 'exp1'
        new_folder = os.path.join(base_directory, 'exp1')
        os.makedirs(new_folder)
        print(f"Created folder: {new_folder}")
    else:
        # Extract the numbers from the existing 'exp*' folders and find the highest one
        exp_numbers = [int(folder[3:]) for folder in exp_folders if folder[3:].isdigit()]

        if exp_numbers:
            next_number = max(exp_numbers) + 1
        else:
            # If there are no valid numbers after 'exp', start with 1
            next_number = 1

        # Create the next folder with the highest number + 1
        new_folder = os.path.join(base_directory, f'exp{next_number}')
        os.makedirs(new_folder)
        print(f"Created folder: {new_folder}")
    return new_folder

def load_pretrained_model(model, w_path, nclasses):
    pretrained_dict = torch.load(w_path)
    if model == 'resnet18':
        from torchvision.models import resnet18
        model_2d = resnet18()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'fc' not in k}
        model_2d.fc = torch.nn.Linear(512, nclasses)
    elif model == 'resnet50':
        from torchvision.models import resnet50
        model_2d = resnet50()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'fc' not in k}
        model_2d.fc = torch.nn.Linear(2048, nclasses)
    elif model == 'resnet101':
        from torchvision.models import resnet101
        model_2d = resnet101()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'fc' not in k}
        model_2d.fc = torch.nn.Linear(2048, nclasses)
    elif model == 'ADnet':
        from models import CNN_8CL_2D, CNN_2D
        net_config = CNN_8CL_2D()
        model_2d = CNN_2D(net_config)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'f' not in k}
        model_2d.f[-1] = torch.nn.Linear(256, nclasses)
    else:
        print('Error. Possible pre-trained models: resnet18, resnet50, resnet101, ADnet')
        sys.exit()
    model_2d.load_state_dict(pretrained_dict, strict=False)
    return model_2d


def load_ACS_pretrained_model(model, config, w_path):
    pretrained_dict = torch.load(w_path)
    if 'ADnet' in model:
        from models import CNN_8CL_2D, CNN_2D
        net_config = CNN_8CL_2D(config.input_dim)
        model_2d = CNN_2D(net_config)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'f' not in k}
        model_2d.f[-1] = torch.nn.Linear(64, config.n_classes)
    else:
        if model == 'resnet18':
            from torchvision.models import resnet18, ResNet18_Weights
            model_2d = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model_2d.fc = torch.nn.Linear(512, config.n_classes)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'fc' not in k}
        elif model == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            model_2d = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            model_2d.fc = torch.nn.Linear(2048, config.n_classes)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'fc' not in k}
        else:
            print('Model not found. Please select a model among: ADnet, ADnet_extractor, Resnet18,'
                  ' Resnet50, ResNet101 or SqueezeNet')
            sys.exit()
    model_2d.load_state_dict(pretrained_dict, strict=False)
    from acsconv.converters import ACSConverter
    net = ACSConverter(model_2d)
    next(net.parameters()).requires_grad = False
    return net


class Trainer:
    def __init__(self, net, config, saver_path, logfile, tfwriter, eval_metric='accuracy', kfold=None):
        self.net = net
        self.config = config
        self.saver_path = saver_path
        self.logfile = logfile
        self.tfwriter = tfwriter
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = self.config.criterion
        self.optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate, weight_decay=config.l2_lambda)
        self.best_score = 0.0
        self.best_epoch = 0
        self.errors = 0
        self.net_weights = dcopy(net.state_dict())
        self.kfold = kfold
        if eval_metric and eval_metric not in ['accuracy', 'f1']:
            raise ValueError(f"Invalid eval_metric: {eval_metric}. Must be 'accuracy' or 'f1'.")
        self.eval_metric = eval_metric

    def eval(self, data_loader, test=False):
        self.net = self.net.to(self.device)
        self.net.eval()
        tot_acc, N = 0.0, 0
        all_preds = torch.tensor([], dtype=torch.long)
        all_y = torch.tensor([], dtype=torch.long)
        all_probs = torch.tensor([], dtype=torch.float)  # Store probabilities for AUC
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                if self.config.nchannels == 1:
                    x = x.unsqueeze(1)
                predictions = self.net(x)
                acc = torch.sum(torch.max(predictions, 1)[1] == y.to(self.device).long()).cpu()
                tot_acc += acc
                N += x.shape[0]
                y_pred = torch.argmax(predictions, dim=1).detach().cpu()
                all_preds = torch.cat((all_preds, y_pred), dim=0)
                all_y = torch.cat((all_y, y), dim=0)
                if test:
                    # Apply softmax to get probabilities
                    probs = torch.nn.functional.softmax(predictions, dim=1).detach().cpu()
                    all_probs = torch.cat((all_probs, probs), dim=0)

        f1 = f1_score(all_y.numpy(), all_preds.numpy(), average='weighted')
        if test:
            cmatrix = confusion_matrix(all_y.numpy(), all_preds.numpy())
            if self.config.n_classes == 2:
                auc_score = roc_auc_score(all_y.numpy(), all_probs.numpy()[:, 1])
            else:
                auc_score = roc_auc_score(all_y.numpy(), all_probs.numpy(),  average="weighted", multi_class="ovr")
            return (tot_acc / N).item(), f1, cmatrix, auc_score
        else:
            return (tot_acc / N).item(), f1



    def train(self, train_loader, val_loader):
        self.net = self.net.to(self.device)
        total_step = len(train_loader)
        train_acc, val_acc = [], []
        train_f1, val_f1 = [], []

        for epoch_counter in range(self.config.num_epochs):
            epoch_total_loss = 0.0
            epoch_loss = 0.0
            epoch_l1_loss = 0.0
            self.net.train()

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                if self.config.nchannels == 1:
                    batch_x = batch_x.unsqueeze(1)
                outputs = self.net(batch_x)
                loss = self.criterion(outputs, batch_y.long())
                l1_penalty = sum(torch.sum(torch.abs(param)) for param in self.net.parameters())
                total_loss = loss + self.config.l1_lambda * l1_penalty

                epoch_total_loss += total_loss.item()
                epoch_loss += loss.item()
                epoch_l1_loss += l1_penalty.item()

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            train_accuracy, train_f1score = self.eval(train_loader)
            val_accuracy, val_f1score = self.eval(val_loader)

            if self.eval_metric == 'f1':
                current_score = val_f1score
            else:
                current_score = val_accuracy
            self.log_and_save(epoch_counter, total_step, epoch_total_loss, epoch_loss, epoch_l1_loss,
                              train_accuracy, val_accuracy, train_f1score, val_f1score)
            train_acc.append(train_accuracy)
            val_acc.append(val_accuracy)
            train_f1.append(train_f1score)
            val_f1.append(val_f1score)
            if self.eval_metric and self.best_score >= current_score:
                self.errors += 1
                if self.errors > self.config.patience:
                    self.log_early_stopping()
                    break
            else:
                self.save_best_weights(epoch_counter, current_score)
        return train_acc, val_acc

    def log_and_save(self, epoch, total_step, total_loss, loss, l1_loss, train_acc, val_acc, train_f1, val_f1):
        avg_total_loss = total_loss / total_step
        avg_loss = loss / total_step
        avg_l1_loss = l1_loss / total_step

        log_message = (f'Epoch {epoch} - Total Loss {avg_total_loss:.4f} - Loss {avg_loss:.4f} '
                       f'- L1 reg {avg_l1_loss:.4f} - Train Acc {train_acc:.4f} - Val Acc {val_acc:.4f}'
                       f'- Train F1 {train_f1:.4f} - Val F1 {val_f1:.4f}')
        print(log_message)
        self.logfile.write(log_message + '\n')
        self.logfile.flush()
        if self.kfold:
            path = f"Fold{self.kfold}_"
        else:
            path = f""
        self.tfwriter.add_scalar(path + "Loss/total", avg_total_loss, epoch)
        self.tfwriter.add_scalar(path + "Loss/CE", avg_loss, epoch)
        self.tfwriter.add_scalar(path + "Loss/L1reg", avg_l1_loss, epoch)
        self.tfwriter.add_scalar(path + "Accuracy/train", train_acc, epoch)
        self.tfwriter.add_scalar(path + "Accuracy/val", val_acc, epoch)
        self.tfwriter.add_scalar(path + "F1score/train", train_f1, epoch)
        self.tfwriter.add_scalar(path + "F1score/val", val_f1, epoch)

    def save_best_weights(self, epoch, val_score):
        if self.kfold:
            path = f"/Fold{self.kfold}_"
        else:
            path = f"/"
        print(f'Saved weights at epoch {epoch}')
        save_path = self.saver_path + path + 'cnn_best_weights.pt'
        torch.save(dcopy(self.net.state_dict()), save_path)
        print(f'At {save_path}')
        self.net_weights = dcopy(self.net.state_dict())
        self.best_epoch = epoch
        self.best_score = val_score
        self.errors = 0

    def log_early_stopping(self):
        log_message = f'Early stopping applied at epoch {self.best_epoch}. Best val. score {self.best_score:.4f}'
        self.logfile.write(log_message + '\n')
        self.logfile.flush()
        print(log_message)
        self.net.load_state_dict(self.net_weights)
        print('Best weights loaded.')


