import numpy as np
import os
import torch
import wandb
import torch.nn as nn
import torchvision
from torchvision import transforms
from tqdm.auto import tqdm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

class PytorchTrainingPipeline:
    def __init__(self, model, hyerparameters, device):
        self.model = model.to(device)
        self.config = hyerparameters
        self.device = device
        self.make()

    def model_pipeline(self, project_name, run_name):
        # training pipeline using hyperparameters
        with wandb.init(project=project_name, name=run_name, config=self.config):
            # access all hyperparameters through wandb.config
            config = wandb.config
            # use config to train the model
            self.train()

    def make(self):
        # Make the data_loader obj
        train, test = self.get_data(train=True), \
            self.get_data(train=False)
        self.train_loader = self.make_loader(train, batch_size=self.config['batch_size'])
        self.test_loader = self.make_loader(test, batch_size=self.config['batch_size'])

        # Make the loss the optimizaer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config['learning_rate'])

    def get_data(self, train):
        # Preprocessing Module for Training dataset
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor()
                                              ])
        # Preprocessing Module for Testing dataset
        test_transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor()
                                             ])
        if train:
            # return training dataset
            path = os.path.join(self.config['dataset_dir'], 'train')
            return torchvision.datasets.ImageFolder(path, train_transform)
        else:
            # return testing dataset
            path = os.path.join(self.config['dataset_dir'], 'val')
            return torchvision.datasets.ImageFolder(path, test_transform)

    @staticmethod
    def make_loader(dataset, batch_size, num_workers=4):
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=num_workers)
        return loader

    def train(self):
        # tell wandb to watch what the model gets up to: gradients, weights, etc.
        wandb.watch(self.model, self.criterion, log='all', log_freq=10)

        # run training and track with wandb
        total_batches = len(self.train_loader) * self.config['epochs']
        example_ct = 0  # number of input seen
        batch_ct = 0
        best_test_accuracy = 0  # keep track of the best test accuracy
        for epoch in tqdm(range(self.config['epochs'])):
            for _, (images, labels) in enumerate(self.train_loader):
                log_train = self.train_batch(images, labels, epoch, batch_ct)
                wandb.log(log_train)
                example_ct += len(images)
                batch_ct += 1
            # test
            log_test = self.test(epoch)
            best_test_accuracy = self.save_best_model(best_test_accuracy, log_test)
            wandb.log(log_test)

    def save_best_model(self, best_test_accuracy, log_test):
        if log_test['test_accuracy'] > best_test_accuracy:
            # del old model file
            old_best_checkpoint_path = 'checkpoint/best-{:.3f}.pth'.format(
                best_test_accuracy)
            if os.path.exists(old_best_checkpoint_path):
                os.remove(old_best_checkpoint_path)
            # save new model file
            best_test_accuracy = log_test['test_accuracy']
            new_best_checkpoint_path = 'checkpoint/best-{:.3f}.pth'.format(
                log_test['test_accuracy'])
            torch.save(self.model.state_dict(), new_best_checkpoint_path)
            print('Saving Best Model',
                  'checkpoint/best-{:.3f}.pth'.format(
                      best_test_accuracy))
            best_test_accuracy = log_test['test_accuracy']

        return best_test_accuracy

    def train_batch(self, images, labels, epoch, batch_idx):
        # Put tensor onto GPU
        images, labels = images.to(self.device), labels.to(self.device)

        # Forward pass
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Step with optimizer
        self.optimizer.step()

        # Get all training results
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()
        loss = loss.detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        # Calculate all training metrics and save to train log
        log_train = dict(
            epoch=epoch,
            batch=batch_idx,
            train_loss=loss,
            train_accuracy=accuracy_score(labels, preds),
            train_precison=precision_score(labels, preds, average='macro', zero_division=0),
            train_recall=recall_score(labels, preds, average='macro', zero_division=0),
            train_f1_score=f1_score(labels, preds, average='macro', zero_division=0)
        )
        return log_train

    def test(self, epoch):
        self.model.eval()
        # Create list to store loss, labels and predictions
        loss_list = []
        labels_list = []
        preds_list = []
        # Run the model on some test examples
        with torch.no_grad():
            correct, total = 0, 0
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                # Get Labels and Predictions for testing dataset
                _, preds = torch.max(outputs.data, 1)
                preds = preds.cpu().numpy()
                loss = self.criterion(outputs, labels)
                loss = loss.detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()

                loss_list.append(loss)
                labels_list.extend(labels)
                preds_list.extend(preds)

        log_test = {}
        log_test['epoch'] = epoch
        log_train = dict(
            test_loss=loss,
            test_accuracy=accuracy_score(labels, preds),
            test_precison=precision_score(labels, preds, average='macro', zero_division=0),
            test_recall=recall_score(labels, preds, average='macro', zero_division=0),
            test_f1_score=f1_score(labels, preds, average='macro', zero_division=0)
        )
        return log_train

if __name__ == '__main__':
    # Call GPU backend
    device = torch.device("mps" if torch.has_mps else "cpu")
    # Start Training Server (First time need to login into wandb)
    # TODO: Convert to local training server
    wandb.login()
    # Define config dictionary
    config = dict(
        epochs=30,
        classes=4,
        batch_size=32,
        learning_rate=0.005,
        dataset_dir='DataSet'
    )
    # Load a pretrained model
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)

    # Config Training Pipeline
    pipeline = PytorchTrainingPipeline(model,config,device)
    # Starting Training Pipeline
    pipeline.model_pipeline(project_name='Test_Classification_Project',
                            run_name='Test_001')
    # Clean and Save Run
    wandb.finish()
