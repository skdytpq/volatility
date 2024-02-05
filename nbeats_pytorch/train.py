import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#import sys
#sys.path.append("./")
import argparse
import logging
import os
from model import NBeatsNet
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import utils
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm
import pdb
#import model.net as net
from evaluate import evaluate
from dataloader import *
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger('DeepAR.Train')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='', help='Name of the dataset')
parser.add_argument('--data-folder', default='../traindata', help='Parent dir of the dataset')
parser.add_argument('--model-name', default='content', help='Directory containing params.json')
parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--sampling', action='store_true', help='Whether to sample during evaluation')
parser.add_argument('--save-best', action='store_true', help='Whether to save best ND to param_search.txt')
parser.add_argument('--restore-file', default=None,
                    help='Optional, name of the file in --model_dir containing weights to reload before \
                    training')  # 'best' or 'epoch_#'


def train(model: nn.Module,
          optimizer: optim,
          loss_fn,
          train_loader: DataLoader,
          test_loader: DataLoader,
          params: utils.Params,
          epoch: int) -> float:
    '''Train the model on one epoch by batches.
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        train_loader: load train data and labels
        test_loader: load test data and labels
        params: (Params) hyperparameters
        epoch: (int) the current training epoch
    '''
    model.train()
    loss_epoch = np.zeros(len(train_loader))
    # Train_loader:
    # train_batch ([batch_size, train_window, 1+cov_dim]): z_{0:T-1} + x_{1:T}, note that z_0 = 0;
    # idx ([batch_size]): one integer denoting the time series id;
    # labels_batch ([batch_size, train_window]): z_{1:T}.
    for i, (train_batch, idx, labels_batch) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        batch_size = train_batch.shape[0]
        train_batch = train_batch.to(torch.float32).to(params.device)  # not scaled
        labels_batch = labels_batch.to(torch.float32).to(params.device)  # not scaled 24 , batch , 1
        labels_batch = labels_batch[:,-params.forecast_length:]
        train_batch = train_batch[:,:-params.forecast_length,:].squeeze(-1)
        
        #labels_batch = train_batch.unsqueeze(-1)[:,:,-1]
        idx = idx.unsqueeze(0).to(params.device)
        _, forecast = model(torch.tensor(train_batch, dtype=torch.float).to(params.device))
        loss = F.mse_loss(forecast, torch.tensor(labels_batch, dtype=torch.float).to(params.device))
        loss.backward()  
        optimizer.step()
        loss = loss.item() / params.train_window  # loss per timestep
        loss_epoch[i] = loss
        if i % 1000 == 0:
            test_metrics = evaluate(model, loss_fn, test_loader, params, epoch, sample=args.sampling)
            model.train()
            logger.info(f'train_loss: {loss}')
        if i == 0:
            logger.info(f'train_loss: {loss}')
    return loss_epoch


def train_and_evaluate(model: nn.Module,
                       train_loader: DataLoader,
                       test_loader: DataLoader,
                       optimizer: optim, loss_fn,
                       params: utils.Params,
                       restore_file: str = None) -> None:
    '''Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the Deep AR model
        train_loader: load train data and labels
        test_loader: load test data and labels
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        params: (Params) hyperparameters
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    '''
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(params.model_dir, restore_file + '.pth.tar')
        logger.info('Restoring parameters from {}'.format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)
    logger.info('begin training and evaluation')
    best_test_ND = float('inf')
    train_len = len(train_loader)
    ND_summary = np.zeros(params.num_epochs)
    loss_summary = np.zeros((train_len * params.num_epochs))
    for epoch in range(params.num_epochs):
        logger.info('Epoch {}/{}'.format(epoch + 1, params.num_epochs))
        loss_summary[epoch * train_len:(epoch + 1) * train_len] = train(model, optimizer, loss_fn, train_loader,
                                                                        test_loader, params, epoch)
        test_metrics = evaluate(model, loss_fn, test_loader, params, epoch, sample=True)#args.sampling)# always True
        ND_summary[epoch] = test_metrics['ND']
        is_best = ND_summary[epoch] <= best_test_ND

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              epoch=epoch,
                              is_best=is_best,
                              checkpoint=params.model_dir)

        if is_best:
            logger.info('- Found new best ND')
            best_test_ND = ND_summary[epoch]
            best_json_path = os.path.join(params.model_dir, 'metrics_test_best_weights.json')
            utils.save_dict_to_json(test_metrics, best_json_path)

        logger.info('Current Best ND is: %.5f' % best_test_ND)

        utils.plot_all_epoch(ND_summary[:epoch + 1], args.dataset + '_ND', params.plot_dir)
        utils.plot_all_epoch(loss_summary[:(epoch + 1) * train_len], args.dataset + '_loss', params.plot_dir)

        last_json_path = os.path.join(params.model_dir, 'metrics_test_last_weights.json')
        utils.save_dict_to_json(test_metrics, last_json_path)

    if args.save_best:
        f = open('./param_search.txt', 'w')
        f.write('-----------\n')
        list_of_params = args.search_params.split(',')
        print_params = ''
        for param in list_of_params:
            param_value = getattr(params, param)
            print_params += f'{param}: {param_value:.2f}'
        print_params = print_params[:-1]
        f.write(print_params + '\n')
        f.write('Best ND: ' + str(best_test_ND) + '\n')
        logger.info(print_params)
        logger.info(f'Best ND: {best_test_ND}')
        f.close()
        utils.plot_all_epoch(ND_summary, print_params + '_ND', location=params.plot_dir)
        utils.plot_all_epoch(loss_summary, print_params + '_loss', location=params.plot_dir)


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    model_dir = os.path.join(args.model_name)
    json_path = 'params.json'
    data_dir = os.path.join(args.data_folder, args.dataset)
    assert os.path.isfile(json_path), f'No json configuration file found at {json_path}'
    params = utils.Params(json_path)

    params.relative_metrics = args.relative_metrics
    params.sampling =  args.sampling
    params.model_dir = model_dir
   # params.plot_dir = os.path.join(model_dir, 'figures')
    params.plot_dir = os.path.join('figures')

    # create missing directories
    try:
        os.mkdir(params.plot_dir)
    except FileExistsError:
        pass

    # use GPU if available
    cuda_exist =torch.cuda.is_available()
    # Set random seeds for reproducible experiments if necessary
    if cuda_exist:
        params.device =torch.device("cuda")
        # torch.cuda.manual_seed(240)
        logger.info('Using Cuda...')
        model  = NBeatsNet(
        stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),
        forecast_length=params.forecast_length,
        backcast_length=params.backcast_length,
        hidden_layer_units=128,
    ).to('cuda')
    else:
        params.device = torch.device('cpu')
        # torch.manual_seed(230)
        logger.info('Not using cuda...')
        model  = NBeatsNet(
        stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),
        forecast_length=params.forecast_length,
        backcast_length=params.backcast_length,
        hidden_layer_units=128,
    ).to('cpu')
    print(os.getcwd())
    #net = model
    #utils.set_logger(os.path.join(model_dir, 'train.log'))
    utils.set_logger(os.path.join('train.log'))
    logger.info('Loading the datasets...')

    train_set = TrainDataset(data_dir, 'coin', params.num_class)
    test_set = TestDataset(data_dir, 'coin', params.num_class)
    sampler = WeightedSampler(data_dir, 'coin') # Use weighted sampler instead of random sampler
    train_loader = DataLoader(train_set, batch_size=params.batch_size, sampler=sampler, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=params.predict_batch, sampler=RandomSampler(test_set),num_workers=4)
    logger.info('Loading complete.')
    logger.info(f'Model: \n{str(model)}')
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate,weight_decay= 0.01)

    # fetch loss function
    loss_fn = F.mse_loss
    
    # Train the model
    logger.info('Starting training for {} epoch(s)'.format(params.num_epochs))
    train_and_evaluate(model,
                       train_loader,
                       test_loader,
                       optimizer,
                       loss_fn,
                       params,
                       args.restore_file)
