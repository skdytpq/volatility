import argparse
import logging
import os
import sys
sys.path.append("./")
import numpy as np
import torch
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm
import pdb
import utils
from model import NBeatsNet
from dataloader import *
import model as net
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger('DeepAR.Eval')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='elect', help='Name of the dataset')
parser.add_argument('--data-folder', default='data', help='Parent dir of the dataset')
parser.add_argument('--model-name', default='base_model', help='Directory containing params.json')
parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--sampling', action='store_true', help='Whether to sample during evaluation')
parser.add_argument('--restore-file', default='best',
                    help='Optional, name of the file in --model_dir containing weights to reload before \
                    training')  # 'best' or 'epoch_#'


def evaluate(model, loss_fn, test_loader, params, plot_num, sample=True):
    '''Evaluate the model on the test set.
    Args:
        model: (torch.nn.Module) the Deep AR model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        test_loader: load test data and labels
        params: (Params) hyperparameters
        plot_num: (-1): evaluation from evaluate.py; else (epoch): evaluation on epoch
        sample: (boolean) do ancestral sampling or directly use output mu from last time step
    '''
    model.train() # MCDropout
    with torch.no_grad():
      plot_batch = np.random.randint(len(test_loader)-1)

      summary_metric = {}
      raw_metrics = utils.init_metrics(sample=True) # MCdropout

      # Test_loader: 
      # test_batch ([batch_size, train_window, 1+cov_dim]): z_{0:T-1} + x_{1:T}, note that z_0 = 0;
      # id_batch ([batch_size]): one integer denoting the time series id;
      # v ([batch_size, 2]): scaling factor for each window;
      # labels ([batch_size, train_window]): z_{1:T}.
      for i, (test_batch, id_batch, v, labels) in enumerate(tqdm(test_loader)):
          test_batch = test_batch.to(torch.float32).to(params.device)
          id_batch = id_batch.to(params.device)
          v_batch = v.to(torch.float32).to(params.device)

          labels = labels.to(torch.float32).to(params.device)
          batch_size = test_batch.shape[1]
          test_batch = test_batch.to(torch.float32).to(params.device)  # not scaled
         # labels_batch = labels_batch.to(torch.float32).to(params.device)  # not scaled
          labels_batch = labels[:,-params.forecast_length:] # Batch X 1 # backcast 지정
          test_batch = test_batch[:,:-params.forecast_length,:].squeeze(-1)# 23 , batch , 1 backcast = 1
          #labels_batch = test_batch.unsqueeze(-1)[:,:,-1]
      #    idx = idx.unsqueeze(0).to(params.device)
          mc_samples = 30
          pred_i = torch.zeros((mc_samples,test_batch.shape[0],params.forecast_length))
          sample = True
          v_ = v_batch[:,0].unsqueeze(1)
          v_1 = v_batch[:,1].unsqueeze(1)
          v_ = v_.expand(v_batch.shape[0],params.forecast_length)
          v_1 = v_1.expand(v_batch.shape[0],params.forecast_length)
          if sample:
            for iteration in range(mc_samples):
                _, forecast = model(torch.tensor(test_batch, dtype=torch.float).to(params.device))
                forecast = v_ * forecast + v_1
                pred_i[iteration] = forecast
            forecast = pred_i
            forecast = forecast.to(params.device) #iter batch len -> batch iter len
            samples = forecast # iter batch len -> 200 256 6
            sample_mu = torch.mean(forecast,axis=0 )
            sample_mu = v_ * sample_mu + v_1
            sample_sigma = torch.std(forecast,axis=0) #* v_1
            pdb.set_trace()
            raw_metrics = utils.update_metrics(raw_metrics, forecast,  sample_mu, labels_batch , params.forecast_length, samples, relative = params.relative_metrics)
          else:
              sample_sigma,sample_mu = _, forecast = model(torch.tensor(test_batch, dtype=torch.float).to(params.device))
              raw_metrics = utils.update_metrics(raw_metrics, forecast, sample_mu, labels_batch , params.test_predict_start, relative = params.relative_metrics)

          if i == plot_batch:
              if sample:
                  sample_metrics = utils.get_metrics(sample_mu, labels_batch , params.test_predict_start, samples, relative = params.relative_metrics)
              else:
                  sample_metrics = utils.get_metrics(sample_mu, labels_batch , params.test_predict_start, relative = params.relative_metrics)                
              # select 10 from samples with highest error and 10 from the rest
              top_10_nd_sample = (-sample_metrics['ND']).argsort()[:batch_size // 10]  # hard coded to be 10
              chosen = set(top_10_nd_sample.tolist())
              all_samples = set(range(batch_size))
              not_chosen = np.asarray(list(all_samples - chosen))
              if batch_size < 100: # make sure there are enough unique samples to choose top 10 from
                  random_sample_10 = np.random.choice(top_10_nd_sample, size=10, replace=True)
              else:
                  random_sample_10 = np.random.choice(top_10_nd_sample, size=10, replace=False)
              if batch_size < 12: # make sure there are enough unique samples to choose bottom 90 from
                  random_sample_90 = np.random.choice(not_chosen, size=10, replace=True)
              else:
                  random_sample_90 = np.random.choice(not_chosen, size=10, replace=False)
              combined_sample = np.concatenate((random_sample_10, random_sample_90))
              # Labels 는 forecast를 해야하는 Label 데이터
              label_plot = labels[combined_sample].data.cpu().numpy() # 실제 라벨 값
              predict_mu = sample_mu[combined_sample].data.cpu().numpy()
              predict_sigma = sample_sigma[combined_sample].data.cpu().numpy()
              plot_mu = np.concatenate((test_batch[combined_sample].data.cpu().numpy(), predict_mu), axis=1)
              plot_sigma = np.concatenate((test_batch[combined_sample].data.cpu().numpy(), predict_sigma), axis=1)
              plot_metrics = {_k: _v[combined_sample] for _k, _v in sample_metrics.items()}
              plot_eight_windows(params.plot_dir, plot_mu, plot_sigma, label_plot, params.test_window, params.test_predict_start, plot_num, plot_metrics, sample)

      summary_metric = utils.final_metrics(raw_metrics, sampling=sample)
      metrics_string = '; '.join('{}: {:05.3f}'.format(k, v) for k, v in summary_metric.items())
      logger.info('- Full test metrics: ' + metrics_string)
    return summary_metric


def plot_eight_windows(plot_dir,
                       predict_values,
                       predict_sigma,
                       labels,
                       window_size,
                       predict_start,
                       plot_num,
                       plot_metrics,
                       sampling=False):

    x = np.arange(window_size)
    f = plt.figure(figsize=(8, 42), constrained_layout=True)
    nrows = 21
    ncols = 1
    ax = f.subplots(nrows, ncols)
    for k in range(nrows):
        if k == 10:
            ax[k].plot(x, x, color='g')
            ax[k].plot(x, x[::-1], color='g')
            ax[k].set_title('This separates top 10 and bottom 90', fontsize=10)
            continue
        m = k if k < 10 else k - 1
        ax[k].plot(x, predict_values[m], color='b')
        ax[k].fill_between(x[predict_start :], predict_values[m, predict_start :] - 2 * predict_sigma[m, predict_start:],
                         predict_values[m, predict_start:] + 2 * predict_sigma[m, predict_start:], color='blue',
                         alpha=0.2)
        ax[k].plot(x, labels[m, :], color='r')
        ax[k].axvline(predict_start, color='g', linestyle='dashed')

        #metrics = utils.final_metrics_({_k: [_i[k] for _i in _v] for _k, _v in plot_metrics.items()})


        plot_metrics_str = f'ND: {plot_metrics["ND"][m]: .3f} ' \
            f'RMSE: {plot_metrics["RMSE"][m]: .3f}'
        if sampling:
            plot_metrics_str += f' rou90: {plot_metrics["rou90"][m]: .3f} ' \
                                f'rou50: {plot_metrics["rou50"][m]: .3f}'

        ax[k].set_title(plot_metrics_str, fontsize=10)

    f.savefig(os.path.join(plot_dir, str(plot_num) + '.png'))
    plt.close()

if __name__ == '__main__':
    # Load the parameters
    args = parser.parse_args()
    model_dir = os.path.join('experiments', args.model_name) 
    json_path = os.path.join(model_dir, 'params.json')
    data_dir = os.path.join(args.data_folder, args.dataset)
    assert os.path.isfile(json_path), 'No json configuration file found at {}'.format(json_path)
    params = utils.Params(json_path)

    utils.set_logger(os.path.join(model_dir, 'eval.log'))

    params.relative_metrics = args.relative_metrics
    params.sampling = args.sampling
    params.model_dir = model_dir
    params.plot_dir = os.path.join(model_dir, 'figures')
    cuda_exist = torch.cuda.is_available()  # use GPU is available
    # Set random seeds for reproducible experiments if necessary
    if cuda_exist:
        params.device = torch.device('cuda')
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

    # Create the input data pipeline
    logger.info('Loading the datasets...')
    test_set = TestDataset(data_dir, args.dataset, params.num_class)
    test_loader = DataLoader(test_set, batch_size=params.predict_batch, sampler=RandomSampler(test_set), num_workers=4)
    logger.info('- done.')

    print('model: ', model)
    loss_fn = net.loss_fn

    logger.info('Starting evaluation')

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(model_dir, args.restore_file + '.pth.tar'), model)

    test_metrics = evaluate(model, loss_fn, test_loader, params, -1, params.sampling)
    save_path = os.path.join(model_dir, 'metrics_test_{}.json'.format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
