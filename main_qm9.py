# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import os
import sys
import datetime
import numpy as np
import copy
import utils
import argparse
import wandb
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9 import dataset
from qm9.models import get_optim, get_model
from equivariant_diffusion import en_diffusion
from equivariant_diffusion import en_cnf
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
import torch
import time
import pickle
from qm9.utils import prepare_context, compute_mean_mad
from train_test import train_epoch, test, analyze_and_save
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=720))


def cleanup():
    dist.destroy_process_group()

def _main(rank, world_size, args, logger):

    dataset_info = get_dataset_info(args.dataset, args.remove_h)

    device = torch.device(f'cuda:{rank:d}' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    utils.create_folders(args)

    if rank == 0:
        # Wandb config
        if args.no_wandb:
            mode = 'disabled'
        else:
            mode = 'online' if args.online else 'offline'
        kwargs = {'entity': 'lipman-lab', 'name': args.exp_name, 'project': 'e3_diffusion', 'config': args,
                'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
        wandb.init(**kwargs)
        wandb.save('*.txt')

    # Retrieve QM9 dataloaders
    dataloaders, _ = dataset.retrieve_dataloaders(args)

    data_dummy = next(iter(dataloaders['train']))

    if len(args.conditioning) > 0:
        logger.info(f'Conditioning on {args.conditioning}')
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
        context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
        context_node_nf = context_dummy.size(2)
    else:
        context_node_nf = 0
        property_norms = None

    args.context_node_nf = context_node_nf

    # Create EGNN flow
    model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'])
    if prop_dist is not None:
        prop_dist.set_normalizer(property_norms)
    model = model.to(device)
    optim = get_optim(args, model)
    # logger.info(model)

    gradnorm_queue = utils.Queue()
    gradnorm_queue.add(3000)  # Add large value that will be flushed.

    if args.resume is not None:
        print('loading model')
        flow_state_dict = torch.load(join(args.resume, f'generative_model_ema.npy'))
        optim_state_dict = torch.load(join(args.resume, f'optim.npy'))
        model.load_state_dict(flow_state_dict)
        optim.load_state_dict(optim_state_dict)

    # Initialize dataparallel if enabled and possible.
    logger.info(f'Training using {torch.cuda.device_count()} GPUs')
    model_dp = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Initialize model copy for exponential moving average of params.
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = flow_utils.EMA(args.ema_decay)

        if args.dp and torch.cuda.device_count() > 1:
            model_ema_dp = DDP(model_ema, device_ids=[rank], find_unused_parameters=True)
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp

    best_nll_val = 1e8
    best_nll_test = 1e8
    for epoch in range(args.start_epoch, args.n_epochs):
        start_epoch = time.time()
        train_epoch(args=args, rank=rank, loader=dataloaders['train'], epoch=epoch, model=model, model_dp=model_dp,
                    model_ema=model_ema, ema=ema, device=device, dtype=dtype, property_norms=property_norms,
                    nodes_dist=nodes_dist, dataset_info=dataset_info,
                    gradnorm_queue=gradnorm_queue, optim=optim, prop_dist=prop_dist)
        if rank == 0:
            logger.info(f"Epoch took {time.time() - start_epoch:.1f} seconds.")

        if rank == 0 and epoch > 0 and epoch % args.test_epochs == 0:
            if isinstance(model, en_diffusion.EnVariationalDiffusion):
                wandb.log(model.log_info(), commit=True)

            if isinstance(model, en_cnf.EnVariationalCNF):
                wandb.log(model.log_info(), commit=True)

            if not args.break_train_epoch:
                analyze_and_save(args=args, epoch=epoch, model_sample=model_ema, nodes_dist=nodes_dist,
                                dataset_info=dataset_info, device=device,
                                prop_dist=prop_dist, n_samples=args.n_stability_samples)
            # nll_val = test(args=args, loader=dataloaders['valid'], epoch=epoch, eval_model=model_ema_dp,
            #             partition='Val', device=device, dtype=dtype, nodes_dist=nodes_dist,
            #             property_norms=property_norms)
            # nll_test = test(args=args, loader=dataloaders['test'], epoch=epoch, eval_model=model_ema_dp,
            #                 partition='Test', device=device, dtype=dtype,
            #                 nodes_dist=nodes_dist, property_norms=property_norms)
            nll_val = 0
            nll_test = 0

            if nll_val < best_nll_val:
                best_nll_val = nll_val
                best_nll_test = nll_test
                if args.save_model:
                    args.current_epoch = epoch + 1
                    utils.save_model(optim, 'outputs/%s/optim.npy' % args.exp_name)
                    utils.save_model(model, 'outputs/%s/generative_model.npy' % args.exp_name)
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema.npy' % args.exp_name)
                    with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                        pickle.dump(args, f)

            if epoch % 10 == 0:
                if args.save_model:
                    utils.save_model(optim, 'outputs/%s/optim_%d.npy' % (args.exp_name, epoch))
                    utils.save_model(model, 'outputs/%s/generative_model_%d.npy' % (args.exp_name, epoch))
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema_%d.npy' % (args.exp_name, epoch))
                    with open('outputs/%s/args_%d.pickle' % (args.exp_name, epoch), 'wb') as f:
                        pickle.dump(args, f)
                    args.current_epoch = epoch + 1
                    utils.save_model(optim, 'outputs/%s/optim.npy' % args.exp_name)
                    utils.save_model(model, 'outputs/%s/generative_model.npy' % args.exp_name)
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema.npy' % args.exp_name)
                    with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                        pickle.dump(args, f)
                    
            logger.info('Val loss: %.4f \t Test loss:  %.4f' % (nll_val, nll_test))
            logger.info('Best val loss: %.4f \t Best test loss:  %.4f' % (best_nll_val, best_nll_test))
            wandb.log({"Val loss ": nll_val}, commit=True)
            wandb.log({"Test loss ": nll_test}, commit=True)
            wandb.log({"Best cross-validated test loss ": best_nll_test}, commit=True)


def main(rank, world_size, args):
    setup(rank, world_size, args.port)

    logger = utils.get_logger(logpath=os.path.join('outputs', args.exp_name, 'main_qm9.log'))

    try:
        _main(rank, world_size, args, logger)
    except:
        import traceback
        logger.error(traceback.format_exc())
        raise

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='E3Diffusion')
    parser.add_argument('--exp_name', type=str, default='debug_10')
    parser.add_argument('--model', type=str, default='egnn_dynamics',
                        help='our_dynamics | schnet | simple_dynamics | '
                            'kernel_dynamics | egnn_dynamics |gnn_dynamics')
    parser.add_argument('--probabilistic_model', type=str, default='diffusion', choices=['diffusion', 'cnf'],
                        help='diffusion')

    # Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
    parser.add_argument('--diffusion_steps', type=int, default=500)
    parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                        help='learned, cosine')
    parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                        )
    parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                        help='vlb, l2')

    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--brute_force', type=eval, default=False,
                        help='True | False')
    parser.add_argument('--actnorm', type=eval, default=True,
                        help='True | False')
    parser.add_argument('--break_train_epoch', type=eval, default=False,
                        help='True | False')
    parser.add_argument('--dp', type=eval, default=True,
                        help='True | False')
    parser.add_argument('--condition_time', type=eval, default=True,
                        help='True | False')
    parser.add_argument('--clip_grad', type=eval, default=True,
                        help='True | False')
    parser.add_argument('--trace', type=str, default='hutch',
                        help='hutch | exact')
    # EGNN args -->
    parser.add_argument('--n_layers', type=int, default=6,
                        help='number of layers')
    parser.add_argument('--inv_sublayers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--nf', type=int, default=128,
                        help='number of layers')
    parser.add_argument('--tanh', type=eval, default=True,
                        help='use tanh in the coord_mlp')
    parser.add_argument('--attention', type=eval, default=True,
                        help='use attention in the EGNN')
    parser.add_argument('--norm_constant', type=float, default=1,
                        help='diff/(|diff| + norm_constant)')
    parser.add_argument('--sin_embedding', type=eval, default=False,
                        help='whether using or not the sin embedding')
    # <-- EGNN args
    parser.add_argument('--ode_regularization', type=float, default=1e-3)
    parser.add_argument('--dataset', type=str, default='qm9',
                        help='qm9 | qm9_second_half (train only on the last 50K samples of the training dataset)')
    parser.add_argument('--datadir', type=str, default='qm9/temp',
                        help='qm9 directory')
    parser.add_argument('--filter_n_atoms', type=int, default=None,
                        help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
    parser.add_argument('--dequantization', type=str, default='argmax_variational',
                        help='uniform | variational | argmax_variational | deterministic')
    parser.add_argument('--n_report_steps', type=int, default=1)
    parser.add_argument('--wandb_usr', type=str)
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
    parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--save_model', type=eval, default=True,
                        help='save model')
    parser.add_argument('--generate_epochs', type=int, default=1,
                        help='save model')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
    parser.add_argument('--test_epochs', type=int, default=10)
    parser.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
    parser.add_argument("--conditioning", nargs='+', default=[],
                        help='arguments : homo | lumo | alpha | gap | mu | Cv' )
    parser.add_argument('--resume', type=str, default=None,
                        help='')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='Amount of EMA decay, 0 means off. A reasonable value'
                            ' is 0.999.')
    parser.add_argument('--augment_noise', type=float, default=0)
    parser.add_argument('--n_stability_samples', type=int, default=500,
                        help='Number of samples to compute the stability')
    parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 1],
                        help='normalize factors for [x, categorical, integer]')
    parser.add_argument('--remove_h', action='store_true')
    parser.add_argument('--include_charges', type=eval, default=True,
                        help='include atom charge or not')
    parser.add_argument('--visualize_every_batch', type=int, default=1e8,
                        help="Can be used to visualize multiple times per epoch")
    parser.add_argument('--normalization_factor', type=float, default=1,
                        help="Normalize the sum aggregation of EGNN")
    parser.add_argument('--aggregation_method', type=str, default='sum',
                        help='"sum" or "mean"')
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    if args.port is None:
        args.port = int(np.random.randint(10000, 20000))

    # args, unparsed_args = parser.parse_known_args()
    args.wandb_usr = utils.get_wandb_username(args.wandb_usr)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # logger
    os.makedirs(os.path.join('outputs', args.exp_name), exist_ok=True)
    logger = utils.get_logger(logpath=os.path.join('outputs', args.exp_name, 'main_qm9.log'))
    logger.info(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        logger.info('Found {} CUDA devices.'.format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info('{} \t Memory: {:.2f}GB'.format(props.name, props.total_memory / (1024**3)))
    else:
        logger.info('WARNING: Using device {}'.format(device))

    if args.resume is not None:
        exp_name = args.exp_name #+ '_resume'
        resume = args.resume
        wandb_usr = args.wandb_usr
        normalization_factor = args.normalization_factor
        aggregation_method = args.aggregation_method
        n_epochs = args.n_epochs

        with open(join(args.resume, 'args.pickle'), 'rb') as f:
            args = pickle.load(f)

        args.resume = resume
        args.break_train_epoch = False
        args.n_epochs = n_epochs

        args.exp_name = exp_name
        args.start_epoch = args.current_epoch
        args.wandb_usr = wandb_usr
        os.makedirs(os.path.join('outputs', args.exp_name), exist_ok=True)
        logger = utils.get_logger(logpath=os.path.join('outputs', args.exp_name, 'main_qm9.log'))

        # Careful with this -->
        if not hasattr(args, 'normalization_factor'):
            args.normalization_factor = normalization_factor
        if not hasattr(args, 'aggregation_method'):
            args.aggregation_method = aggregation_method

        logger.info(args)

    ngpus = torch.cuda.device_count()

    try:
        mp.set_start_method("forkserver")
        mp.spawn(main,
                 args=(ngpus, args),
                 nprocs=ngpus,
                 join=True)
    except Exception:
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
