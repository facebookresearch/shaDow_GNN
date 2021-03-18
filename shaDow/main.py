# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from shaDow import TRAIN, VALID, TEST, MODE2STR, STR2MODE
from shaDow.metric import Metrics
import numpy as np
from shaDow.globals import (
    args_global,
    meta_config, 
    timestamp,
    DATA_METRIC,
    device,
    Logger
)
from shaDow.minibatch import MinibatchShallowSampler
from shaDow.models import DeepGNN
from shaDow.preproc import PreprocessGraph
from shaDow.utils import (
    parse_n_prepare,
    parse_n_prepare_postproc,
    load_data
)
import time
import sys
import torch
import glob



def instantiate(
        name_data   : str, 
        dir_data    : dict, 
        data_train  : dict, 
        params_train: dict, 
        arch_gnn    : dict, 
        config_sampler_preproc  : dict,
        config_sampler_train    : dict,
        parallelism : int,
        full_tensor_on_gpu      : bool,
        no_pbar     : bool
    ):
    """
    Prepare some data structure and initialize model / minibatch handler before
    the actual iterative training taking place.
    """
    adj_full, adj_train = data_train['adj_full'], data_train['adj_train']
    feat_full, label_full = data_train['feat_full'], data_train['label_full']
    node_set, bin_adj_files = data_train['node_set'], data_train['bin_adj_files']
    is_transductive = (adj_full.size == adj_train.size)
    for k in data_train:
        data_train[k] = None            # dereference everything to help with memory free later on. 
    if bin_adj_files is not None and all(cf['method'] != 'full'
            for cf in config_sampler_preproc['configs'] + config_sampler_train['configs']):         # full sampler skips the C++ layers
        adj_full = adj_train = None
    adjs = {TRAIN: adj_train, VALID: adj_full, TEST: adj_full}
    # preprocess
    dir_data['is_adj_changed'] = dir_data['is_feat_changed'] = False
    dim_feat_raw = feat_full.shape[1]
    dim_label_raw = label_full[label_full == label_full].max().item() + 1 if len(label_full.shape) == 1 else label_full.shape[1]
    if len(config_sampler_preproc['configs']) > 0:
        minibatch_preproc = MinibatchShallowSampler(name_data, dir_data, adjs, node_set, 
            config_sampler_preproc, arch_gnn['feature_augment'], params_train, feat_full, 
            label_full, is_transductive, parallelism, full_tensor_on_gpu, bin_adj_files)
        (   # feat_full now also contains label if using label propagation
            adjs, feat_full, 
            is_adj_changed, is_feat_changed, 
            dim_feat_smooth, dim_label_smooth
         ) = PreprocessGraph(arch_gnn, minibatch_preproc, no_pbar).preprocess()
        dir_data['is_adj_changed'] = is_adj_changed
        dir_data['is_feat_changed'] = is_feat_changed
    else:
        dim_feat_smooth = dim_feat_raw
        dim_label_smooth = 0
    # instantiate minibatch and model for main training loop
    minibatch = MinibatchShallowSampler(name_data, dir_data, adjs, node_set, 
        config_sampler_train, arch_gnn['feature_augment'], params_train, feat_full, 
        label_full, is_transductive, parallelism, full_tensor_on_gpu, bin_adj_files)
    aug_feat = [(k, minibatch.get_aug_dim(k)) for k in arch_gnn['feature_augment']]
    model = DeepGNN(dim_feat_raw, dim_feat_smooth, dim_label_raw, dim_label_smooth, 
                    arch_gnn, aug_feat, minibatch.num_ensemble, params_train).to(device)
    # reload model, if the previous checkpoint is provided as hyperparams
    if 'retrain_dir' in params_train:
        f_model = glob.glob(f"{params_train['retrain_dir']}/saved_model_*")[0]
        f_optm  = glob.glob(f"{params_train['retrain_dir']}/saved_optimizer_*")[0]
        model.load_state_dict(torch.load(f_model))
        model.optimizer.load_state_dict(torch.load(f_optm))
    return model, minibatch


def one_epoch(ep, mode, model, minibatch, logger, status='running', pred_mat=None, emb_ens=None):
    """
    rebuild_pred_mat is for postproc. 
    """
    assert status in ['running', 'final'] and mode in [TRAIN, VALID, TEST]
    minibatch.epoch_start_reset(ep, mode)
    minibatch.shuffle_nodes(mode)
    logger.epoch_start_reset(ep, mode, minibatch.node_set[mode].size)
    t1 = time.time()
    while not minibatch.is_end_epoch(mode):
        input_batch = minibatch.one_batch(mode=mode, ret_raw_idx=(pred_mat is not None or emb_ens is not None))
        if pred_mat is not None or emb_ens is not None:
            idx_pred_raw = input_batch.pop('idx_raw')[0][input_batch['target_ens'][0]]
        output_batch = model.step(mode, status, **input_batch)
        if pred_mat is not None:    # prepare for C&S
            pred_mat[idx_pred_raw] = output_batch['preds']
        if emb_ens is not None:     # prepare for subgraph ensemble
            assert len(emb_ens) == len(output_batch['emb_ens'])
            for ie, e in enumerate(emb_ens):
                e[idx_pred_raw] = output_batch['emb_ens'][ie]
        logger.update_batch(mode, minibatch.batch_num, output_batch)
    t2 = time.time()
    minibatch.epoch_end_reset(mode)
    logger.update_epoch(ep, mode)
    return logger.log_key_step(mode, status=status, time=t2 - t1)


def train(model, minibatch, max_epoch, logger, nocache=None):
    # log running info into CSV file, which has 8 columns for each epoch
    logger.init_log2file(status='running')
    logger.init_log2file(status='final')
    if type(nocache) == str and len(nocache) > 0:     # don't cache the sampled subgraphs to be reused by later epochs
        modes = [TRAIN, VALID, TEST] if nocache == 'all' else [STR2MODE[nocache]]
        for mode in modes:
            minibatch.disable_cache(mode)
    # ---- main training loop ----
    for e in range(max_epoch):
        one_epoch(e, TRAIN, model, minibatch, logger)
        one_epoch(e, VALID, model, minibatch, logger)
        if logger.log_test_convergence > 0 and e % logger.log_test_convergence == 0:
            one_epoch(int(e / logger.log_test_convergence), TEST, model, minibatch, logger)
        logger.update_best_model(e, model, model.optimizer)
    # ---- final testing ----
    logger.printf(("======================\n"
                   "Optimization Finished!\n"
                   "======================\n"), style="red")
    logger.restore_model(model, optimizer=None)
    ep_final_test = 0 if logger.log_test_convergence <= 0 else int(e / logger.log_test_convergence) + 1
    ep_final = {TRAIN: e + 1, VALID: e + 1, TEST: ep_final_test}
    for md in [TRAIN, VALID, TEST]:
        one_epoch(ep_final[md], md, model, minibatch, logger, status='final')
    logger.validate_result()


def inference(model, minibatch, logger):
    logger.init_log2file(status='final')
    logger.load_model(model, optimizer=None, copy=False)    # no need to restore optimizer for pure inference
    for mode in [TRAIN, VALID, TEST]:
        minibatch.disable_cache(mode)       # no need to cache subgraphs since we only make one pass
        one_epoch(0, mode, model, minibatch, logger, status='final')
    

def postprocessing(data_post, model, minibatch, logger, config, acc_record):
    """
    If acc_record is None, then we don't check accuracy. This enables CS for still running jobs. 
    """
    from shaDow.utils import merge_stat_record
    logger.init_log2file(status='final')
    def _common_setup(dmodel):
        logger.set_loader_path(dmodel)
        logger.load_model(model, optimizer=None, copy=False)
        logger.info_batch[TRAIN].PERIOD_LOG = 1
        for md in [TRAIN, VALID, TEST]:
            minibatch.disable_cache(md)
    if config['method'].lower() == 'cs':
        from shaDow.postproc_CnS import correct_smooth
        # NOTE: setting the TRAIN evaluation period to > 1 will only make the 
        #   log / print message "appear" to be nondeterministic. However, the 
        #   full prediction matrix `pred_mat` is always deterministic regardless
        #   of the evaluation frequency. So PERIOD_LOG has no effect on the C&S output. 
        assert acc_record is None or (type(acc_record) == list and len(acc_record) == len(config['dir_pred_mat']))
        # generate and store prediction matrix if not yet available from external file
        for i, dmodel in enumerate(config['dir_pred_mat']):
            if config['pred_mat'][i] is None:
                _common_setup(dmodel)
                if minibatch.name_data not in ['arxiv', 'products']:
                    logger.printf(f"POSTPROC OF CS ONLY DOES NOT SUPPORT {minibatch.name_data} YET")
                    raise NotImplementedError
                pred_mat = torch.zeros(minibatch.label_full.shape).to(config['dev_torch'])
                for md in [TRAIN, VALID, TEST]:
                    one_epoch(0, md, model, minibatch, logger, status='final', pred_mat=pred_mat)
                fname_pred = 'pred_mat_{}.cs' if acc_record is not None else '__pred_mat_{}.cs'
                logger.save_tensor(pred_mat, fname_pred, use_path_loader=True)
                config['pred_mat'][i] = pred_mat
                logger.reset()
        if acc_record is not None:
            acc_record = merge_stat_record(acc_record)
        acc_orig, acc_post = correct_smooth(
                config['name_data'], config['dev_torch'], config['pred_mat'], 
                config['hyperparameter']['norm_sym'], config['hyperparameter']['alpha']
            )
        # double check if acc calulated by C&S matches with the record (i.e., acc_orig & acc_record)
        if acc_record is not None:
            for md in [VALID, TEST]:
                acc_orig_m = [round(a, 4) for a in acc_orig[md]]
                acc_recd_m = [round(a, 4) for a in acc_record['accuracy'][md]]
                assert all(abs(acc_orig_m[i] - acc_recd_m[i]) <= 0.0001 for i in range(len(acc_orig_m))),\
                            "[ACC MISMATCH] MAYBE YOU WANT TO REMOVE THE STORED IN THIS RUN. "
    elif config['method'].lower() == 'ensemble':
        from shaDow.postproc_ens import ensemble_multirun
        assert acc_record is None or (type(acc_record) == dict and len(acc_record) == len(config['dir_emb_mat']))
        # the below 'for' loop is eval / inference only (no need to reset model)
        for sname, dirs_l in config['dir_emb_mat'].items(): # ppr: [,,], khop: [,,]
            for i, dmodel in enumerate(dirs_l):             # [,,]
                if config['emb_mat'][sname][i] is None:     # single model
                    # inference
                    _common_setup(dmodel)
                    N, F = minibatch.feat_full.shape[0], model.dim_hidden
                    emb_mat = [torch.zeros((N, F)).to(config['dev_torch']) for i in range(model.num_ensemble)]
                    for md in [TRAIN, VALID, TEST]:
                        one_epoch(0, md, model, minibatch, logger, status='final', emb_ens=emb_mat)
                    fname_emb = 'emb_mat_{}.ens' if acc_record is not None else '__emb_mat_{}.ens'
                    _fname = logger.save_tensor(emb_mat, fname_emb, use_path_loader=True)
                    config['emb_mat'][sname][i] = emb_mat
                    logger.reset()
        # ensemble and train
        acc_orig, acc_post = ensemble_multirun(data_post['node_set'], config['emb_mat'], data_post['label'], 
                config['architecture'], config['hyperparameter'], logger, config['dev_torch'], acc_record)
    # wrap up
    logger.print_table_postproc(acc_orig, acc_post)
    


def compute_complexity(model, minibatch, num_roots_budget, logger, modes=[VALID], unit='G'):
    from tqdm import tqdm
    logger.printf((f"-.-.-.-.-.-.-.-.-.-.-.\n"
                   f"COMPUTE INFERENCE COST\n"
                   f".-.-.-.-.-.-.-.-.-.-.-\n"), style='blue')
    ops_mode = {}
    for md in modes:
        if num_roots_budget is None or num_roots_budget <= 0:
            num_roots_budget = minibatch.node_set[md].size
        minibatch.disable_cache(md)
        minibatch.epoch_start_reset(0, md)
        minibatch.shuffle_nodes(md)
        num_roots_eval = 0
        ops = []
        pbar = tqdm(total=num_roots_budget)
        while num_roots_eval < num_roots_budget:     # TODO: replace with budget check
            batch = minibatch.one_batch(mode=md, ret_raw_idx=False)
            cur_batch_size = batch['target_ens'][0].size
            num_roots_eval += cur_batch_size
            ops.append(model.calc_complexity_step(batch['adj_ens'], batch['feat_ens'], batch['size_subg_ens']))
            pbar.update(cur_batch_size)
        pbar.close()
        minibatch.epoch_end_reset(md)
        ops_mode[md] = np.array(ops).sum() / num_roots_eval
    logger.printf(f"Average inference cost per node: ", style='blue')
    norm_factor = {"G": 1e9, "M": 1e6}
    for md in modes:
        logger.printf(f"[{MODE2STR[md]:^8s}]    {ops_mode[md] / norm_factor[unit]:.3f}{unit}", style='blue')



""" MAIN FUNCTION
Can handle one of the three tasks: train, inference or postprocessing
For Train:
    We go through the normal training iterations to optimize a randomly initialized shaDow-GNN model.
For Inference:
    We load a pretrained model to make one-time forward pass to the Train / Val / Test nodes. 
For postprocessing:
    We load a pretrained model and apply postprocessing algorithms (e.g., C&S)
"""
def main(task, args):
    assert task in ['train', 'inference', 'postproc']
    dataset = args.dataset
    dir_log = meta_config['logging']['dir']['local']
    os_ = meta_config['device']['software']['os']
    (
        params_train, 
        config_sampler_preproc,
        config_sampler_train, 
        config_data, 
        arch_gnn, 
        dir_log_full
    ) = parse_n_prepare(task, args, dataset, dir_log, os_=os_)
    metric = DATA_METRIC[dataset]
    logger = Logger(task,
                    {"args"         : args, 
                     "arch_gnn"     : arch_gnn, 
                     "data"         : config_data,
                     "hyperparams"  : params_train, 
                     "sampler_preproc": config_sampler_preproc,
                     "sampler_train"  : config_sampler_train}, 
                    dir_log_full, Metrics(dataset, (arch_gnn['loss'] == 'sigmoid'), metric), 
                    no_log=args.no_log, timestamp=timestamp, 
                    log_test_convergence=args.log_test_convergence,
                    period_batch_train=args.eval_train_every, no_pbar=args.no_pbar)
    if task == 'postproc':
        config_postproc, acc_record, skip_instantiate \
            = parse_n_prepare_postproc(
                    args.postproc_dir, 
                    args.postproc_configs, 
                    dataset, dir_log, 
                    arch_gnn, 
                    logger
                )
    else:
        skip_instantiate = []

    # skip_instantiate specifies if we want to skip certain steps in instantiating the model:
    # e.g., For C&S postproc, don't need to load the model if we have already stored the generated embeddings. 
    dir_data = meta_config['data']['dir']
    if 'data' not in skip_instantiate:
        data_train = load_data(dir_data, dataset, config_data, os_=os_)
    else:
        data_train = None
    if 'model' not in skip_instantiate:
        assert 'data' not in skip_instantiate
        model, minibatch = instantiate(
                dataset, dir_data, data_train, 
                params_train, arch_gnn, 
                config_sampler_preproc, config_sampler_train,
                meta_config['device']['cpu']['max_threads'],
                args.full_tensor_on_gpu,
                args.no_pbar
            )
        logger.printf(f"TOTAL NUM OF PARAMS = {sum(p.numel() for p in model.parameters())}", style="yellow")
    else:
        model = minibatch = None
    
    # Now handle the specific tasks
    if task == 'train':
        try:
            nocache = args.nocache if type(args.nocache) != str else args.nocache.lower()
            train(model, minibatch, params_train["end"], logger, nocache=nocache)
            status = 'finished'
        except KeyboardInterrupt:
            status = 'killed'
            print("Pressed CTRL-C! Stopping. ")        
        except Exception as err:
            status = 'crashed'
            import traceback
            traceback.print_tb(err.__traceback__)
        finally:
            # logger will only remove file when you are running the test *.yml
            logger.end_training(status)     # cleanup the unwanted log files
    elif task == 'inference':
        if not args.compute_complexity_only:
            logger.set_loader_path(args.inference_dir)
            inference(model, minibatch, logger)
        else:
            compute_complexity(model, minibatch, args.inference_budget, logger)
    else:       # postprocessing
        config_postproc['dev_torch'] = device
        config_postproc['name_data'] = dataset
        if minibatch is not None:
            data_postproc = {"label": minibatch.label_full, "node_set": minibatch.node_set}
        elif data_train is not None:
            data_postproc = {"label": data_train['label_full'], "node_set": data_train['node_set']}
        else:
            data_postproc = None
        postprocessing(data_postproc, model, minibatch, logger, config_postproc, acc_record)



if __name__ == "__main__":
    if (args_global.inference_dir is None and args_global.inference_configs is None) and args_global.postproc_configs is None:
        task = 'train'
    elif args_global.inference_dir is not None or args_global.inference_configs is not None:
        assert args_global.postproc_dir is None and args_global.postproc_configs is None
        task = 'inference'
    else:
        task = 'postproc'
    str_start = f"PERFORM {task.upper()} TASK"
    print(f"# {'*' * len(str_start)} #")
    print(f"* {str_start} *")
    print(f"# {'*' * len(str_start)} #")

    main(task, args_global)
