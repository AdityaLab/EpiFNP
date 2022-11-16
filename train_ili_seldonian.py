import json
import os
from optparse import OptionParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_utils import *
from model_utils import *

from models.utils import float_tensor
from sop_utils import * 
from ast import literal_eval
from copy import deepcopy

import global_config


def set_seed(manualseed):
    torch.manual_seed(manualseed)
    np.random.seed(manualseed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(manualseed)
        torch.cuda.manual_seed_all(manualseed)
        torch.backends.cudnn.enabled = False 
        # True ensures the algorithm selected by CUFA is deterministic
        torch.backends.cudnn.deterministic = True
        # torch.set_deterministic(True)
        # False ensures CUDA select the same algorithm each time the application is run
        torch.backends.cudnn.benchmark = False


class Logger:
    def __init__(self, filepath) -> None:
        self.filepath = filepath

        self.data = {}
        self.savelag = 10
        self.prior_prefix = None
    
    def prefix_subsequence_log(self, indexstr):
        indexarr = indexstr.split("/")
        self.prior_prefix = indexarr

    def log_value(self, indexstr, value, epoch=None, use_prefix=True, flush=False):
        indexarr = indexstr.split("/")
        if self.prior_prefix is not None and use_prefix:
            indexarr = self.prior_prefix + indexarr

        ref = self.data
        pref = self.data
        for idx in indexarr:
            if idx not in ref:
                ref[idx] = {}
                pref = ref
                ref = ref[idx]
            else:
                pref = ref
                ref = ref[idx]

        if epoch is not None:
            ref[epoch] = value
        else:
            pref[idx] = value
            epoch = -1

        if not (epoch % self.savelag) or flush:
            with open(self.filepath, 'w') as f:
                json.dump(self.data, f, indent = 4)
    
    def close(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.data, f, indent = 4)

        
class SeldonianHooker(nn.Module):
    def __init__(self, reg_loss_func, size_Ds, params, logger, barrier_value=10) -> None:
        super().__init__()

        self.reg_loss_func = reg_loss_func
        self.size_Ds = size_Ds
        self.params = params
        self.logger = logger
        self.barrier_value = barrier_value

    def forward(self, model_loss, preds, y, batch_regions, **kwargs):
        '''
        Finding candidate loss function
        '''
        reg_loss_func, size_Ds, params = self.reg_loss_func, self.size_Ds, self.params
        seldonian_loss = model_loss
        is_reset = False # reset model_loss with barier value
        Zs = reg_loss_func(preds, y, batch_regions)
        ups = {}
        cnt_failed = 0
        for idx, rpair in enumerate(Zs.keys()):
            Z, delta, epsilon = Zs[rpair], reg_loss_func.deltas[rpair], reg_loss_func.epsilons[rpair]
            upperBound = max(predict_TTest(Z, delta/2,size_Ds), predict_TTest(-Z, delta/2,size_Ds))
            ups[rpair] = upperBound.item()
            if upperBound <= epsilon and not is_reset: # so far so good
                seldonian_loss += params['lamda'] * Z.abs().mean() / len(Zs) # LAMBDA * ((Z**2)**0.5).mean() 
            else: # upperBound > epsilon OR is_reset == True
                if not is_reset: # first time the upperbound > epslion
                    seldonian_loss = self.barrier_value
                    is_reset = True
                if upperBound > epsilon: # don't care about cases with upperbound <= epsilon
                    cnt_failed += 1
                    seldonian_loss += upperBound + (params['lamda'] - 1)*epsilon / len(Zs)
        
        self.logger.log_value('train/upperbound', ups, kwargs['ep'])
        self.logger.log_value('train/failed', cnt_failed, kwargs['ep'])
        return seldonian_loss


def plot_train_info(runtimeid, losses, errors, train_errors, variances, savedir): 
    plt.figure(1)
    plt.plot(losses)
    plt.savefig(f"plots/{savedir}/losses{runtimeid}.png")
    plt.figure(2)
    plt.plot(errors)
    plt.plot(train_errors)
    plt.savefig(f"plots/{savedir}/errors{runtimeid}.png")
    plt.figure(3)
    plt.plot(variances)
    plt.savefig(f"plots/{savedir}/vars{runtimeid}.png")

def evaluate_and_plot(runtimeid, eval_params, savedir):
    e, yp, yt, vars, fem, tem, _ = evaluate(*eval_params, sample=True)
    yp = np.array([evaluate(*eval_params, sample=True)[1] for _ in range(n_eval_test)])
    yp, vars = np.mean(yp, 0), np.var(yp, 0)
    e = np.mean((yp - yt) ** 2)
    dev = np.sqrt(vars) * 1.95
    plt.figure(4)
    plt.plot(yp, label="Predicted 95%", color="blue")
    plt.fill_between(np.arange(len(yp)), yp + dev, yp - dev, color="blue", alpha=0.2)
    plt.plot(yt, label="True Value", color="green")
    plt.legend()
    plt.title(f"MSE: {e}")
    plt.savefig(f"plots/{savedir}/Test{runtimeid}.png")
    dt = {
        "mse": e,
        "target": yt,
        "pred": yp,
        "vars": vars,
        "fem": fem,
        "tem": tem,
    }
    save_data(dt, f"./saves/{savedir}/{runtimeid}_test.pkl")

    e, yp, yt, vars, _, _, _ = evaluate(*eval_params, sample=True, dtype="val")
    yp = np.array([evaluate(*eval_params, sample=True, dtype="val")[1] for _ in range(n_eval_test)])
    yp, vars = np.mean(yp, 0), np.var(yp, 0)
    e = np.mean((yp - yt) ** 2)
    dev = np.sqrt(vars) * 1.95
    plt.figure(5)
    plt.plot(yp, label="Predicted 95%", color="blue")
    plt.fill_between(np.arange(len(yp)), yp + dev, yp - dev, color="blue", alpha=0.2)
    plt.plot(yt, label="True Value", color="green")
    plt.legend()
    plt.title(f"RMSE: {e}")
    plt.savefig(f"plots/{savedir}/Val{runtimeid}.png")

    e, yp, yt, vars, fem, tem , _= evaluate(*eval_params, sample=True, dtype="train")
    yp = np.array([evaluate(*eval_params, sample=True, dtype="train")[1] for _ in range(n_eval_train)])
    yp, vars = np.mean(yp, 0), np.var(yp, 0)
    e = np.mean((yp - yt) ** 2)
    dev = np.sqrt(vars) * 1.95
    plt.figure(6)
    plt.plot(yp, label="Predicted 95%", color="blue")
    plt.fill_between(np.arange(len(yp)), yp + dev, yp - dev, color="blue", alpha=0.2)
    plt.plot(yt, label="True Value", color="green")
    plt.legend()
    plt.title(f"RMSE: {e}")
    plt.savefig(f"plots/{savedir}/Train{runtimeid}.png")
    dt = {
        "rmse": e,
        "target": yt,
        "pred": yp,
        "vars": vars,
        "fem": fem,
        "tem": tem,
    }
    save_data(dt, f"./saves/{savedir}/{runtimeid}_train.pkl")


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-m", "--model_name", dest="model_name", type="string")
    parser.add_option("-d", "--dev", dest="dev", type="string", default="cpu")
    parser.add_option("-y", "--year", dest="testyear", type="int")
    parser.add_option("-w", "--week", dest="week_ahead", type="int")
    parser.add_option("-a", "--atten", dest="atten", type="string")
    parser.add_option("-e", "--epoch", dest="epochs", type="int")
    parser.add_option("-s", "--seed", dest="seed", type="int", default=10)
    parser.add_option("--n_eval_train", dest="n_eval_train", type="int", default=40)
    parser.add_option("--n_eval_test", dest="n_eval_test", type="int", default=50)
    parser.add_option("-p", "--constraint_pairs", dest="constraint_pairs", type="string", default=None)
    parser.add_option("--combine", dest="combine", action='store_true', default=False)
    parser.add_option("--lamda", dest="lamda", type="float", default=1.0)
    parser.add_option("--eps", dest="eps", type="float", default=0.5)
    parser.add_option("--delta", dest="delta", type="float", default=0.05)
    parser.add_option("--savedir", dest="savedir", type="string", default=None)
    parser.add_option("--chkp_pattern", dest="chkp_pattern", type="string", default=None)
    parser.add_option("--test", dest="test", action="store_true", default=False)

    (options, args) = parser.parse_args()

    set_seed(options.seed)

    device = 'cpu' if options.dev=='cpu' else torch.device(f'cuda:{options.dev}')
    global_config.device = device

    train_seasons = list(range(2003, options.testyear))
    # TODO change validation season
    val_seasons = [train_seasons[-1]]
    train_seasons = train_seasons[:-1]
    test_seasons = [options.testyear]
    # train_seasons = list(range(2003, 2019))
    # test_seasons = [2019]
    print(train_seasons, val_seasons, test_seasons)

    # train_seasons = [2003, 2004, 2005, 2006, 2007, 2008, 2009]
    # test_seasons = [2010]
    # regions = ["X"]
    regions = ["X"] + [f"Region {i}" for i in range(1,11)]
    city_idx = {f"Region {i}": i for i in range(1, 11)}
    city_idx["X"] = 0

    week_ahead = options.week_ahead
    val_frac = 5
    attn = options.atten
    # model_num = options.num
    n_eval_train = options.n_eval_train # TODO
    n_eval_test = options.n_eval_test # TODO 
    model_name = options.model_name
    # model_num = 22
    EPOCHS = options.epochs
    print(week_ahead, attn, EPOCHS)

    ## Logging
    log_savedir = Path('logs/')
    if options.savedir is not None:
        log_savedir = log_savedir/options.savedir
    log_savedir.mkdir(exist_ok=True, parents=True)

    prior_pairs = literal_eval(str(options.constraint_pairs)) # [('X', 'FL')]

    adjustable_params_to_filename = {
        'numConstraint': 'All' if prior_pairs is None else len(prior_pairs),
        'combine': options.combine,
        'testyear': options.testyear,
        'weekahead': options.week_ahead,
        'eps': options.eps, 
        'delta': options.delta, 
        'lambda': options.lamda,
        'epoch': options.epochs
    }

    fname_param_str = "_".join([f"{k}{v}" for k, v in adjustable_params_to_filename.items()])
    runtimeid = f'{options.model_name}_{fname_param_str}_seed{options.seed}'

    logfilepath = str(log_savedir / f"{runtimeid}.json")
    print(f'logging to {logfilepath}')
    logger = Logger(logfilepath)

    for d in ["model_chkp", "plots", "saves"]:
        Path(f"{d}/{options.savedir}").mkdir(exist_ok=True, parents=True)

    ## Prepare data
    train_info, val_info, test_info = \
        extract_data(train_seasons, val_seasons, test_seasons, regions, city_idx)
    full_x, full_y, full_meta = train_info 
    full_x_val, full_y_val, full_meta_val = val_info
    full_x_test, full_y_test, full_meta_test = test_info

    train_meta, train_x, train_y = \
        create_dataset(full_meta, full_x, week_ahead=week_ahead)

    val_meta, val_x, val_y = \
        create_dataset(full_meta_val, full_x_val, week_ahead=week_ahead)

    test_meta, test_x, test_y = \
        create_dataset(full_meta_test, full_x_test, week_ahead=week_ahead)

    emb_model, emb_model_full, fnp_model = \
        create_model(attn, city_idx, train_meta, device)

    optimizer = optim.Adam(
        list(emb_model.parameters())
        + list(fnp_model.parameters())
        + list(emb_model_full.parameters()),
        lr=1e-3,
    )

    # emb_model_full = emb_model

    train_meta, train_x, train_y, train_lens = \
        create_tensors(train_meta, train_x, train_y, device)

    val_meta, val_x, val_y, val_lens = \
        create_tensors(val_meta, val_x, val_y, device)
    
    test_meta, test_x, test_y, test_lens = \
        create_tensors(test_meta, test_x, test_y, device)

    # full_x_chunks = np.zeros((full_x.shape[0] * 4, full_x.shape[1], full_x.shape[2]))
    # full_meta_chunks = np.zeros((full_meta.shape[0] * 4, full_meta.shape[1]))
    # for i, s in enumerate(full_x):
    #     full_x_chunks[i * 4, -20:] = s[:20]
    #     full_x_chunks[i * 4 + 1, -30:] = s[:30]
    #     full_x_chunks[i * 4 + 2, -40:] = s[:40]
    #     full_x_chunks[i * 4 + 3, :] = s
    #     full_meta_chunks[i * 4 : i * 4 + 4] = full_meta[i]

    full_x = float_tensor(full_x)
    full_meta = float_tensor(full_meta)
    full_y = float_tensor(full_y)

    train_mask, val_mask, test_mask = (
        create_mask(train_lens, device),
        create_mask(val_lens, device),
        create_mask(test_lens, device),
    )

    ## Train-val split
    # TODO: remove permuation
    # perm = np.random.permutation(train_meta_.shape[0])
    # val_perm = perm[: train_meta_.shape[0] // val_frac]
    # train_perm = perm[train_meta_.shape[0] // val_frac :]
    
    # train_meta, train_x, train_y, train_lens, train_mask = (
    #     train_meta_[train_perm],
    #     train_x_[train_perm],
    #     train_y_[train_perm],
    #     train_lens_[train_perm],
    #     train_mask_[:, train_perm, :],
    # )
    # val_meta, val_x, val_y, val_lens, val_mask = (
    #     train_meta_[val_perm],
    #     train_x_[val_perm],
    #     train_y_[val_perm],
    #     train_lens_[val_perm],
    #     train_mask_[:, val_perm, :],
    # )

    params = dict(
        lamda = options.lamda, 
        eps = options.eps, 
        delta = options.delta
    )
    save_params = deepcopy(options.__dict__)
    save_params.update(params)
    logger.log_value("params", save_params)

    regional_criterion = RegionalExpertLoss(params, city_idx.values(),
                                    prior_pairs=prior_pairs)
                                    
    if options.chkp_pattern is not None: 
        # load model for testing or fine-tuning
        print(f"loading pretrained model from {options.chkp_pattern}")
        chkp_prefix = options.chkp_pattern.split('*')[0]
        load_model(emb_model, emb_model_full, fnp_model, chkp_prefix)


    if not options.test:

        size_Ds = val_x.shape[0]
        seldonian_hooker = SeldonianHooker(regional_criterion, size_Ds, params, logger)

        if 'SOP' in model_name.upper():
            before_backward_hooker = seldonian_hooker
        else:
            before_backward_hooker = None

        error, errors, losses, \
        train_errors, variances = \
            train_model(emb_model, emb_model_full, fnp_model, optimizer, 
                        full_meta, full_x, full_y, 
                        train_meta, train_x, train_y, train_mask, 
                        val_meta, val_x, val_y, val_mask, 
                        test_meta, test_x, test_y, test_mask,
                        EPOCHS, runtimeid, n_eval_train, 
                        regional_criterion, before_backward_hooker, logger, savedir=options.savedir)

        plot_train_info(runtimeid, losses, errors, train_errors, variances, options.savedir)


    """
        Safety test in Ds
    """
    eval_params = (emb_model, emb_model_full, fnp_model, 
            full_meta, full_x, full_y, 
            train_meta, train_x, train_y, train_mask,
            val_meta, val_x, val_y, val_mask, 
            test_meta, test_x, test_y, test_mask)

    yps = []
    for _ in range(n_eval_test): 
        _, yp, ys, _, _, _, batch_regions = evaluate(*eval_params, sample=True, dtype='val')
        yps.append(yp) 
    yps = np.array(yps)
    yp = np.mean(yps, 0)

    rmse = np.sqrt(np.mean((yp - ys) ** 2))
    logger.log_value('safetytest/rmse', rmse.item())
    if not check_prob_bound(regional_criterion, yp, ys, batch_regions, logger):
        # raise Exception('NSF')
        NSF = True
        print('>>>NSF')
    else:
        print('>>>prob bound OK')

    """
        Evaluation
    """
    yps = []
    for _ in range(n_eval_test): 
        _, yp, ys, _, _, _, batch_regions = evaluate(*eval_params, sample=True, dtype='test')
        yps.append(yp) 
    yps = np.array(yps)
    yp = np.mean(yps, 0)

    rmse = np.sqrt(np.mean((yp - ys) ** 2))
    logger.log_value('test/rmse', rmse.item())

    test_model(regional_criterion, yp, ys, batch_regions, logger)

    # plots
    evaluate_and_plot(runtimeid, eval_params, options.savedir)

    # Finish 
    logger.close()
    print(f'logged to {logfilepath}')
