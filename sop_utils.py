import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
from scipy import stats
import itertools
import time 
import multiprocessing

def predict_TTest(Z, delta, k):
    Z_mean = Z.mean()
    sigma = Z.std(unbiased=True)
    _sigma = sigma.item()
    # t_stat = stats.t.pdf(1-delta, k-1)
    t_stat = stats.t.ppf(1-delta, k-1)  # this is a constant
    return Z_mean + 2*sigma*t_stat/(k**0.5)

def TTestUpperBound(Z, delta):
    Z_mean = Z.mean()
    m = Z.size()[0]
    sigma = Z.std(unbiased=True)
    _sigma = sigma.item()
    # t_stat = stats.t.pdf(1-delta, m-1)
    t_stat = stats.t.ppf(1-delta, m-1) # this is a constant
    return Z_mean + sigma*t_stat/(m**0.5)  # NOTE: normal is without any multiplicant

class RegionalExpertLoss(nn.Module):
    '''
    Put contraints over all pairs of regions
    '''
    def __init__(self, params, regions, prior_pairs=None) -> None:
        super().__init__()
        self.regions = [str(ai) for ai in sorted(regions)]
        self.params = params
        self.all_region_pairs = [ci for ci in itertools.combinations(self.regions, 2)]

        if prior_pairs is not None:
            assert np.alltrue([[rii in self.regions for rii in ri] for ri in prior_pairs])
            self.constraint_region_pairs = prior_pairs
        else:
            self.constraint_region_pairs = self.all_region_pairs

        print("#pair constrained: ", len(self.constraint_region_pairs))
        # TODO: delta, epsilon can be predefined differently for different pairs
        self.deltas = {f"{r1}_{r2}": params['delta'] for r1, r2 in self.constraint_region_pairs} 
        self.epsilons = {f"{r1}_{r2}": params['eps'] for r1, r2 in self.constraint_region_pairs} 

    def forward(self, preds, ys, batch_regions, constraint_region_pairs=None, return_constrained_pairs=False):
        if not isinstance(batch_regions, np.ndarray):
            batch_regions = np.array(batch_regions)

        if not isinstance(preds, torch.Tensor):
            preds = torch.tensor(preds)
            ys = torch.tensor(ys)
        ## Residual
        errors = preds - ys

        if constraint_region_pairs is None:
            constraint_region_pairs = self.constraint_region_pairs
        
        all_Z = {}

        for i, (r1, r2) in enumerate(constraint_region_pairs):
            # r1_loss
            r1_errs = errors[batch_regions == r1]
            # r2_loss
            r2_errs = errors[batch_regions == r2]
            
            min_len = min(len(r1_errs), len(r2_errs))
            r1_errs, r2_errs = r1_errs[:min_len], r2_errs[:min_len]

            Zis = r1_errs - r2_errs

            rp = f'{r1}_{r2}'

            all_Z[rp] = Zis

        return all_Z

@torch.no_grad()
def check_prob_bound(regional_criterion, preds, ys, batch_regions, logger): 
    '''
        check probabilistic bound
    '''
    print('check_prob_bound()')

    # save prediciton for inspection 
    eval_info = {r: {'predict': p.item(), 'gt': y.item()} 
                for r, p, y in zip(batch_regions, preds, ys)}
    logger.log_value('safetytest/prediction', eval_info)

    # save Zs
    Zs = regional_criterion(preds, ys, batch_regions, constraint_region_pairs=regional_criterion.all_region_pairs)

    Zsmean = {rp: zs.abs().mean().detach().cpu().numpy().item() for rp, zs in Zs.items()}
    logger.log_value('safetytest/Zsmean', Zsmean)

    failed_pairs = []
    all_pairs_info = {}
    
    for idx, rpair in enumerate(regional_criterion.constraint_region_pairs):
        rpair_str = f"{rpair[0]}_{rpair[1]}"

        Z, delta, epsilon = Zs[rpair_str], regional_criterion.deltas[rpair_str], regional_criterion.epsilons[rpair_str]

        upperBound = max(TTestUpperBound(Z, delta/2), TTestUpperBound(-Z, delta/2))

        all_pairs_info[rpair_str] = {'ub': upperBound.item(), 'zmean': Z.abs().mean().item()}

        # if within bounds
        if upperBound > epsilon:
            failed_pairs.append(rpair_str)

    # import pdb;pdb.set_trace()
    logger.log_value('safetytest/all_pairs', all_pairs_info)

    if len(failed_pairs) > 0:
        print(f'Failed at {len(failed_pairs)} pairs of regions (eps={epsilon}), (e.g. {failed_pairs[0]}: {all_pairs_info[failed_pairs[0]]})')
        logger.log_value('safetytest/fail', failed_pairs)

        return False

    return True

@torch.no_grad()
def test_model(regional_criterion, preds, ys, batch_regions, logger): 
    print("testing model ...")
    # save prediciton for inspection 
    eval_info = {r: {'predict': p.item(), 'gt': y.item()} 
                for r, p, y in zip(batch_regions, preds, ys)}
    logger.log_value('test/prediction', eval_info)

    # save Zs
    Zs = regional_criterion(preds, ys, batch_regions, 
            constraint_region_pairs=regional_criterion.all_region_pairs)

    Zsmean = {rp: zs.abs().mean().detach().cpu().numpy().item() for rp, zs in Zs.items()}
    logger.log_value('test/Zsmean', Zsmean)

    failed_pairs = []
    all_pairs_info = {}
    
    for idx, rpair in enumerate(regional_criterion.constraint_region_pairs):
        rpair_str = f"{rpair[0]}_{rpair[1]}"

        Z, delta, epsilon = Zs[rpair_str], regional_criterion.deltas[rpair_str], regional_criterion.epsilons[rpair_str]

        upperBound = max(TTestUpperBound(Z, delta/2), TTestUpperBound(-Z, delta/2))

        all_pairs_info[rpair_str] = {'ub': upperBound.item(), 'zmean': Z.abs().mean().item()}

        # if within bounds
        if upperBound > epsilon:
            failed_pairs.append(rpair_str)

    # import pdb;pdb.set_trace()
    logger.log_value('test/all_pairs', all_pairs_info)

    failure_rate = len(failed_pairs) / len(regional_criterion.constraint_region_pairs)
    logger.log_value('test/failure_rate', failure_rate, flush=True)
    print(f'failure rate: {failure_rate:.2f}')
    return failure_rate

