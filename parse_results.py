import numpy as np
import json
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
from glob import glob

def isiter(obj):
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True

def get_data(data, jsonpath, prefix=None, epoch=None):
    indexarr = jsonpath.split('/')

    root = data 
    if prefix is not None:
        prefixarr = prefix.split('/')
        indexarr = prefixarr + indexarr

    for i, index in enumerate(indexarr):
        if index not in root:
            print(f"{indexarr[:i+1]} not found!")
            return None
        else: 
            root = root[index]

    if (epoch is not None) and (epoch in root):
        return root[epoch]
    return root


def is_rejected(data, prefix, eps, fail_threshold=0.0):
    all_pairs = get_data(data, 'safetytest/all_pairs', prefix=prefix)
    ubs = [ai['ub'] for ai in all_pairs.values()]
    failed = [1 if ubi > eps else 0 for ubi in ubs]
    if np.mean(failed) > fail_threshold:
        return True
    else:
        return False

def get_model_name(model_logfile_temp):
    if isinstance(model_logfile_temp, tuple):
        model_logfile_temp, modelname = model_logfile_temp
    else:
        modelname = '_'.join(Path(model_logfile_temp).stem.split("_")[:2])
        if 'rnn' in modelname:
            modelname = 'rnn'
    return model_logfile_temp, modelname

def plot_rmse_avg_over_seeds(models_logfile_template, save_imgpath_postfix=None, 
    pred_epiweeks=['202210', '202214', '202218', '202222', '202226', '202230'], 
    seeds=[1234, 1235, 1236, 1237, 1238], outdir='figures', 
    verbose=False, predweeks_pairs=None, trial=None, plot_eps=None):
    
    ew_res = {}
    model_names = []
    for ew in pred_epiweeks:
        ew_res[ew] = {}

        for model_logfile_temp in models_logfile_template:
            model_logfile_temp, modelname = get_model_name(model_logfile_temp)
            if modelname not in model_names:
                model_names.append(modelname)

            # if 'rnn' in modelname.lower():
            #     prefix=None
            # else:
            prefix = trial
            model_rmse = {}
            for seed in seeds:
                logfile = model_logfile_temp.format(ew, seed)
                print(logfile)
                data = json.load(open(logfile))

                # model is reject?
                # if get_data(data, 'safetytest/fail', prefix=prefix) is not None: 
                #     reject = True
                # else:
                #     reject = False
                eps = plot_eps if plot_eps is not None else data['params']['eps']

                if predweeks_pairs is None:
                    print('show contraint all regions...')
                    # constraint_regions = data['test']['prediction'].keys()

                    constraint_regions = get_data(data, 'test/prediction', prefix=prefix).keys()
                    
                else:
                    constraint_regions_pairs = predweeks_pairs[ew]
                    print(f'contraint {len(constraint_regions_pairs)} pairs of regions...')
                    constraint_regions = set()
                    for pair in constraint_regions_pairs:
                        constraint_regions.update(pair.split('_'))

                # model rmse
                test_preds = get_data(data, 'test/prediction', prefix=prefix)

                rmse = np.mean([(test_preds[region_pair]['predict'] - test_preds[region_pair]['gt'])**2 
                                for region_pair in constraint_regions])

                rmse = np.sqrt(rmse)
                model_rmse[seed] = {'value': rmse, 'reject': is_rejected(data, prefix, eps, fail_threshold=0)}

                # model params
                if 'params' in data and 'sop' in modelname.lower():
                    paramsstr = f"$\delta={data['params']['delta']}, \epsilon={data['params']['eps']}, \lambda={data['params']['lamda']}$"

            ew_res[ew][modelname] = {}
            ew_res[ew][modelname]['seeds_infos'] = model_rmse
            ew_res[ew][modelname]['value'] = np.mean([seedinfo['value'] for seed, seedinfo in model_rmse.items()])
            ew_res[ew][modelname]['reject'] = np.any([seedinfo['reject'] for seed, seedinfo in model_rmse.items()])

    if verbose: print(ew_res)

    labels = [ew for ew in pred_epiweeks]
    data_df = {}

    for i, mn in enumerate(model_names):
        model_res = [ew_res[ew][mn]['value'] for ew in pred_epiweeks]
        data_df[mn] = model_res

    df = pd.DataFrame(data_df, index=labels)

    ax = df.plot.bar(rot=0, ylabel="RMSE", xlabel='Week', title=f'Params: {paramsstr}')
    # add reject status
    for idx, ew in enumerate(pred_epiweeks):
        rejected = []
        ann_y_pos = 0
        for model_idx, mn in enumerate(model_names):
            if ew_res[ew][mn]['reject']:
                rejected.append(str(model_idx))
                ann_y_pos = max(ann_y_pos, ew_res[ew][mn]['value'])

        ax.annotate(
                f'{", ".join(rejected)}',                      # Use `label` as label
                (idx, ann_y_pos),         # Place label at end of the bar
                xytext=(0, 10),          # Vertically shift label by `space`
                textcoords="offset points", # Interpret `xytext` as offset in points
                ha='center',                # Horizontally center label
                va='bottom',                # Vertically align label differently for 
                                            # positive and negative values.
                color='r')          

    Path(outdir).mkdir(parents=True, exist_ok=True)
    save_imgpath = Path(outdir) / f"report_rmse_{save_imgpath_postfix}.png"
    plt.savefig(save_imgpath)
    print(f'saved to {save_imgpath}')
    plt.close()


def plot_failure_rate_avg_over_seeds(models_logfile_template, 
        save_imgpath_postfix=None, 
        pred_epiweeks=['202210', '202214', '202218', '202222', '202226', '202230'], 
        seeds=[1234, 1235, 1236, 1237, 1238], 
        outdir='figures', verbose=False, 
        predweeks_pairs=None, trial=None, plot_eps = None):

    ew_res = {}
    model_names = []

    for ew in pred_epiweeks:
        ew_res[ew] = {}
        for model_logfile_temp in models_logfile_template:
            model_logfile_temp, modelname = get_model_name(model_logfile_temp)
                
            if modelname not in model_names:
                model_names.append(modelname)

            # if 'rnn' in modelname.lower():
            #     prefix=None
            # else:
            prefix = trial

            model_frs = {}
            for seed in seeds:
                logfile = model_logfile_temp.format(ew, seed)
                print(logfile)
                data = json.load(open(logfile))
                # model is reject?
                # if get_data(data, 'safetytest/fail', prefix=prefix) is not None:
                #     reject = True
                # else:
                #     reject = False
                # interested value
                if predweeks_pairs is None:
                    print('contraint all regions...')
                    # fr = get_data(data, 'test/failure_rate', prefix=prefix)
                    
                    zmeans_data = get_data(data, 'test/Zsmean', prefix=prefix)
                    if zmeans_data is None:
                        zmeans_data = get_data(data, 'test/Zmean', prefix=prefix)

                    constraint_regions = zmeans_data.keys()
                    constraint_regions_zmean_values = [zmeans_data[rp] for rp in constraint_regions]
                    eps = plot_eps if plot_eps is not None else data['params']['eps']
                    fr = np.mean([ai > eps for ai in constraint_regions_zmean_values])
                else:
                    constraint_regions = predweeks_pairs[ew]
                    print(f'contraint {len(constraint_regions)} pairs of regions...')
                    constraint_regions_zmean_values = [get_data(data, 'test/Zsmean', prefix=prefix)[rp] for rp in constraint_regions]
                    eps = plot_eps if plot_eps is not None else data['params']['eps']
                    fr = np.mean([ai > eps for ai in constraint_regions_zmean_values])

                model_frs[seed] = {'value': fr, 'reject': is_rejected(data, prefix, eps, fail_threshold=0)}

                # model params
                if 'params' in data and 'sop' in modelname.lower():
                    paramsstr = f"$\delta={data['params']['delta']}, \epsilon={data['params']['eps']}, \lambda={data['params']['lamda']}$"

            ew_res[ew][modelname] = {}
            ew_res[ew][modelname]['seeds_infos'] = model_frs
            ew_res[ew][modelname]['value'] = np.mean([seedinfo['value'] for seed, seedinfo in model_frs.items()])
            ew_res[ew][modelname]['reject'] = np.mean([seedinfo['reject'] for seed, seedinfo in model_frs.items()])

    if verbose: print(ew_res)
    labels = [ew for ew in pred_epiweeks]
    data_df = {}

    for i, mn in enumerate(model_names):
        model_res = [ew_res[ew][mn]['value'] for ew in pred_epiweeks]
        data_df[mn] = model_res

    df = pd.DataFrame(data_df, index=labels)

    ax = df.plot.bar(rot=0, ylabel="Failure rate", xlabel='Week', title=f'Params: {paramsstr}')
    # add reject status
    for idx, ew in enumerate(pred_epiweeks):
        rejected = []
        ann_y_pos = 0
        for model_idx, mn in enumerate(model_names):
            # if ew_res[ew][mn]['reject']:
            rj = ew_res[ew][mn]['reject']
            rejected.append(f"{rj:.2f}")
            ann_y_pos = max(ann_y_pos, ew_res[ew][mn]['value'])

        ax.annotate(
                f'{", ".join(rejected)}',                      # Use `label` as label
                (idx, ann_y_pos),         # Place label at end of the bar
                xytext=(0, 10),          # Vertically shift label by `space`
                textcoords="offset points", # Interpret `xytext` as offset in points
                ha='center',                # Horizontally center label
                va='bottom',                # Vertically align label differently for 
                                            # positive and negative values.
                color='r')   

    save_imgpath = Path(outdir) / f"report_failure_rate_{save_imgpath_postfix}.png"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_imgpath)
    print(f'saved to {save_imgpath}')
    plt.close()



def plot_Z_avg_seeds_with_lambda(models_logfile_template, save_imgpath_postfix=None,
        pred_week='201920',
        seeds=[1234, 1235, 1236, 1237, 1238], 
        lambdas = [0, 0.25, 0.5, 0.75, 1.0],
        outdir='figures', avg_over_weeks=True, verbose=False, 
        phase='test', find_most_diff_pairs=False,
        constraint_regions=None, trial=None):

    from epiweeks import Week
    pred_week_epiw = Week.fromstring(pred_week)
    ## copy from train_ili_seldonian.py
    regions_ = ["X"] + [f"Region {i}" for i in range(1,11)]
    city_idx = {f"Region {i}": i for i in range(1, 11)}
    city_idx["X"] = 0
    regions = [str(v) for v in city_idx.values()]
    # regions, _, _, _, ys_scalers = preprocessing({'disease': 'COVID', 'pred_week': pred_week_epiw, 'temporal': 'weekly', 'device': 'cpu'})

    model_name = None
    zmeans_all_lambdas = []
    abs_err_all_lamdas = []
    rel_err_all_lamdas = []
    # zmeans_all_lambdas_safetytest_failed = []
    paramsstr = ""
    for _lambda in lambdas:
        zmeans_seeds = []
        zmeans_safe_seeds = []
        abs_err_seeds = []
        rel_err_seeds = []
        
        for seed in seeds:
            assert models_logfile_template.count("{}") == 2
            logfile = models_logfile_template.format(_lambda, seed)
            print(logfile)
            data = json.load(open(logfile))

            model_name = data['params']['model_name']
            if 'sop' in model_name.lower():
                paramsstr = f"$\delta={data['params']['delta']}, \epsilon={data['params']['eps']}, \lambda={data['params']['lamda']}$"

            # model is reject?
            reject = get_data(data, 'safetytest/fail', prefix=trial)

            # zmeans_safe = data['safetytest']['fail']
            # zmeans_safe_seeds.append(np.mean([v for v in zmeans_safe.values()]))

            if phase == 'test':
                # interested value
                zmeans = get_data(data, f"{phase}/Zsmean", prefix=trial)
                if zmeans is None:
                    zmeans = get_data(data, f"{phase}/Zmean", prefix=trial)

                zmeans_seeds.append(zmeans)
                region_pairs = zmeans.keys()
                # error
                pred_info = get_data(data, f"{phase}/prediction", prefix=trial)
            elif phase == 'train':
                # interested value
                zmeans = get_data(data, f"{phase}/Zsmean", prefix=trial)
                zmeans_seeds.append(zmeans)
                region_pairs = zmeans.keys()
                # error
                pred_info = get_data(data, f"{phase}/prediction", prefix=trial)
                pred_info = {r: {'predict': ai[0], 'gt': ai[1]} for r, ai in pred_info.items()}
            elif phase == 'safetytest':
                zmeans = get_data(data, f"{phase}/Zsmean", prefix=trial)
                zmeans_seeds.append(zmeans)
                region_pairs = zmeans.keys()
                # error
                pred_info = get_data(data, f"{phase}/prediction", prefix=trial)

            # predictions = {region: ys_scalers[region].inverse_transform(np.array(info['predict']).reshape(-1,1)).item() for region, info in pred_info.items()}
            # gt = {region: ys_scalers[region].inverse_transform(np.array(info['gt']).reshape(-1,1)).item() for region, info in pred_info.items()}
            import pdb;pdb.set_trace()
            predictions = {region: info['predict'] for region, info in pred_info.items()}
            gt = {region: info['gt'] for region, info in pred_info.items()}
            abs_err = {region: np.abs(predictions[region] - gt[region]) 
                                        for region in predictions.keys()}
            rel_err = {region: np.abs((predictions[region] - gt[region])/(1+gt[region])) 
                                                        for region in predictions.keys()}
            abs_err_seeds.append(abs_err)
            rel_err_seeds.append(rel_err)

        if constraint_regions is not None: 
            region_pairs = constraint_regions

        zmeans_avg_seeds = {region: np.mean([ai[region] for ai in zmeans_seeds]) for region in region_pairs}
        # zmeans_safe_avg_seeds =  np.mean(zmeans_safe_seeds)
        if phase in ['train', 'safetytest', 'test']:
            abs_err_avg_seeds = {region: np.mean([ai[region] for ai in abs_err_seeds]) for region in regions}
            rel_err_avg_seeds = {region: np.mean([ai[region] for ai in rel_err_seeds]) for region in regions}

        if find_most_diff_pairs:
            most_diff_region_pairs = sorted(zmeans_avg_seeds.items(), key=lambda x: x[1])[-10:]
            print(most_diff_region_pairs)

            most_diff_region_pairs = [ai[0] for ai in most_diff_region_pairs]
        
            print(f"most_diff_region_pairs lambdas={_lambda} in {phase}: \n", most_diff_region_pairs)

        zmeans_all_lambdas.append(zmeans_avg_seeds)
        # zmeans_all_lambdas_safetytest_failed.append(zmeans_safe_avg_seeds)
        if phase in ['train', 'safetytest', 'test']:
            abs_err_all_lamdas.append(abs_err_avg_seeds)
            rel_err_all_lamdas.append(rel_err_avg_seeds)

    # plot z_mean
    zmeans_all_lambdas_avg_regions = [np.mean([v for v in ai.values()]) for ai in zmeans_all_lambdas]
    fig, ax = plt.subplots()
    ax.set_title(f'{model_name}-{phase} Zmeans {paramsstr}, avg all pair of regions')
    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('avg zmeans')

    ax.plot(lambdas, zmeans_all_lambdas_avg_regions, marker='.')

    Path(outdir).mkdir(exist_ok=True, parents=True)
    save_imgpath = Path(outdir) / f"zmeans_{pred_week}_{save_imgpath_postfix}_{model_name}_{phase}.png"
    fig.savefig(save_imgpath)
    print(f'saved to {save_imgpath}')
    plt.close(fig)

    # plot z_mean safetytest failed
    # fig, ax = plt.subplots()
    # ax.set_title(f'Zmeans avg all failed pair of regions, avg over {len(seeds)} seeds')
    # ax.set_xlabel('$\lambda$')
    # ax.set_ylabel('avg zmeans failed')

    # ax.plot(lambdas, zmeans_all_lambdas_safetytest_failed)

    # save_imgpath = Path(outdir) / f"zmeans_safetytest_failed_{save_imgpath_postfix}.png"
    # fig.savefig(save_imgpath)
    # print(f'saved to {save_imgpath}')

    if phase in ['train', 'safetytest', 'test']:
        # plot rel_err
        rel_err_all_lamdas_avg_regions = [np.mean([v for v in ai.values()]) for ai in rel_err_all_lamdas]
        fig, ax = plt.subplots()
        ax.set_title(f'{model_name}-{phase} Relative errors {paramsstr}, avg all pair of regions')
        ax.set_xlabel('$\lambda$')
        ax.set_ylabel('avg relative error')
        ax.plot(lambdas, rel_err_all_lamdas_avg_regions, marker='.')

        save_imgpath = Path(outdir) / f"relative_err_{pred_week}_{save_imgpath_postfix}_{model_name}_{phase}.png"
        fig.savefig(save_imgpath)
        print(f'saved to {save_imgpath}')
        plt.close(fig)

        # plot abs_err
        abs_err_all_lamdas_avg_regions = [np.mean([v for v in ai.values()]) for ai in abs_err_all_lamdas]
        fig, ax = plt.subplots()
        ax.set_title(f'{model_name}-{phase} Absolute errors {paramsstr}, avg all pair of regions')
        ax.set_xlabel('$\lambda$')
        ax.set_ylabel('avg absolute error')
        ax.plot(lambdas, abs_err_all_lamdas_avg_regions, marker='.')

        save_imgpath = Path(outdir) / f"absolute_err_{pred_week}_{save_imgpath_postfix}_{model_name}_{phase}.png"
        fig.savefig(save_imgpath)
        plt.close(fig)
        print(f'saved to {save_imgpath}')


def check_rerun():
            
    # [('7_8', 0.5421825051307678), ('6_9', 0.5495976805686951), ('5_8', 0.5820356607437134), ('4_8', 0.590314507484436), ('1_8', 0.6003063917160034), ('6_10', 0.6056899428367615), ('3_8', 0.6564177870750427), ('8_9', 0.6701736450195312), ('2_8', 0.6801490783691406), ('8_10', 0.717146098613739)]
    prior_pairs = ['7_8', '6_9', '5_8', '4_8', '1_8', '6_10', '3_8', '8_9', '2_8', '8_10']
    # [['7', '8'], ['6', '9'], ['5', '8'], ['4', '8'], ['1', '8'], ['6', '10'], ['3', '8'], ['8', '9'], ['2', '8'], ['8', '10']]
    # after comparing upperbound and Z value
    # runtime_id = "after_rerun"
    runtime_id = "base_model"
    rerun_failure_rate_and_rmse = False
    if rerun_failure_rate_and_rmse:
        output_folder = f'figures_report/{runtime_id}'

        for lamda in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]:
            for eps in [0.1, 0.2, 0.3, 0.4, 0.5, 1.0]:
                models_logfile_patterns = [
                        ('logs_report/base_model/rnn_numConstraintAll_combineFalse_predew{}_eps0.5_delta0.05_lambda1.0_n_trials1_seed{}.json', 'RNN'),
                        ('logs_report/'+runtime_id+'/sop_numConstraint10_combineFalse_predew{}_eps' + str(eps) + '_delta0.05_lambda'+str(lamda)+'_n_trials1_seed{}.json', 'SOP')]

                plot_rmse_avg_over_seeds(models_logfile_patterns, 
                save_imgpath_postfix=f'{runtime_id}_eps{eps}_lamda{lamda}', 
                pred_epiweeks=['202210', '202214', '202218', '202222', '202226', '202230'],
                seeds=[1234, 1235, 1236], 
                outdir=output_folder, 
                trial='trial_0', plot_eps=eps, predweeks_pairs=predweeks_pairs)

                plot_failure_rate_avg_over_seeds(models_logfile_patterns, 
                save_imgpath_postfix=f'{runtime_id}_eps{eps}_lamda{lamda}', 
                seeds=[1234, 1235, 1236], 
                pred_epiweeks=['202210', '202214', '202218', '202222', '202226', '202230'],
                outdir=output_folder, 
                trial='trial_0', plot_eps=eps, predweeks_pairs=predweeks_pairs)

    get_most_diff_pairs = True
    if get_most_diff_pairs:
        output_folder = f"figures_report/{runtime_id}"
        for eps in [0.5]:
            models_logfile_patterns = f"logs_report/base_model/FNP_numConstraintAll_combineFalse_testyear2019_weekahead1_eps" + str(eps) + "_delta0.05_lambda{}_epoch2000_seed{}.json"

            for phase in ['safetytest']:
                plot_Z_avg_seeds_with_lambda(
                    models_logfile_template=models_logfile_patterns, 
                    seeds=[1234],
                    lambdas = [1.0],
                    save_imgpath_postfix = f'base_model_eps{eps}',
                    outdir=output_folder, 
                    phase=phase,
                    find_most_diff_pairs=True)


    plot_z_values = False
    if plot_z_values:
        for pred_ew in ['202210']:
            for eps in [0.1, 0.2, 0.3, 0.4, 0.5]:
                models_logfile_patterns = "logs_report/rerun/_numConstraint10_combineFalse_predew" + pred_ew + "_eps" + str(eps) + "_delta0.05_lambda{}_n_trials1_seed{}.json"
                for phase in ['train', 'safetytest', 'test']:
                    plot_Z_avg_seeds_with_lambda(
                        models_logfile_template=models_logfile_patterns, 
                        pred_week=pred_ew,
                        seeds=[1234, 1235, 1236, 1237, 1238],
                        lambdas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0],
                        save_imgpath_postfix = f'rerun_eps{eps}',
                        outdir=output_folder, 
                        phase=phase,
                        trial='trial_0')

if __name__ == '__main__': 
    check_rerun()