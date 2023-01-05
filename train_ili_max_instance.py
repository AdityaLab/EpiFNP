"""Instance normalization"""
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from models.utils import float_tensor, device
import pickle
from models.fnpmodels import (
    EmbedAttenSeq,
    RegressionFNP,
    EmbedSeq,
    RegressionFNP3,
    EmbedTransSeq,
)
from transforms import scale_to_max, shift_start
import matplotlib.pyplot as plt
import pandas as pd
from optparse import OptionParser
import os, sys
from aim import Run
import logging


for d in ["model_chkp", "plots", "saves"]:
    if not os.path.exists(d):
        os.mkdir(d)


np.random.seed(20)

city_idx = {f"Region {i}": i for i in range(1, 11)}
city_idx["X"] = 0


parser = OptionParser()
parser.add_option("-y", "--year", dest="testyear", type="int", default=2022)
parser.add_option("-c", "--curr", dest="curr", type="int", default=5)
parser.add_option("-w", "--week", dest="week_ahead", type="int", default=4)
parser.add_option("-a", "--atten", dest="atten", type="string", default="trans")
parser.add_option("-d", "--decoder", dest="decoder", type="string", default="rnn")
parser.add_option("-r", "--region", dest="region", type="string", default="X")
parser.add_option("-e", "--epoch", dest="epochs", type="int")
(options, args) = parser.parse_args()


test_seasons = [options.testyear]
# train_seasons = list(range(2003, 2019))
# test_seasons = [2019]

# train_seasons = [2003, 2004, 2005, 2006, 2007, 2008, 2009]
# test_seasons = [2010]
regions = [options.region]

week_ahead = options.week_ahead
val_frac = 5
attn = options.atten
model_num = f"InstanceAR22_{options.region}_{options.decoder}_{options.curr}"
# model_num = 22
EPOCHS = options.epochs


EXPERIMENT = "TransvsRNN"
BASE_DIR = "./saves/logs"
log_dir = os.path.join(BASE_DIR, model_num)
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

aim_run = Run(repo=BASE_DIR, experiment=EXPERIMENT)
aim_run.name = model_num

hyperparams = options.__dict__
aim_run["hparams"] = hyperparams

log_format = "%(asctime)s:%(levelname)s:%(name)s:%(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    filemode="w",
    filename=os.path.join(log_dir, "log.txt"),
)
logger = logging.getLogger(__name__)
formatter = logging.Formatter(log_format)
stdouthandler = logging.StreamHandler(sys.stdout)
stdouthandler.setFormatter(formatter)
stdouthandler.setLevel(logging.INFO)
logger.addHandler(stdouthandler)

if ("Region" in regions[0]) or ("X" in regions[0]):
    csv_file = "data/ILINet.csv"
    target_feat = "% WEIGHTED ILI"
    train_seasons = list(range(2004, 2020))
else:
    csv_file = "data/ILINet_states.csv"
    target_feat = "%UNWEIGHTED ILI"
    train_seasons = list(range(2011, 2020))

logger.info(f"{train_seasons}, {test_seasons}")
logger.info(f"{week_ahead}, {attn}, {EPOCHS}")
logger.info(f"{aim_run['hparams']}")


df = pd.read_csv(csv_file)
df = df[["REGION", "YEAR", "WEEK", target_feat]]
df[target_feat] = pd.to_numeric(df[target_feat], errors="coerce").fillna(0)
df = df[(df["YEAR"] >= 2004) | ((df["YEAR"] == 2003) & (df["WEEK"] >= 20))]


def get_dataset(year: int, region: str, df=df):
    ans = df[
        ((df["YEAR"] == year) & (df["WEEK"] >= 20))
        | ((df["YEAR"] == year + 1) & (df["WEEK"] <= 20))
    ]
    return ans[ans["REGION"] == region][target_feat]


def one_hot(idx=0):
    ans = np.zeros(len(city_idx), dtype="float32")
    ans[0] = 1.0
    return ans


def save_data(obj, filepath):
    folder = os.path.dirname(filepath)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(filepath, "wb") as fl:
        pickle.dump(obj, fl)


full_x = np.array(
    [
        np.array(get_dataset(s, r), dtype="float32")[-53:]
        for s in train_seasons
        for r in regions
    ]
)

full_x_test = np.array(
    [
        np.array(get_dataset(s, r), dtype="float32")[-53:]
        for s in test_seasons
        for r in regions
    ]
)

mean_full_x, std_full_x = full_x.mean(), full_x.std()


full_meta = np.array([one_hot() for s in train_seasons for r in regions])
full_y = full_x.argmax(-1)
full_x = full_x[:, :, None]
mean_first = full_x[:, 0].mean()


full_meta_test = np.array([one_hot() for s in test_seasons for r in regions])
full_y_test = full_x_test.argmax(-1)
full_x_test = full_x_test[:, :, None]


# Scale the first vale of test data to be the same as train data
# full_x_test = full_x_test - (full_x_test[:, 0].mean() - mean_first)

# full_x, full_x_mask = shift_start(full_x.squeeze(-1), full_x_test.squeeze())
# full_x = full_x[:, :, None]

plt.figure(figsize=(10, 10))
for i, r in enumerate(train_seasons):
    plt.plot(full_x[i], color="blue", alpha=0.1)
plt.plot(full_x_test[0], color="red")
plt.savefig(f"./plots/ILI_{test_seasons[0]}_test.png")
plt.clf()


def create_dataset(full_meta, full_x, week_ahead=week_ahead):
    metas, seqs, y, dp_mean, dp_std = [], [], [], [], []
    for meta, seq in zip(full_meta, full_x):
        for i in range(20, full_x.shape[1]):
            metas.append(meta)
            sq = seq[: i - week_ahead + 1]
            sq_mean, sq_std = sq.mean(0), sq.std(0)
            seqs.append((sq - sq_mean) / sq_std)
            y.append((seq[i - week_ahead + 1 : i + 1] - sq_mean) / sq_std)
            dp_mean.append(sq_mean)
            dp_std.append(sq_std)
    return (
        np.array(metas, dtype="float32"),
        seqs,
        np.array(y, dtype="float32"),
        np.array(dp_mean, dtype="float32"),
        np.array(dp_std, dtype="float32"),
    )


def create_dataset_test(full_meta, full_x, week_ahead=week_ahead):
    metas, seqs, y, dp_mean, dp_std = [], [], [], [], []
    for meta, seq in zip(full_meta, full_x):
        for i in range(20, full_x.shape[1]):
            metas.append(meta)
            sq = seq[: i - week_ahead + 1]
            dp_mean.append(sq.mean(0))
            dp_std.append(sq.std(0))
            seqs.append((sq - sq.mean(0)) / sq.std(0))
            y.append((seq[i - week_ahead + 1 : i + 1] - sq.mean(0)) / sq.std(0))
        for i in range(full_x.shape[1], full_x.shape[1] + week_ahead - 1):
            metas.append(meta)
            sq = seq[: i - week_ahead + 1]
            dp_mean.append(sq.mean(0))
            dp_std.append(sq.std(0))
            seqs.append((sq - sq.mean(0)) / sq.std(0))
            y.append(
                np.array(
                    [(seq[i - week_ahead + 1] - sq.mean(0)) / sq.std(0)] * week_ahead,
                    dtype="float32",
                )
            )
    return (
        np.array(metas, dtype="float32"),
        seqs,
        np.array(y, dtype="float32"),
        np.array(dp_mean, dtype="float32"),
        np.array(dp_std, dtype="float32"),
    )


train_meta, train_x, train_y, train_mean, train_std = create_dataset(full_meta, full_x)
test_meta, test_x, test_y, test_mean, test_std = create_dataset_test(
    full_meta_test, full_x_test
)
train_meta = np.concatenate([train_meta, test_meta[: options.curr]], axis=0)
train_x = train_x + test_x[: options.curr]
train_y = np.concatenate([train_y, test_y[: options.curr]], axis=0)
train_mean = np.concatenate([train_mean, test_mean[: options.curr]], axis=0)
train_std = np.concatenate([train_std, test_std[: options.curr]], axis=0)


def create_tensors(metas, seqs, ys, means, stds):
    metas = float_tensor(metas)
    ys = float_tensor(ys)
    means, stds = float_tensor(means), float_tensor(stds)
    max_len = max([len(s) for s in seqs])
    out_seqs = np.zeros((len(seqs), max_len, seqs[0].shape[-1]), dtype="float32")
    lens = np.zeros(len(seqs), dtype="int32")
    for i, s in enumerate(seqs):
        out_seqs[i, : len(s), :] = s
        lens[i] = len(s)
    out_seqs = float_tensor(out_seqs)
    return metas, out_seqs, ys, lens, means, stds


def create_mask1(lens, out_dim=1):
    ans = np.zeros((max(lens), len(lens), out_dim), dtype="float32")
    for i, j in enumerate(lens):
        ans[j - 1, i, :] = 1.0
    return float_tensor(ans)


def create_mask(lens, out_dim=1):
    ans = np.zeros((max(lens), len(lens), out_dim), dtype="float32")
    for i, j in enumerate(lens):
        ans[:j, i, :] = 1.0
    return float_tensor(ans)


if attn == "trans":
    emb_model: torch.nn.Module = EmbedAttenSeq(
        dim_seq_in=1,
        dim_metadata=len(city_idx),
        dim_out=50,
        n_layers=2,
        bidirectional=True,
    ).cuda()
    emb_model_full: torch.nn.Module = EmbedAttenSeq(
        dim_seq_in=1,
        dim_metadata=len(city_idx),
        dim_out=50,
        n_layers=2,
        bidirectional=True,
    ).cuda()
elif attn == "self":
    emb_model = EmbedTransSeq(
        dim_seq_in=1,
        dim_metadata=len(city_idx),
        dim_out=50,
        n_layers=2,
        n_heads=2,
    ).cuda()
    emb_model_full = EmbedTransSeq(
        dim_seq_in=1,
        dim_metadata=len(city_idx),
        dim_out=50,
        n_layers=2,
        n_heads=2,
    ).cuda()
else:
    emb_model = EmbedSeq(
        dim_seq_in=1,
        dim_metadata=len(city_idx),
        dim_out=50,
        n_layers=2,
        bidirectional=True,
    ).cuda()
    emb_model_full = EmbedSeq(
        dim_seq_in=1,
        dim_metadata=len(city_idx),
        dim_out=50,
        n_layers=2,
        bidirectional=True,
    ).cuda()
fnp_model = RegressionFNP3(
    dim_x=50,
    dim_y=1,
    dim_h=100,
    n_layers=3,
    num_M=train_meta.shape[0],
    dim_u=50,
    dim_z=50,
    fb_z=0.0,
    use_ref_labels=False,
    use_DAG=False,
    add_atten=False,
    time_steps=week_ahead,
    use_GRU=(options.decoder == "gru"),
).cuda()
optimizer = optim.Adam(
    list(emb_model.parameters())
    + list(fnp_model.parameters())
    + list(emb_model_full.parameters()),
    lr=1e-3,
)

# schduler = optim.lr_scheduler.MultiStepLR(optimizer, [10, 500, 3000])

# emb_model_full = emb_model

train_meta_, train_x_, train_y_, train_lens_, train_mean_, train_std_ = create_tensors(
    train_meta, train_x, train_y, train_mean, train_std
)

test_meta, test_x, test_y, test_lens, test_mean, test_std = create_tensors(
    test_meta, test_x, test_y, test_mean, test_std
)

full_x_chunks = np.zeros((full_x.shape[0] * 4, full_x.shape[1], full_x.shape[2]))
full_meta_chunks = np.zeros((full_meta.shape[0] * 4, full_meta.shape[1]))
for i, s in enumerate(full_x):
    full_x_chunks[i * 4, -20:] = s[:20]
    full_x_chunks[i * 4 + 1, -30:] = s[:30]
    full_x_chunks[i * 4 + 2, -40:] = s[:40]
    full_x_chunks[i * 4 + 3, :] = s
    full_meta_chunks[i * 4 : i * 4 + 4] = full_meta[i]

full_x = float_tensor(full_x)
full_meta = float_tensor(full_meta)
full_y = float_tensor(full_y)

train_mask_, test_mask = (
    create_mask(train_lens_),
    create_mask(test_lens),
)

perm = np.random.permutation(train_meta_.shape[0])
val_perm = perm[: train_meta_.shape[0] // val_frac]
train_perm = perm[train_meta_.shape[0] // val_frac :]

train_meta, train_x, train_y, train_lens, train_mask, train_mean, train_std = (
    train_meta_[train_perm],
    train_x_[train_perm],
    train_y_[train_perm],
    train_lens_[train_perm],
    train_mask_[:, train_perm, :],
    train_mean_[train_perm],
    train_std_[train_perm],
)
val_meta, val_x, val_y, val_lens, val_mask, val_mean, val_std = (
    train_meta_[val_perm],
    train_x_[val_perm],
    train_y_[val_perm],
    train_lens_[val_perm],
    train_mask_[:, val_perm, :],
    train_mean_[val_perm],
    train_std_[val_perm],
)


def save_model(file_prefix: str):
    torch.save(emb_model.state_dict(), file_prefix + "_emb_model.pth")
    torch.save(emb_model_full.state_dict(), file_prefix + "_emb_model_full.pth")
    torch.save(fnp_model.state_dict(), file_prefix + "_fnp_model.pth")


def load_model(file_prefix: str):
    emb_model.load_state_dict(torch.load(file_prefix + "_emb_model.pth"))
    emb_model_full.load_state_dict(torch.load(file_prefix + "_emb_model_full.pth"))
    fnp_model.load_state_dict(torch.load(file_prefix + "_fnp_model.pth"))


def evaluate(sample=True, dtype="test"):
    with torch.no_grad():
        emb_model.eval()
        emb_model_full.eval()
        fnp_model.eval()
        full_embeds = emb_model_full(full_x.transpose(1, 0), full_meta)
        if dtype == "val":
            x_embeds = emb_model.forward_mask(val_x.transpose(1, 0), val_meta, val_mask)
        elif dtype == "test":
            x_embeds = emb_model.forward_mask(
                test_x.transpose(1, 0), test_meta, test_mask
            )
        elif dtype == "train":
            x_embeds = emb_model.forward_mask(
                train_x.transpose(1, 0), train_meta, train_mask
            )
        elif dtype == "all":
            x_embeds = emb_model.forward_mask(
                train_x_.transpose(1, 0), train_meta_, train_mask_
            )
        else:
            raise ValueError("Incorrect dtype")
        y_pred, _, vars, _, _, _, _ = fnp_model.predict(
            x_embeds, full_embeds, full_y, sample=sample
        )
    labels_dict = {
        "val": (val_y, val_mean, val_std),
        "test": (test_y, test_mean, test_std),
        "train": (train_y, train_mean, train_std),
        "all": (train_y_, train_mean_, train_std_),
    }
    labels = (labels_dict[dtype][0].squeeze(2) * labels_dict[dtype][2]) + labels_dict[
        dtype
    ][1]
    y_pred = (y_pred * labels_dict[dtype][2]) + labels_dict[dtype][1]
    mse_error = torch.pow(y_pred - labels, 2).mean().sqrt().detach().cpu().numpy()
    return (
        mse_error,
        y_pred.detach().cpu().numpy(),
        labels.detach().cpu().numpy(),
        vars.mean().detach().cpu().numpy(),
        full_embeds.detach().cpu().numpy(),
        x_embeds.detach().cpu().numpy(),
    )


error = 100.0
losses = []
errors = []
train_errors = []
variances = []
best_ep = 0

for ep in range(EPOCHS):
    emb_model.train()
    emb_model_full.train()
    fnp_model.train()
    logger.info(f"Epoch: {ep+1}")
    optimizer.zero_grad()
    x_embeds = emb_model.forward_mask(train_x.transpose(1, 0), train_meta, train_mask)
    full_embeds = emb_model_full(full_x.transpose(1, 0), full_meta)
    loss, yp, _ = fnp_model.forward(full_embeds, full_y, x_embeds, train_y)
    loss.backward()
    optimizer.step()
    # schduler.step()
    losses.append(loss.detach().cpu().numpy())
    train_errors.append(
        torch.pow(yp[full_x.shape[0] :] - train_y.squeeze(2), 2)
        .mean()
        .sqrt()
        .detach()
        .cpu()
        .numpy()
    )

    e, yp, yt, _, _, _ = evaluate(False)
    e_vars = [evaluate(True, dtype="val") for _ in range(40)]
    e = np.mean([e[0] for e in e_vars])
    vars = np.mean([e[3] for e in e_vars])
    errors.append(e)
    variances.append(vars)
    idxs = np.random.randint(yp.shape[0], size=10)
    logger.info(f"Loss: {loss.detach().cpu().numpy():.3f}")
    logger.info(f"Val RMSE: {e:.3f}, Train RMSE: {train_errors[-1]:.3f}")
    aim_run.track(e, "error", context={"subset": "val"})
    aim_run.track(loss.detach().cpu().numpy(), "loss", context={"subset": "train"})
    aim_run.track(train_errors[-1], "error", context={"subset": "train"})
    aim_run.track(vars, "variance", context={"subset": "val"})
    # print(f"MSE: {e}")
    if ep > 500 and min(errors[-300:]) > error + 0.1:
        errors = errors[: best_ep + 1]
        losses = losses[: best_ep + 1]
        logger.info(f"Done in {ep+1} epochs")
        break
    if e < error:
        save_model(f"model_chkp/model{model_num}")
        error = e
        best_ep = ep + 1
        aim_run["best_epoch"] = int(best_ep)
        aim_run["best_val_error"] = float(error)


logger.info(f"Val MSE error: {error}")
plt.figure(1)
plt.plot(losses)
plt.savefig(f"plots/losses{model_num}.png")
plt.figure(2)
plt.plot(errors)
plt.plot(train_errors)
plt.savefig(f"plots/errors{model_num}.png")
plt.figure(3)
plt.plot(variances)
plt.savefig(f"plots/vars{model_num}.png")

load_model(f"model_chkp/model{model_num}")

e, yp, yt, vars, fem, tem = evaluate(True)
yp_raw = np.array([evaluate(True)[1] for _ in range(1000)])
yp, vars = np.mean(yp_raw, 0), np.var(yp_raw, 0)
e = np.mean((yp - yt) ** 2)
dev = np.sqrt(vars) * 1.95
for i in range(week_ahead):
    plt.figure(4)
    plt.clf()
    plt.plot(yp[:, i], label="Predicted 95%", color="blue")
    plt.fill_between(
        np.arange(len(yp)),
        yp[:, i] + dev[:, i],
        yp[:, i] - dev[:, i],
        color="blue",
        alpha=0.2,
    )
    plt.plot(yt[:, i], label="True Value", color="green")
    plt.legend()
    plt.title(f"RMSE: {e}")
    plt.savefig(f"plots/Test{model_num}_{i+1}.png")
dt = {
    "rmse": e,
    "target": yt,
    "pred": yp,
    "pred_raw": yp_raw,
    "vars": vars,
    "fem": fem,
    "tem": tem,
}
save_data(dt, f"./saves/real_time/{model_num}_test.pkl")

e, yp, yt, vars, _, _ = evaluate(True, dtype="val")
yp = np.array([evaluate(True, dtype="val")[1] for _ in range(1000)])
yp, vars = np.mean(yp, 0), np.var(yp, 0)
e = np.mean((yp - yt) ** 2)
dev = np.sqrt(vars) * 1.95
for i in range(week_ahead):
    plt.figure(5)
    plt.clf()
    plt.plot(yp[:, i], label="Predicted 95%", color="blue")
    plt.fill_between(
        np.arange(len(yp)),
        yp[:, i] + dev[:, i],
        yp[:, i] - dev[:, i],
        color="blue",
        alpha=0.2,
    )
    plt.plot(yt[:, i], label="True Value", color="green")
    plt.legend()
    plt.title(f"RMSE: {e}")
    plt.savefig(f"plots/Val{model_num}_{i+1}.png")

e, yp, yt, vars, fem, tem = evaluate(True, dtype="all")
yp = np.array([evaluate(True, dtype="all")[1] for _ in range(40)])
yp, vars = np.mean(yp, 0), np.var(yp, 0)
e = np.mean((yp - yt) ** 2)
dev = np.sqrt(vars) * 1.95
for i in range(week_ahead):
    plt.figure(5)
    plt.clf()
    plt.plot(yp[:, i], label="Predicted 95%", color="blue")
    plt.fill_between(
        np.arange(len(yp)),
        yp[:, i] + dev[:, i],
        yp[:, i] - dev[:, i],
        color="blue",
        alpha=0.2,
    )
    plt.plot(yt[:, i], label="True Value", color="green")
    plt.legend()
    plt.title(f"RMSE: {e}")
    plt.savefig(f"plots/Train{model_num}_{i+1}.png")
dt = {
    "rmse": e,
    "target": yt,
    "pred": yp,
    "vars": vars,
    "fem": fem,
    "tem": tem,
}
save_data(dt, f"./saves/{model_num}_train.pkl")
aim_run.close()
