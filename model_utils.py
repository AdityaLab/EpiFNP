from models.utils import float_tensor
import numpy as np 
import torch
from models.fnpmodels import (EmbedAttenSeq, EmbedSeq, RegressionFNP,
                              RegressionFNP2)
import torch.optim as optim
import global_config

def create_tensors(metas, seqs, ys, device):
    metas = float_tensor(metas)
    ys = float_tensor(ys)
    max_len = max([len(s) for s in seqs])
    out_seqs = np.zeros((len(seqs), max_len, seqs[0].shape[-1]), dtype="float32")
    lens = np.zeros(len(seqs), dtype="int32")
    for i, s in enumerate(seqs):
        out_seqs[i, : len(s), :] = s
        lens[i] = len(s)
    out_seqs = float_tensor(out_seqs)

    metas = metas.to(device)
    out_seqs = out_seqs.to(device)
    ys = ys.to(device)
    
    return metas, out_seqs, ys, lens


def create_mask1(lens, out_dim=1):
    ans = np.zeros((max(lens), len(lens), out_dim), dtype="float32")
    for i, j in enumerate(lens):
        ans[j - 1, i, :] = 1.0
    return float_tensor(ans)


def create_mask(lens, device, out_dim=1):
    ans = np.zeros((max(lens), len(lens), out_dim), dtype="float32")
    for i, j in enumerate(lens):
        ans[:j, i, :] = 1.0
    ans = float_tensor(ans)
    ans = ans.to(device)
    return ans

def save_model(emb_model, emb_model_full, fnp_model, file_prefix: str):
    torch.save(emb_model.state_dict(), file_prefix + "_emb_model.pth")
    torch.save(emb_model_full.state_dict(), file_prefix + "_emb_model_full.pth")
    torch.save(fnp_model.state_dict(), file_prefix + "_fnp_model.pth")


def load_model(emb_model, emb_model_full, fnp_model, file_prefix: str):
    emb_model.load_state_dict(torch.load(file_prefix + "_emb_model.pth", map_location=global_config.device))
    emb_model_full.load_state_dict(torch.load(file_prefix + "_emb_model_full.pth", map_location=global_config.device))
    fnp_model.load_state_dict(torch.load(file_prefix + "_fnp_model.pth", map_location=global_config.device))


def evaluate(emb_model, emb_model_full, fnp_model, 
            full_meta, full_x, full_y, 
            train_meta, train_x, train_y, train_mask,
            val_meta, val_x, val_y, val_mask, 
            test_meta, test_x, test_y, test_mask, sample=True, dtype="test"):
    with torch.no_grad():
        emb_model.eval()
        emb_model_full.eval()
        fnp_model.eval()
        full_embeds = emb_model_full(full_x.transpose(1, 0), full_meta)
        if dtype == "val":
            x_embeds = emb_model.forward_mask(
                val_x.transpose(1, 0), val_meta, val_mask
            )
            batch_regions = [str(ai) for ai in torch.where(val_meta == 1.0)[1].cpu().numpy()]
        elif dtype == "test":
            x_embeds = emb_model.forward_mask(
                test_x.transpose(1, 0), test_meta, test_mask
            )
            batch_regions = [str(ai) for ai in torch.where(test_meta == 1.0)[1].cpu().numpy()]

        elif dtype == "train":
            x_embeds = emb_model.forward_mask(
                train_x.transpose(1, 0), train_meta, train_mask
            )
            batch_regions = [str(ai) for ai in torch.where(train_meta == 1.0)[1].cpu().numpy()]

        # elif dtype == "all":
        #     x_embeds = emb_model.forward_mask(
        #         train_x_.transpose(1, 0), train_meta_, train_mask_
        #     )
        else:
            raise ValueError("Incorrect dtype")
        y_pred, _, vars, _, _, _, _ = fnp_model.predict(
            x_embeds, full_embeds, full_y, sample=sample
        )
    labels_dict = {"val": val_y, "test": test_y, "train": train_y} # "all": train_y_
    labels = labels_dict[dtype]
    mse_error = torch.pow(y_pred - labels, 2).mean().sqrt().detach().cpu().numpy()
    return (
        mse_error,
        y_pred.detach().cpu().numpy().ravel(),
        labels.detach().cpu().numpy().ravel(),
        vars.mean().detach().cpu().numpy().ravel(),
        full_embeds.detach().cpu().numpy(),
        x_embeds.detach().cpu().numpy(),
        batch_regions
    )

def create_model(attn, city_idx, train_meta, device): 
    if attn == "trans":
        emb_model = EmbedAttenSeq(
            dim_seq_in=1,
            dim_metadata=len(city_idx),
            dim_out=50,
            n_layers=2,
            bidirectional=True,
        ).to(device)
        emb_model_full = EmbedAttenSeq(
            dim_seq_in=1,
            dim_metadata=len(city_idx),
            dim_out=50,
            n_layers=2,
            bidirectional=True,
        ).to(device)
    else:
        emb_model = EmbedSeq(
            dim_seq_in=1,
            dim_metadata=len(city_idx),
            dim_out=50,
            n_layers=2,
            bidirectional=True,
        ).to(device)
        emb_model_full = EmbedSeq(
            dim_seq_in=1,
            dim_metadata=len(city_idx),
            dim_out=50,
            n_layers=2,
            bidirectional=True,
        ).to(device)

    fnp_model = RegressionFNP2(
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
    ).to(device)

    return emb_model, emb_model_full, fnp_model


def train_model(emb_model, emb_model_full, fnp_model, optimizer, 
            full_meta, full_x, full_y, 
            train_meta, train_x, train_y, train_mask, 
            val_meta, val_x, val_y, val_mask, 
            test_meta, test_x, test_y, test_mask,
            EPOCHS, runtimeid, n_eval, regional_criterion, before_backward_hooker=None, logger=None, savedir=None):
    
    val_error = 100.0
    losses = []
    errors = []
    train_errors = []
    variances = []
    best_ep = 0
    printed = False
    batch_regions = [str(ai) for ai in torch.where(train_meta == 1.0)[1].cpu().numpy()]

    for ep in range(EPOCHS):
        emb_model.train()
        emb_model_full.train()
        fnp_model.train()
        print(f"Epoch: {ep+1}")
        optimizer.zero_grad()
        x_embeds = emb_model.forward_mask(train_x.transpose(1, 0), train_meta, train_mask)
        full_embeds = emb_model_full(full_x.transpose(1, 0), full_meta)
        loss, yp, _ = fnp_model.forward(full_embeds, full_y, x_embeds, train_y)

        # seldonian obj loss if activated
        if before_backward_hooker is not None:
            if not printed:
                print("using Seldonian loss ...")
                printed = True
            loss = before_backward_hooker(loss, yp[full_x.shape[0] :], train_y, batch_regions, ep=ep)
        # done

        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
        train_errors.append(
            torch.pow(yp[full_x.shape[0] :] - train_y, 2)
            .mean()
            .sqrt()
            .detach()
            .cpu()
            .numpy()
        )
        print(f"Train RMSE: {train_errors[-1]:.3f}")


        # eval_params = emb_model, emb_model_full, fnp_model, \
        #             full_meta, full_x, full_y, \
        #             train_meta, train_x, train_y, train_mask,\
        #             train_meta_, train_x_, train_y_, train_mask_,\
        #             val_meta, val_x, val_y, val_mask, \
        #             test_meta, test_x, test_y, test_mask

        # e, yp, yt, _, _, _ = evaluate(*eval_params, sample=False)
        # eval_res = [evaluate(*eval_params, True, dtype="val") for _ in range(n_eval)]
        # e = np.mean([eval_resi[0] for eval_resi in eval_res])
        # vars = np.mean([[eval_resi[3] for eval_resi in eval_res]])
        # errors.append(e)
        # variances.append(vars)
        # idxs = np.random.randint(yp.shape[0], size=10)
        # print("Loss:", loss.detach().cpu().numpy())
        # print(f"Val RMSE: {e:.3f}, Train RMSE: {train_errors[-1]:.3f}")
        # # print(f"MSE: {e}")
        # if ep > 100 and min(errors[-100:]) > error + 0.1:
        #     errors = errors[: best_ep + 1]
        #     losses = losses[: best_ep + 1]
        #     print(f"Done in {ep+1} epochs")
        #     break
        # if e < error:
        #     save_model(emb_model, emb_model_full, fnp_model, f"model_chkp/model{runtimeid}")
        #     error = e
        #     best_ep = ep + 1

        logger.log_value('train/rmse', train_errors[-1].item(), ep) 

        if (ep == 0) or (ep == EPOCHS - 1): # last epoch: 
            # import pdb;pdb.set_trace()
            # save prediciton for inspection 
            obj = {r: (predi.item(), yi.item()) for r , predi, yi in zip(batch_regions, yp, train_y)}
            logger.log_value('train/prediction', obj, ep)
            Zs = regional_criterion(yp[full_x.shape[0] :], train_y, batch_regions)
            Zsmean = {rp: zs.abs().mean().detach().cpu().numpy().item() for rp, zs in Zs.items()}
            logger.log_value('train/Zsmean', Zsmean, ep, flush=True)

        if (ep % 100 == 0) or (ep == EPOCHS -1):
            save_model(emb_model, emb_model_full, fnp_model, f"model_chkp/{savedir}/model{runtimeid}")

    return val_error, errors, losses, train_errors, variances