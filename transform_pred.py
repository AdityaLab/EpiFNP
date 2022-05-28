import pickle
import numpy as np
from DistCal.GP_Beta_cal import GP_Beta
import DistCal.utils as utils
from sklearn.isotonic import IsotonicRegression


def build_GP(mu, sigma, target, n_u=8):
    model = GP_Beta()
    model.fit(target, mu, sigma, n_u)
    print("Done training")
    return model


def build_iso(mu, sigma, target):
    iso_q, iso_q_hat = utils.get_iso_cal_table(target, mu, sigma)
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(iso_q, iso_q_hat)
    return model


def predict_GP(model, mu, sigma, y_range):
    pdf, cdf = model.predict(y_range, mu, sigma)
    preds = utils.get_y_hat(y_range, pdf)
    var = utils.get_y_var(y_range, pdf)
    return preds, var, pdf, cdf


def predict_iso(model, mu, sigma, y_range):
    pdf1, cdf1 = utils.get_norm_q(mu.ravel(), sigma.ravel(), y_range.ravel())
    cdf = model.predict(pdf1.ravel()).reshape(pdf1.shape)
    pdf = np.diff(cdf, axis=1) / (y_range[0, 1:] - y_range[0, :-1]).ravel().reshape(
        1, -1
    ).repeat(len(mu), axis=0)
    preds = utils.get_y_hat(y_range.ravel(), pdf)
    var = utils.get_y_var(y_range.ravel(), pdf)
    return preds, var, pdf, cdf


def load_data(path):
    with open(path, "rb") as fl:
        dt = pickle.load(fl)
    return dt


model = "ili2_2017_2"
test_data = load_data(f"./saves/{model}_test.pkl")
train_data = load_data(f"./saves/{model}_train.pkl")


def transform_gp(test_data, train_data):

    gp_model = build_GP(
        train_data["pred"].astype(np.float64).reshape(-1, 1),
        np.sqrt(train_data["vars"]).astype(np.float64).reshape(-1, 1),
        train_data["target"].astype(np.float64).reshape(-1, 1),
    )
    y_range = np.linspace(
        train_data["pred"].min() - train_data["vars"].max(),
        train_data["pred"].max() + train_data["vars"].max(),
        100,
    ).reshape(1, -1)
    preds_gp, var_gp, pdf_gp, cdf_gp = predict_GP(
        gp_model,
        test_data["pred"].astype(np.float64).reshape(-1, 1),
        np.sqrt(test_data["vars"]).astype(np.float64).reshape(-1, 1),
        y_range,
    )

    return preds_gp, var_gp


def transform_iso(test_data, train_data):
    iso_model = build_iso(
        train_data["pred"].astype(np.float64).reshape(-1, 1),
        np.sqrt(train_data["vars"]).astype(np.float64).reshape(-1, 1),
        train_data["target"].astype(np.float64).reshape(-1, 1),
    )
    y_range = np.linspace(
        train_data["pred"].min() - train_data["vars"].max(),
        train_data["pred"].max() + train_data["vars"].max(),
        100,
    ).reshape(1, -1)
    preds_iso, var_iso, pdf_iso, cdf_iso = predict_iso(
        iso_model,
        test_data["pred"].astype(np.float64).reshape(-1, 1),
        np.sqrt(test_data["vars"]).astype(np.float64).reshape(-1, 1),
        y_range,
    )

    return preds_iso, var_iso
