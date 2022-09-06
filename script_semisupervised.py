#%%
"""Script to test semisupervised learning methods on synthetic and real data
"""
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import scale
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from torch import cuda
from torch.nn.functional import softmax
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from functions.functions_semisupervise import (
    NN,
    ShowLatentSpace,
    netBio,
    proj_l1ball,
    proj_l1infball,
    proj_l11ball,
    show_img,
    weights_and_sparsity,
)

mpl.rcParams["axes.titlesize"] = 15  # titre du plot
mpl.rcParams["axes.labelsize"] = 15  # titres des axes
mpl.rcParams["legend.fontsize"] = 12  # legende des courbes

#%%
## Script constants
results_dir = "results_semi/"
plots_dir = "plots/"
# set to True if you want to complete an existing accuracy results csv
resume = False
output_name = "Acc_Temp.csv"  # name of the accuracy/unlabeled prop or accuracy/separability output csv
losses_file_name = "Losses_Temp.csv"  # name of the losses-per-epoch output csv
SEEDS = [7]
PROGRESS_BAR = True  # display progress bars for the training of the SAE

# display distribution of labels (plots will be saved in plots_dir even if the plot booleans are False)
PLOT_DISTRIB = False
BW = 1  # smoothing parameter for the distribution plots

SHOW_LATENT_SPACE = False  # plot latent space of SAE
PLOT_MATRICES = True  # plot the feature/neurons matrices

## Parameters for real data
file_name = None  # To use synthetic data
# file_name = "LUNG.csv"  # To use a real csv
# file_name = "IPFcellATLAS.csv"  # To use a real csv
# UNL_PROPS = [
#     0.1,
#     0.2,
#     0.3,
#     0.4,
#     0.5,
#     0.6,
#     0.7,
#     0.8,
#     0.9,
# ]  # unlabeled proportions to try
UNL_PROPS = [0.4]  # unlabeled proportions to try

## Parameters for synthetic data
UNLABELED_PROPORTION = 0.4  # default unlabeled proportion
# SEPARABILITIES = [0.3, 0.6, 0.9, 1.2, 1.5, 2]  # separabilities to try
SEPARABILITIES = [0.8]
N_FEATURES = 10000
N_SAMPLES = 5000

## SAE params
N_EPOCHS = 40  # total number of epochs (Adam + SWA)
LR_NN = 1e-4  # FCNN Learning rate
LEARNING_RATE = 1e-4  # Adam learning rate
LOSS_LAMBDA = 0.0005  # weight of the reconstruction loss
BATCH_SIZE = 4
# SWA params
SWA_START = 33  # start of the Stochastic Weight Averaging
SWA_LR = 1e-4  # swa learning rate

# PROJECTION = None  # No projection
PROJECTION = proj_l11ball  # Projection L11
# PROJECTION = proj_l1ball     # Projection L1
# PROJECTION = proj_l1infball  # Projection L1inf
ETA = 2375  # ETA for IPF
# ETA = 93  # ETA for LUNG

#%%
def plot_distributions(df_softmax, model_name, file_name, is_swa=False, bW=1):
    """Plot the prediction scores distributions"""
    plt.figure(figsize=(5, 2))
    sns.kdeplot(
        data=1
        - df_softmax["Score class 0"]
        .where(df_softmax["Score class 0"] >= df_softmax["Score class 1"])
        .dropna(),
        bw_adjust=bW,
        shade=True,
        color="tab:blue",
        label="Class 0",
    )
    sns.kdeplot(
        data=df_softmax["Score class 1"]
        .where(df_softmax["Score class 0"] < df_softmax["Score class 1"])
        .dropna(),
        bw_adjust=bW,
        shade=True,
        color="tab:orange",
        label="Class 1",
    )
    plt.xlabel("")
    plt.ylabel("")
    plt.legend()
    distrib_fig_name = f"plots/distribs/{file_name[:-4] if file_name is not None else 'synth'}_{model_name}_{'SWA_' if is_swa else ''}distrib.png"
    plt.savefig(
        distrib_fig_name, dpi=400, facecolor="white", bbox_inches="tight",
    )
    if PLOT_DISTRIB:
        plt.show()
    else:
        plt.close()


#%%
# Create synthetic data


def get_data(
    file_name=None,
    unlabeled_prop=UNLABELED_PROPORTION,
    n_samples=N_SAMPLES,
    separability=1.5,
    seed=6,
):
    """Get or Make data, and split it into unlabeled and labeled sets.
    To make synthetic data, pass file_name = None.
    returns (X_labeled, X_unlabeled, y_labeled, y_unlabeled)
    """
    if file_name is None:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=N_FEATURES,
            n_informative=N_FEATURES,
            n_redundant=0,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=1,
            # weights=[0.1, 0.9],  # to make an unbalanced dataset
            class_sep=separability,
            hypercube=True,
            random_state=seed,
        )

    else:
        df = pd.read_csv(f"data/{file_name}", sep=";", header=0).transpose()
        df.columns = df.loc["Name"]
        df.drop("Name", inplace=True)
        y = df["Label"].to_numpy() - 1  # Label in [1, 2] -> Label in [0, 1]
        X = df.drop("Label", axis=1).to_numpy()

        # log transform, mean centering and unit scaling
        X = np.log(abs(X.astype(np.float32)) + 1)
        X = X - np.mean(X, axis=0)
        X = scale(X, axis=0, with_mean=False)
        n_samples = len(y)

    # generate indexes of unlabeled points
    np.random.seed(seed)
    unlabeled_idx = np.random.randint(
        0, n_samples, size=int(n_samples * unlabeled_prop)
    )
    # get the rest of the indexes, which will represent labeled points
    labeled_idx = list(set(range(n_samples)).difference(set(unlabeled_idx)))

    # split data
    X_unlabeled, X_labeled = X[unlabeled_idx], X[labeled_idx]
    y_unlabeled, y_labeled = y[unlabeled_idx], y[labeled_idx]

    return X_labeled, X_unlabeled, y_labeled, y_unlabeled


#%%
# Classify with LabelPropagation


def label_sklearn(X_l, X_unl, y_l, y_unl, seed, algo=LabelPropagation):
    """Label data using a semi-supervised algorithm from sklearn

    Parameters
    ----------
    X_l : Labeled X data
    X_unl : Unlabeled X data
    y_l : Labels of the X_l samples
    y_unl : True Labels of the X_unl samples
    algo : Classifier class, optional
        Semi-supervised sklearn classifier, by default LabelPropagation
    """
    n_classes = len(np.unique(y_l))

    # Note : rbf kernel often does not converge on high-dimensional data
    classifier = algo(kernel="knn")
    classifier.fit(X_l, y_l.astype(int))

    y_unl_pred = classifier.predict(X_unl)
    res = pd.DataFrame({"y": y_unl.astype(int), "y_pred": y_unl_pred})
    res["correct"] = res.apply(lambda row: row.y == row.y_pred, axis=1)

    df_softmax = pd.DataFrame(
        classifier.predict_proba(X_unl),
        columns=[f"Score class {i}" for i in range(n_classes)],
    )

    df_softmax.to_csv(
        results_dir
        + f"labelpredicts/labelpredict{file_name[:-4] if file_name is not None else 'Synth'}"
        + algo.__name__
        + ".csv",
        index_label="Name",
    )
    save_metrics(res, df_softmax, name=algo.__name__, seed=seed)

    plot_distributions(df_softmax, model_name=algo.__name__, file_name=file_name, bW=2)

    return {"res": res}


#%%
# Classify with the SAE

if cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


class SAE_Dataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X.astype("float32")
        self.Y = y

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self) -> int:
        return len(self.X)


def label_network(
    X_l,
    X_unl,
    y_l,
    y_unl,
    autoencoder=False,
    n_epochs=N_EPOCHS,
    save_losses_file=None,
    model_name=None,
    seed=None,
):
    # if save_losses_file is not None:
    #     try:
    #         # complete an existing csv
    #         df_losses = pd.read_csv(
    #             results_dir + save_losses_file, sep=";", index_col="epoch"
    #         )
    #         swa_df_losses = pd.read_csv(
    #             results_dir + "SWA_" + save_losses_file, sep=";", index_col="epoch"
    #         )
    #         # check that the existing csv was produced with the same number of epochs
    #         assert len(df_losses.index) == N_EPOCHS
    #         assert len(swa_df_losses.index) == N_EPOCHS
    #     except:
    #         # create a new csv
    #         df_losses = pd.DataFrame(index=list(range(N_EPOCHS)))
    #         swa_df_losses = pd.DataFrame(index=list(range(N_EPOCHS)))
    #     assert model_name is not None  # we need a model's name to save its losses

    torch.manual_seed(seed)
    # prepare torch datasets and dataloaders
    l_dataset = SAE_Dataset(X_l, y_l)
    unl_dataset = SAE_Dataset(X_unl, y_unl)

    unl_dataloader = DataLoader(unl_dataset, shuffle=False, batch_size=1)
    l_dataloader = DataLoader(l_dataset, shuffle=True, batch_size=8)

    # init model, losses and optimizer
    n_classes = len(np.unique(y_l))
    model, model_name = get_model(
        n_inputs=X_l.shape[1],
        n_outputs=n_classes,
        model_name=model_name,
        seed=seed,
        autoencoder=autoencoder,
    )

    res_dict, trained_model = full_network_loop(
        model,
        l_dataloader,
        unl_dataloader,
        n_classes,
        N_EPOCHS,
        model_name,
        autoencoder,
    )

    if PROJECTION is not None and autoencoder:
        # Do second descent
        model, model_name = get_model(
            n_inputs=X_l.shape[1],
            n_outputs=n_classes,
            model_name=model_name,
            seed=seed,
            autoencoder=autoencoder,
            initial=False,
            prev_model=trained_model,
        )
        res_dict, _ = full_network_loop(
            model,
            l_dataloader,
            unl_dataloader,
            n_classes,
            N_EPOCHS,
            model_name,
            autoencoder,
            prev_results=res_dict,
        )

    return res_dict


def full_network_loop(
    model,
    l_dataloader,
    unl_dataloader,
    n_classes,
    n_epochs,
    model_name,
    autoencoder,
    prev_results=None,
):
    model.train()
    c_loss = nn.CrossEntropyLoss(reduction="sum")
    r_loss = nn.SmoothL1Loss(reduction="sum")
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE if autoencoder else LR_NN
    )
    swa_sched = SWALR(optimizer, swa_lr=SWA_LR, anneal_strategy="linear")

    swa_model = AveragedModel(model, device=DEVICE)

    loss_list, swa_loss_list, data_encoder = training_loop(
        n_epochs,
        l_dataloader,
        model,
        c_loss,
        r_loss,
        optimizer,
        swa_model,
        swa_sched,
        autoencoder=autoencoder,
    )

    plot_latent_space(
        autoencoder,
        data_encoder,
        with_proj=prev_results is not None,
        name_modifier="Labeled",
    )

    # if save_losses_file is not None:  # save losses per epoch
    #     df_losses[model_name] = loss_list.numpy()
    #     df_losses.to_csv(
    #         results_dir + PROJECTION.__name__ + save_losses_file,
    #         sep=";",
    #         index_label="epoch",
    #     )
    #     swa_df_losses[model_name] = swa_loss_list.numpy()
    #     swa_df_losses.to_csv(
    #         results_dir + PROJECTION.__name__ + "SWA_" + save_losses_file,
    #         sep=";",
    #         index_label="epoch",
    #     )

    # predict over X_unl
    res, df_softmax, data_encoder_unl = predict(
        model, unl_dataloader, n_classes, DEVICE, autoencoder=autoencoder
    )
    plot_latent_space(
        autoencoder,
        data_encoder_unl,
        with_proj=prev_results is not None,
        name_modifier="Unlabeled",
    )
    # predict over X_unl with the averaged model from SWA
    res_swa, df_softmax_swa, data_encoder_unl_swa = predict(
        swa_model, unl_dataloader, n_classes, DEVICE, autoencoder=autoencoder
    )
    df_softmax_swa.to_csv(
        results_dir
        + f"labelpredicts/labelpredict{file_name[:-4] if file_name is not None else 'Synth'}{'NN' if not autoencoder else ''}.csv",
        index_label="Name",
    )
    plot_latent_space(
        autoencoder,
        data_encoder_unl_swa,
        with_proj=prev_results is not None,
        name_modifier="SWA Unlabeled",
    )

    # save all metrics (Accuracy, AUC, etc...)
    name_modifier = "2nd_descent_" if prev_results is not None else ""
    save_metrics(
        res,
        df_softmax,
        name=f"{name_modifier}{'SAE' if autoencoder else 'NN'}",
        seed=seed,
    )
    save_metrics(
        res_swa,
        df_softmax_swa,
        name=f"SWA_{name_modifier}{'SAE' if autoencoder else 'NN'}",
        seed=seed,
    )

    # plot distributions with kernel
    plot_distributions(df_softmax, model_name, file_name, is_swa=False)
    plot_distributions(df_softmax_swa, model_name, file_name, is_swa=True)

    # Get the labeled data distribution
    _, l_df_softmax, _ = predict(
        swa_model, l_dataloader, n_classes, DEVICE, autoencoder=autoencoder
    )
    l_df_softmax.to_csv(
        results_dir
        + f"labelpredicts/labeled_labelpredict{file_name[:-4] if file_name is not None else 'Synth'}{'NN' if not autoencoder else ''}.csv",
        index_label="Name",
    )
    plot_distributions(
        l_df_softmax,
        model_name,
        file_name=f"labeled_{file_name if file_name is not None else 'synth'}",
        is_swa=True,
        bW=1.2,
    )
    if autoencoder and prev_results is not None:
        # if second descent of the SSAE,
        # plot matrix features/neurons
        enc_w, _ = weights_and_sparsity(model.encoder)
        dec_w, _ = weights_and_sparsity(model.decoder)
        layers_enc = list(enc_w.values())
        layers_dec = list(dec_w.values())
        plt.figure(figsize=(8, 5))
        show_img(layers_enc, layers_dec, file_name=file_name or "Synth")
        plt.savefig(
            f"plots/features_matrix_{file_name[:-4] if file_name is not None else 'Synthetic'}_2nd.png",
            dpi=100,
            facecolor="white",
            bbox_inches="tight",
        )
        if PLOT_MATRICES:
            plt.show()
        else:
            plt.close()

    if prev_results is not None:
        prev_results.update(
            {"res_2nd": res, "res_swa_2nd": res_swa,}
        )
        return prev_results, model
    return {"res": res, "res_swa": res_swa,}, model


def get_model(
    n_inputs,
    n_outputs,
    model_name,
    seed=6,
    autoencoder=True,
    initial=True,
    prev_model=None,
):
    """Initialize a neural network"""
    torch.manual_seed(seed)
    if not autoencoder:
        return NN(n_inputs, n_outputs).to(DEVICE), model_name

    if initial:
        return netBio(n_inputs, n_outputs).to(DEVICE), model_name

    else:
        assert prev_model is not None
        model_name += f"_{PROJECTION.__name__}"
        mask = compute_mask(prev_model, PROJECTION, projection_param=ETA)
        print(
            "Begin second descent (Density : "
            f"{torch.sum(torch.flatten(mask))/torch.prod(torch.tensor(mask.shape)):.4f})"
        )
        model = netBio(n_inputs, n_outputs)
        model.register_buffer(name="mask", tensor=mask)
        model.register_forward_pre_hook(mask_gradient)
        model = model.to(DEVICE)
        return model, model_name


def training_loop(
    n_epochs,
    l_dataloader,
    model,
    c_loss,
    r_loss,
    optimizer,
    swa_model,
    swa_sched,
    loss_lambda=LOSS_LAMBDA,
    autoencoder=False,
):
    loss_list = torch.zeros(size=(n_epochs, 1))
    swa_loss_list = torch.zeros(size=(n_epochs, 1))

    # train for n epochs over X_l
    for i in prog(range(n_epochs)):
        total_batch_loss = 0
        for b_id, batch in enumerate(l_dataloader):
            x, lab = batch
            x = x.to(DEVICE)
            lab = lab.to(DEVICE)

            if autoencoder:
                # SAE
                net_out, decoder_out = model(x)
                loss = c_loss(net_out, lab.long()) + loss_lambda * r_loss(
                    decoder_out, x
                )

            else:
                # NN
                net_out = model(x)
                loss = c_loss(net_out, lab.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # prepare data_encoder to plot the latent space later
            if i == n_epochs - 1:
                # last epoch
                if b_id == 0:
                    # fist batch, we need to define data_encoder
                    data_encoder = torch.cat((net_out, lab.view(-1, 1)), dim=1)
                else:
                    data_encoder = torch.cat(
                        (data_encoder, torch.cat((net_out, lab.view(-1, 1)), dim=1),),
                        dim=0,
                    )

            total_batch_loss += loss.detach().cpu().item()

        current_loss = total_batch_loss / len(l_dataloader)

        if i >= SWA_START:
            swa_model.update_parameters(model)
            swa_loss_list[i] = compute_swa_loss(
                swa_model, c_loss, r_loss, loss_lambda, l_dataloader, autoencoder
            ) / len(l_dataloader)
            swa_sched.step()
        else:
            swa_loss_list[i] = current_loss
        loss_list[i] = current_loss
    return loss_list, swa_loss_list, data_encoder


@torch.no_grad()
def compute_swa_loss(
    swa_model, c_loss, r_loss, loss_lambda, dataloader, autoencoder=False
):
    """Artificially compute the loss for an averaged model during training"""
    swa_model.eval()
    total_batch_loss = 0.0
    for batch in dataloader:
        x, lab = batch
        x = x.to(DEVICE)
        lab = lab.to(DEVICE)
        if autoencoder:
            swa_enc_out, swa_dec_out = swa_model(x)
            swa_loss = c_loss(swa_enc_out, lab.long()) + loss_lambda * r_loss(
                swa_dec_out, x
            )
        else:
            swa_net_out = swa_model(x)
            swa_loss = c_loss(swa_net_out, lab.long())
        total_batch_loss += swa_loss.detach().cpu().item()
    return total_batch_loss


def compute_mask(net, projection, projection_param):
    # threshold under which a weight is considered zero
    tol = 1.0e-4
    full_mask = []
    for index, param in enumerate(net.parameters()):
        if index < len(list(net.parameters())) / 2 - 2 and index % 2 == 0:
            # compute mask for all concerned layers
            mask = torch.where(
                condition=(
                    torch.abs(
                        projection(param.detach().clone(), projection_param, DEVICE)
                    )
                    < tol
                ).to(DEVICE),
                input=torch.zeros_like(param),
                other=torch.ones_like(param),
            )
            full_mask.append(mask)
    # turn list of masks into full mask tensor
    return torch.stack(full_mask)


def mask_gradient(module, _):
    for index, param in enumerate(module.parameters()):
        if index < len(list(module.parameters())) / 2 - 2 and index % 2 == 0:
            param.data = module.mask[index] * param.data


@torch.no_grad()
def predict(model, dataloader, n_classes, device="cpu", autoencoder=False):
    """Inference predictions on a test dataloader"""
    model.eval()
    res = pd.DataFrame(columns=["y", "y_pred", "correct"])
    df_softmax = pd.DataFrame(columns=[f"Score class {i}" for i in range(n_classes)])
    for i, batch in enumerate(dataloader):
        x, lab = batch
        x = x.to(device)
        lab = lab.to(device)
        if autoencoder:
            encoder_out, _ = model(x)
        else:
            encoder_out = model(x)

        if i == 0:
            # fist batch, we need to define data_encoder
            data_encoder = torch.cat((encoder_out, lab.view(-1, 1)), dim=1)
        else:
            data_encoder = torch.cat(
                (data_encoder, torch.cat((encoder_out, lab.view(-1, 1)), dim=1),),
                dim=0,
            )

        df_softmax = pd.concat(
            [
                df_softmax,
                pd.DataFrame(
                    softmax(encoder_out, dim=1).detach().cpu().numpy(),
                    columns=[f"Score class {j}" for j in range(n_classes)],
                ),
            ],
            ignore_index=True,
        )

        if dataloader.batch_size != 1:
            # if batch_size is different than 1, we do not care about the correctness of the predictions
            pass
        else:
            res.at[i, "y"] = int(lab.cpu().item())
            pred_label = np.argmax(encoder_out.detach().cpu().numpy())
            res.at[i, "y_pred"] = pred_label
            res.at[i, "correct"] = pred_label == int(lab.cpu().item())
    return res, df_softmax, data_encoder


def plot_latent_space(autoencoder, data_encoder, with_proj=False, name_modifier=""):
    """Plot and save the latent space of the autoencoder"""
    if autoencoder:  # remove this to plot the NN's output space
        if file_name is None:
            fn = "Synth...."  # artificial file name so that fn[:-4] gives 'Synth'
        else:
            fn = file_name

        if autoencoder:
            name_modifier = "SAE " + name_modifier
        else:
            name_modifier = "NN " + name_modifier
        plt.figure(figsize=(8, 5))
        if with_proj:
            ShowLatentSpace(
                data_encoder,
                tit=f"Latent Space {PROJECTION.__name__} {fn[:-4]} {name_modifier}",
            )
            plt.savefig(
                f"plots/LS_{fn[:-4]}_{PROJECTION.__name__}{name_modifier}.png",
                dpi=400,
                facecolor="white",
                bbox_inches="tight",
            )
        else:
            ShowLatentSpace(
                data_encoder, tit=f"Latent Space {fn[:-4]} {name_modifier}",
            )
            plt.savefig(
                f"plots/LS_{fn[:-4]}{name_modifier}.png",
                dpi=400,
                facecolor="white",
                bbox_inches="tight",
            )
        if SHOW_LATENT_SPACE:
            plt.show()
        else:
            plt.close()


def save_metrics(res, df_softmax, name, seed):
    os.makedirs(results_dir + "metrics/", exist_ok=True)

    try:
        # try to complete an existing csv
        df_metrics = pd.read_csv(
            results_dir + "metrics/" + name + "_metrics.csv", sep=";", index_col="Seed"
        )
    except:
        df_metrics = pd.DataFrame(
            columns=["Accuracy", "AUC", "Precision", "Recall", "F1 Score"]
        )

    # convert from tensors to arrays
    y_true = res["y"].astype(int).to_numpy()
    y_score = df_softmax["Score class 1"].astype(float).to_numpy()
    y_pred = res["y_pred"].astype(int).to_numpy()

    df_metrics.at[seed, "Accuracy"] = sum(res["correct"]) / len(
        res
    )  # note: sum([True, False, True])=2
    df_metrics.at[seed, "AUC"] = roc_auc_score(y_true=y_true, y_score=y_score)
    (
        df_metrics.at[seed, "Precision"],
        df_metrics.at[seed, "Recall"],
        df_metrics.at[seed, "F1 Score"],
        _,
    ) = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average="macro")

    df_metrics.loc[f"Mean over all seeds"] = df_metrics.mean()  # update the avg

    print("\n" + name + " Metrics:")
    print(df_metrics.loc[seed])

    df_metrics.to_csv(
        results_dir + "metrics/" + name + "_metrics.csv", sep=";", index_label="Seed"
    )


#%%
# General labeling function, choosing between algorithms
def labeling_func(X_l, X_unl, y_l, y_unl, algo_name, sae_model_name=None, seed=None):
    if algo_name == "LabelPropagation":
        return label_sklearn(X_l, X_unl, y_l, y_unl, seed=seed, algo=LabelPropagation)
    elif algo_name == "LabelSpreading":
        return label_sklearn(X_l, X_unl, y_l, y_unl, seed=seed, algo=LabelSpreading)
    elif algo_name == "NN":
        return label_network(
            X_l,
            X_unl,
            y_l,
            y_unl,
            save_losses_file=losses_file_name,
            model_name="NN",
            autoencoder=False,
            seed=seed,
        )
    elif algo_name == "SAE":
        return label_network(
            X_l,
            X_unl,
            y_l,
            y_unl,
            save_losses_file=losses_file_name,
            model_name=sae_model_name,
            autoencoder=True,
            seed=seed,
        )
    else:
        raise ValueError(
            "Unsupported func_name. Supported names are 'LabelPropagation', 'LabelSpreading', 'NN', 'SAE'."
        )


def prog(iterable: Iterable):
    """Wrap an iterable in a progress bar if the boolean PROGRESS_BAR is set to True"""
    if PROGRESS_BAR:
        return tqdm(iterable)
    return iterable


#%%
def compute_labeling_result(algo_name, seed=6):
    """Computes the labeling result (accuracy) of labeling_func for different values of separability or unlabeled proportions.
    Takes care of data creation, label prediction.
    """
    print(f"\n{algo_name}:")

    # if using real data, ignore separability and test unlabeled proportions instead
    if file_name is not None:
        param_list = UNL_PROPS
    else:
        param_list = SEPARABILITIES

    accuracies = {"acc": [], "acc_swa": [], "acc_2nd": [], "acc_swa_2nd": []}
    for param in param_list:
        if file_name is not None:
            # vary unlabeled proportions
            X_l, X_unl, y_l, y_unl = get_data(
                file_name=file_name, unlabeled_prop=param, seed=seed
            )
        else:
            # vary separabilities
            X_l, X_unl, y_l, y_unl = get_data(
                file_name=file_name, separability=param, seed=seed
            )
        torch.manual_seed(seed)
        labeling_results = labeling_func(
            X_l,
            X_unl,
            y_l,
            y_unl,
            algo_name=algo_name,
            sae_model_name=f"SAE_{seed}s_{param}{'sep' if file_name is None else 'prop'}",
            seed=seed,
        )

        # general accuracy
        res = labeling_results["res"]
        acc = (res.correct == True).sum() / len(res)
        accuracies["acc"].append(acc)
        if file_name is not None:
            print(f"Unlabeled Prop {param} \t\t Accuracy : {acc:.4f}")
        else:
            print(f"Separability {param} \t\t Accuracy {acc:.4f}")

        if algo_name in ["NN", "SAE"]:
            # if network, then we also have a SWA result
            res_swa = labeling_results["res_swa"]
            acc_swa = (res_swa.correct == True).sum() / len(res)
            accuracies["acc_swa"].append(acc_swa)
            if file_name is not None:
                print(f"SWA : Unlabeled Prop {param} \t\t Accuracy : {acc_swa:.4f}")
            else:
                print(f"SWA : Separability {param} \t\t Accuracy {acc_swa:.4f}")
        if algo_name == "SAE" and PROJECTION is not None:
            # if SAE and projection, then we also have a 2nd descent result
            res_2nd = labeling_results["res_2nd"]
            acc_2nd = (res_2nd.correct == True).sum() / len(res)
            accuracies["acc_2nd"].append(acc_2nd)
            if file_name is not None:
                print(
                    f"2nd Descent : Unlabeled Prop {param} \t\t Accuracy : {acc_2nd:.4f}"
                )
            else:
                print(f"2nd Descent : Separability {param} \t\t Accuracy {acc_2nd:.4f}")
            res_swa_2nd = labeling_results["res_swa_2nd"]
            acc_swa_2nd = (res_swa_2nd.correct == True).sum() / len(res)
            accuracies["acc_swa_2nd"].append(acc_swa_2nd)
            if file_name is not None:
                print(
                    f"2nd Descent, SWA : Unlabeled Prop {param} \t\t Accuracy : {acc_swa_2nd:.4f}"
                )
            else:
                print(
                    f"2nd Descent, SWA : Separability {param} \t\t Accuracy {acc_swa_2nd:.4f}"
                )
    return accuracies


#%%
if __name__ == "__main__":
    import os

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    pd.DataFrame(index=list(range(N_EPOCHS))).to_csv(
        results_dir + losses_file_name, sep=";", index_label="epoch"
    )

    if resume:
        results = pd.read_csv(results_dir + output_name, index_col="Param", sep=";")
    else:
        if file_name is not None:
            results = pd.DataFrame(index=UNL_PROPS)
        else:
            results = pd.DataFrame(index=SEPARABILITIES)

    lp_accs = []
    ls_accs = []
    nn_accs = []
    nn_swa_accs = []
    sae_accs = []
    sae_swa_accs = []
    sae_accs_2nd = []
    sae_swa_accs_2nd = []
    for seed in SEEDS:
        print(f"--- Seed {seed} ---")

        # Perform experiment with LabelPropagation
        accs = compute_labeling_result("LabelPropagation", seed=seed)
        results[f"{seed}s_{N_FEATURES}f_LabProp"] = accs["acc"]
        lp_accs.append(accs["acc"])

        # Perform experiment with LabelSpreading
        accs = compute_labeling_result("LabelSpreading", seed=seed)
        results[f"{seed}s_{N_FEATURES}f_LabSpread"] = accs["acc"]
        ls_accs.append(accs["acc"])

        # Perform experiment with NN and SWA
        torch.manual_seed(seed)
        accs = compute_labeling_result("NN", seed=seed)
        results[f"{seed}s_{N_FEATURES}f_NN"] = accs["acc"]
        results[f"{seed}s_{N_FEATURES}f_NN_SWA"] = accs["acc_swa"]
        nn_accs.append(accs["acc"])
        nn_swa_accs.append(accs["acc_swa"])

        # Perform experiment with SAE and SWA
        torch.manual_seed(seed)
        accs = compute_labeling_result("SAE", seed=seed)
        results[f"{seed}s_{N_FEATURES}f_SAE"] = accs["acc"]
        results[f"{seed}s_{N_FEATURES}f_SAE_SWA"] = accs["acc_swa"]
        results[f"{seed}s_{N_FEATURES}f_SAE_2nd"] = accs["acc_2nd"]
        results[f"{seed}s_{N_FEATURES}f_SAE_SWA_2nd"] = accs["acc_swa_2nd"]
        sae_accs.append(accs["acc"])
        sae_swa_accs.append(accs["acc_swa"])
        sae_accs_2nd.append(accs["acc_2nd"])
        sae_swa_accs_2nd.append(accs["acc_swa_2nd"])

    results[f"Mean_{N_FEATURES}f_LabProp"] = np.array(lp_accs).mean(axis=0)
    results[f"Mean_{N_FEATURES}f_LabSpread"] = np.array(ls_accs).mean(axis=0)
    results[f"Mean_{N_FEATURES}f_NN"] = np.array(nn_accs).mean(axis=0)
    results[f"Mean_{N_FEATURES}f_NN_SWA"] = np.array(nn_swa_accs).mean(axis=0)
    results[f"Mean_{N_FEATURES}f_SAE"] = np.array(sae_accs).mean(axis=0)
    results[f"Mean_{N_FEATURES}f_SAE_SWA"] = np.array(sae_swa_accs).mean(axis=0)
    results[f"Mean_{N_FEATURES}f_SAE_2nd"] = np.array(sae_accs_2nd).mean(axis=0)
    results[f"Mean_{N_FEATURES}f_SAE_SWA_2nd"] = np.array(sae_swa_accs_2nd).mean(axis=0)

    results.to_csv(results_dir + output_name, sep=";", index_label="Param")

#%%
