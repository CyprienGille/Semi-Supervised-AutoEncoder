import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.preprocessing import scale
from scipy.optimize import golden
from torch import cuda
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from functions.functions_semisupervise import (
    netBio,
    proj_l1ball,
    proj_l1infball,
    proj_l11ball,
)

#%%
## Script constants
results_dir = "results_semi/"
plots_dir = "plots/"
SEEDS = [9]
PROGRESS_BAR = True  # display progress bars for the training of the SAE


## Parameters for real data
# file_name = None  # To use synthetic data
file_name = "LUNG.csv"  # To use a real csv
# file_name = "IPFcellATLAS.csv"  # To use a real csv

UNLABELED_PROPORTION = 0.4  # default unlabeled proportion

## Parameters for synthetic data
SEPARABILITY = 0.8
N_SAMPLES = 1000
# Number of features of each type
N_FEATURES = 10000
# NB: N_INFORMATIVE = N_FEATURES - N_REDUNDANT - N_USELESS
N_REDUNDANT = 0
N_USELESS = 10000 - 8

## SAE params
N_EPOCHS = 40  # total number of epochs (Adam + SWA)
LEARNING_RATE = 1e-4  # Adam learning rate
LOSS_LAMBDA = 0.0005  # weight of the reconstruction loss in the total loss
BATCH_SIZE = 4
# SWA params
SWA_START = 33  # start of the Stochastic Weight Averaging
SWA_LR = 1e-5  # swa learning rate

# PROJECTION = None  # No projection
PROJECTION = proj_l11ball  # Projection L11
# PROJECTION = proj_l1ball     # Projection L1
# PROJECTION = proj_l1infball  # Projection L1inf

## ETA optimization params
GOLDEN = True  # do golden section strategy (else simple dichotomy)

ETA_MIN = 100  # initial lower bound
ETA_MAX = 2418  # initial upper bound
THRESH = 5  # convergence threshold for dichotomy

MAX_ITER = 10  # max number of iterations for golden section

# Create synthetic data
def get_data(
    file_name: str = None,
    unlabeled_prop: float = UNLABELED_PROPORTION,
    n_samples: int = N_SAMPLES,
    separability: float = SEPARABILITY,
    seed: int = 6,
):
    """Get or Make data, and split it into unlabeled and labeled sets.
    To make synthetic data, pass file_name = None.
    returns (X_labeled, X_unlabeled, y_labeled, y_unlabeled)
    """
    if file_name is None:
        # Note: changing the number of classes might break other parts of the code
        X, y = make_classification(
            n_samples=n_samples,
            n_features=N_FEATURES,
            n_informative=N_FEATURES - N_REDUNDANT - N_USELESS,
            n_redundant=N_REDUNDANT,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=1,
            # weights=[0.1, 0.9],  # to make an unbalanced dataset
            class_sep=separability,
            hypercube=True,
            random_state=seed,
        )

    else:
        # Format-specific data preparation steps
        df = pd.read_csv(f"data/{file_name}", sep=";", header=0).transpose()
        df.columns = df.loc["Name"]
        df.drop("Name", inplace=True)
        y = df["Label"].to_numpy() - 1  # Label in [1, 2] -> Label in [0, 1]
        X = df.drop("Label", axis=1).to_numpy()

        # Log transform
        X = np.log(abs(X.astype(np.float32)) + 1)

        # Mean centering, unit variance scaling
        X = X - np.mean(X, axis=0)
        X = scale(X, axis=0, with_mean=False)
        n_samples = len(y)

    np.random.seed(seed)  # seed for reproducibility
    # generate indexes of unlabeled points
    unlabeled_idx = np.random.randint(
        0, n_samples, size=int(n_samples * unlabeled_prop)
    )
    # get the rest of the indexes, which will represent labeled points
    labeled_idx = list(set(range(n_samples)).difference(set(unlabeled_idx)))

    # split data
    X_unlabeled, X_labeled = X[unlabeled_idx], X[labeled_idx]
    y_unlabeled, y_labeled = y[unlabeled_idx], y[labeled_idx]

    return X_labeled, X_unlabeled, y_labeled, y_unlabeled


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


def find_optimal_eta(X_l, y_l, X_unl, y_unl, seed):
    # initial training
    torch.manual_seed(seed)

    # prepare torch datasets and dataloaders
    l_dataset = SAE_Dataset(X_l, y_l)
    unl_dataset = SAE_Dataset(X_unl, y_unl)

    unl_dataloader = DataLoader(unl_dataset, shuffle=False, batch_size=1)
    l_dataloader = DataLoader(l_dataset, shuffle=True, batch_size=BATCH_SIZE)

    n_classes = len(np.unique(y_l))

    # get original model and train it (once)
    init_model, _ = get_model(n_inputs=X_l.shape[1], n_outputs=n_classes, seed=seed)

    print("Initial Training...")
    init_res, init_model = full_network_loop(
        init_model, l_dataloader, unl_dataloader, n_classes, N_EPOCHS
    )

    if GOLDEN:
        # second descents, golden ratio strategy
        minimum = golden(
            train_fixed_eta,
            args=(
                l_dataloader,
                unl_dataloader,
                n_classes,
                N_EPOCHS,
                init_model,
                X_l.shape[1],
            ),
            brack=(ETA_MIN, ETA_MAX),
            maxiter=MAX_ITER,
        )
        print(f"Best ETA: {minimum}")
    else:
        # second descents, dichotomy strategy
        eta_min = ETA_MIN
        eta_max = ETA_MAX

        # train the two models at the bounds of the dichotomy
        min_model, min_density = get_model(
            n_inputs=X_l.shape[1],
            n_outputs=n_classes,
            initial=False,
            prev_model=init_model,
            eta=eta_min,
        )
        min_res, _ = full_network_loop(
            min_model, l_dataloader, unl_dataloader, n_classes, N_EPOCHS
        )
        print(
            f"Training for ETA={eta_min} ({min_density:.4f}) completed : Accuracy={min_res['acc']}"
        )

        max_model, max_density = get_model(
            n_inputs=X_l.shape[1],
            n_outputs=n_classes,
            initial=False,
            prev_model=init_model,
            eta=eta_max,
        )
        max_res, _ = full_network_loop(
            max_model, l_dataloader, unl_dataloader, n_classes, N_EPOCHS
        )
        print(
            f"Training for ETA={eta_max} ({max_density:.4f}) completed : Accuracy={max_res['acc']}"
        )

        while eta_max - eta_min > THRESH:
            # Do dichotomy algorithm to find best ETA
            eta_new = (eta_min + eta_max) // 2
            new_model, new_density = get_model(
                n_inputs=X_l.shape[1],
                n_outputs=n_classes,
                initial=False,
                prev_model=init_model,
                eta=eta_new,
            )
            new_res, _ = full_network_loop(
                new_model, l_dataloader, unl_dataloader, n_classes, N_EPOCHS
            )

            print(
                f"Training for ETA={eta_new} ({new_density:.4f}) completed : Accuracy={new_res['acc']}"
            )
            if new_res["acc"] >= min_res["acc"]:
                eta_min = eta_new
                min_res = new_res
            elif new_res["acc"] >= max_res["acc"]:
                eta_max = eta_new
                max_res = new_res
            else:
                break


def train_fixed_eta(
    eta, l_dataloader, unl_dataloader, n_classes, n_epochs, init_model, n_inputs
):
    """Wrapper function for instanciating and training a model for a given ETA"""
    model, _ = get_model(
        n_inputs=n_inputs,
        n_outputs=n_classes,
        initial=False,
        prev_model=init_model,
        eta=eta,
    )
    result, _ = full_network_loop(
        model, l_dataloader, unl_dataloader, n_classes, n_epochs
    )
    print(f"ETA={eta:.0f} --- Accuracy={result['acc']}")
    return 1 - result["acc"]


def get_model(
    n_inputs, n_outputs, seed=6, initial=True, prev_model=None, eta=None,
):
    """Initialize a model, either initial or projected"""
    torch.manual_seed(seed)

    if initial:
        return netBio(n_inputs, n_outputs).to(DEVICE), 1.0

    else:
        assert prev_model is not None  # we need a previously trained model
        mask = compute_mask(prev_model, PROJECTION, projection_param=eta)

        # calculate the proportion of non-zero weights
        density = torch.sum(torch.flatten(mask)) / torch.prod(torch.tensor(mask.shape))

        model = netBio(n_inputs, n_outputs)
        # store the mask as a non-trainable model parameter
        model.register_buffer(name="mask", tensor=mask)
        # register the masking function as a pre-forward call
        model.register_forward_pre_hook(mask_gradient)
        model = model.to(DEVICE)
        return model, density


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


def full_network_loop(
    model, l_dataloader, unl_dataloader, n_classes, n_epochs,
):
    model.train()
    c_loss = nn.CrossEntropyLoss(reduction="sum")
    r_loss = nn.SmoothL1Loss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    swa_sched = SWALR(optimizer, swa_lr=SWA_LR, anneal_strategy="linear")

    swa_model = AveragedModel(model, device=DEVICE)

    # train for n epochs over X_l
    for i in tqdm(range(n_epochs)):
        for batch in l_dataloader:
            x, lab = batch
            x = x.to(DEVICE)
            lab = lab.to(DEVICE)

            # SAE
            net_out, decoder_out = model(x)
            loss = c_loss(net_out, lab.long()) + LOSS_LAMBDA * r_loss(decoder_out, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i >= SWA_START:
            swa_model.update_parameters(model)
            swa_sched.step()

    # predict over X_unl
    acc = get_accuracy(model, unl_dataloader, DEVICE)
    acc_swa = get_accuracy(swa_model, unl_dataloader, DEVICE)
    return {"acc": acc, "acc_swa": acc_swa,}, model


@torch.no_grad()
def get_accuracy(model, dataloader, device=DEVICE):
    model.eval()
    res = pd.DataFrame(columns=["correct"])
    for i, batch in enumerate(dataloader):
        x, lab = batch
        x = x.to(device)
        lab = lab.to(device)
        encoder_out, _ = model(x)

        pred_label = np.argmax(encoder_out.detach().cpu().numpy())
        res.at[i, "correct"] = pred_label == int(lab.cpu().item())
    return (res.correct == True).sum() / len(res)


if __name__ == "__main__":
    for seed in SEEDS:
        X_l, X_unl, y_l, y_unl = get_data(file_name, seed=seed)
        find_optimal_eta(X_l, y_l, X_unl, y_unl, seed)

