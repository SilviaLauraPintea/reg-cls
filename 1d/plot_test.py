import argparse
import pickle
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import math


def plot_predictions(
    args, alloutput_reg, alloutput_regcls, alldata, alltarget, num_bins=64
):
    print(alloutput_reg.shape, alloutput_regcls.shape, alldata.shape, alltarget.shape)
    alloutput_reg = (alloutput_reg).reshape(
        -1,
    )
    alloutput_regcls = (alloutput_regcls).reshape(
        -1,
    )
    alldata = (alldata).reshape(
        -1,
    )
    alltarget = (alltarget).reshape(
        -1,
    )
    print(alloutput_reg.shape, alloutput_regcls.shape, alldata.shape, alltarget.shape)

    plt.rcParams.update({"font.size": 6})

    fig, (ax) = plt.subplots(2)
    fig.tight_layout(pad=5.0)
    ax = ax.flatten()

    bins = np.linspace(alltarget.max(), alltarget.min(), num_bins + 1)
    alphas = np.linspace(0.2, 0.8, num_bins)

    for i in range(0, num_bins):
        ax[0].axhspan(bins[i], bins[i + 1], facecolor="gray", alpha=alphas[i])
        ax[1].axhspan(bins[i], bins[i + 1], facecolor="gray", alpha=alphas[i])
    idx = np.random.randint(0, alldata.shape[0], 1000).tolist()

    # Scatter data
    ax[0].scatter(alldata[idx], alltarget[idx], s=20, alpha=0.1, label="GT")
    ax[0].scatter(
        alldata[idx], alloutput_reg[idx], s=20, alpha=0.2, color="r", label="reg"
    )
    ax[0].set_ylim(alltarget.min(), alltarget.max())
    ax[0].set_xlim(alldata.min(), alldata.max())
    ax[0].set_title(f"Regression")
    ax[0].set_ylabel("f(x)")
    ax[0].set_xlabel("x")
    ax[0].legend()

    # Scatter data
    ax[1].scatter(alldata[idx], alltarget[idx], s=20, alpha=0.1, label="GT")
    ax[1].scatter(
        alldata[idx], alloutput_regcls[idx], s=20, alpha=0.2, color="g", label="reg+cls"
    )
    ax[1].set_ylim(alltarget.min(), alltarget.max())
    ax[1].set_xlim(alldata.min(), alldata.max())
    ax[1].set_title(f"Regression+Classification")
    ax[1].set_ylabel("f(x)")
    ax[1].set_xlabel("x")
    ax[1].legend()
    plt.savefig(os.path.join("logs", args.name + ".pdf"))


def get_stats(x_struct, num_cls, num_fct, num_seeds, log_file):
    # Check that all runs are complete
    for f in range(0, num_fct):
        for c in range(0, num_cls):
            assert num_seeds == x_struct["count"][c][f], (
                "Log file " + log_file + " is incomplete."
            )

    # The size is: ( num_cls x num_fct x num_seeds] )
    x_struct["val_reg_stats"] = [
        np.mean(np.mean(x_struct["val_reg"], axis=2), axis=1),
        np.std(np.std(x_struct["val_reg"], axis=2), axis=1),
    ]

    # The size is: ( num_cls x num_fct x num_seeds] )
    x_struct["val_regcls_stats"] = [
        np.mean(np.mean(x_struct["val_regcls"], axis=2), axis=1),
        np.std(np.std(x_struct["val_regcls"], axis=2), axis=1),
    ]

    x_struct["gap"] = np.abs(
        x_struct["val_reg_stats"][0] - x_struct["val_regcls_stats"][0]
    ).mean()
    return x_struct


def read_data(x_struct, a_row):
    # Find the class position
    classes_pos = {4: 0, 16: 1, 64: 2, 256: 3, 1024: 4}
    cls = int(classes_pos[a_row["cls"]])

    # Find the function index
    pos = a_row["name"].find("fct")
    fct = int(a_row["name"][pos + 3 : pos + 4])

    # Get the current count
    count = int(x_struct["count"][cls][fct])

    # Get the loss values
    x_struct["val_reg"][cls][fct][count] = float(a_row["val_reg_mse"])
    x_struct["val_regcls"][cls][fct][count] = float(a_row["val_regcls_mse"])
    x_struct["count"][cls][fct] += 1

    return x_struct


def load_data(log_file, sampling, num_cls=5, num_seeds=5, num_fct=10):
    if "uniform" in sampling:
        x_uniform = {
            "name": "uniform",
            "count": np.zeros((num_cls, num_fct)),
            "val_reg": np.zeros((num_cls, num_fct, num_seeds)),
            "val_regcls": np.zeros((num_cls, num_fct, num_seeds)),
        }
    if "mild" in sampling:
        x_mild = {
            "name": "mild",
            "count": np.zeros((num_cls, num_fct)),
            "val_reg": np.zeros((num_cls, num_fct, num_seeds)),
            "val_regcls": np.zeros((num_cls, num_fct, num_seeds)),
        }
    if "moderate" in sampling:
        x_moderate = {
            "name": "moderate",
            "count": np.zeros((num_cls, num_fct)),
            "val_reg": np.zeros((num_cls, num_fct, num_seeds)),
            "val_regcls": np.zeros((num_cls, num_fct, num_seeds)),
        }
    if "severe" in sampling:
        x_severe = {
            "name": "severe",
            "count": np.zeros((num_cls, num_fct)),
            "val_reg": np.zeros((num_cls, num_fct, num_seeds)),
            "val_regcls": np.zeros((num_cls, num_fct, num_seeds)),
        }
    with open(log_file, "rb") as f:
        while True:
            try:
                a_row = pickle.load(f)
            except:
                break

            # Try to read stuff
            if a_row["name"].startswith("uniform") and "uniform" in sampling:
                x_uniform = read_data(x_uniform, a_row)
            elif a_row["name"].startswith("mild") and "mild" in sampling:
                x_mild = read_data(x_mild, a_row)
            elif a_row["name"].startswith("moderate") and "moderate" in sampling:
                x_moderate = read_data(x_moderate, a_row)
            elif a_row["name"].startswith("severe") and "severe" in sampling:
                x_severe = read_data(x_severe, a_row)

    all_data = []
    if "uniform" in sampling:
        all_data.append(get_stats(x_uniform, num_cls, num_fct, num_seeds, log_file))
    if "mild" in sampling:
        all_data.append(get_stats(x_mild, num_cls, num_fct, num_seeds, log_file))
    if "moderate" in sampling:
        all_data.append(get_stats(x_moderate, num_cls, num_fct, num_seeds, log_file))
    if "severe" in sampling:
        all_data.append(get_stats(x_severe, num_cls, num_fct, num_seeds, log_file))

    return all_data


def plot_a_dataN(ax, data_struct, idx, setting, testname):
    # Define the class names
    classes = [4, 16, 64, 256, 1024]

    # Define the line colors
    sorted_names = ["forestgreen", "lime"]

    # The samplings
    count_samplings = {"severe": 0, "moderate": 0, "mild": 0, "uniform": 0}

    min_reg = np.ones((len(list(count_samplings.keys())),len(classes)))*math.inf
    min_regcls = np.ones((len(list(count_samplings.keys())),len(classes)))*math.inf

    print(min_reg.shape)

    # Loop over values of lambda
    for (j, skey) in enumerate(list(data_struct.keys())):
        spl_struct = data_struct[skey]

        # Loop over samplings
        for (i, a_struct) in enumerate(spl_struct):
            x = classes
                      
            min_reg[i,:] = np.minimum(min_reg[i,:], a_struct["val_reg_stats"][0])
            min_regcls[i,:] = np.minimum(min_regcls[i,:], a_struct["val_regcls_stats"][0])

            # Normal printing 
            y = a_struct["val_reg_stats"][0]
            error = a_struct["val_reg_stats"][1]
            ax[i].plot(x, y, 
                label="reg" + skey, 
                alpha=1, 
                color="r"
            )
            ax[i].fill_between(x, y - error, y + error, alpha=0.1, facecolor="r")

            # And plot all the rest
            y = a_struct["val_regcls_stats"][0]
            error = a_struct["val_regcls_stats"][1]
            ax[i].plot(
                x,
                y,
                color=sorted_names[(j) % len(sorted_names)],
                label="reg+cls "+ skey,
                alpha=1
            )
            ax[i].fill_between(
                x,
                y - error,
                y + error,
                alpha=0.1,
                facecolor=sorted_names[(j) % len(sorted_names)],
                linestyle="dashdot",
            )
                
        
            ax[i].set_xscale("log", base=2)
            ax[i].legend()
            ax[i].set_xticks(x)
            ax[i].set_ylim(0.0, 1.0)
            ax[i].set_xlim(4, 1024)
            ax[i].set_title(a_struct["name"] + " " + setting)
            ax[i].set_ylabel("MSE")
            ax[i].set_xlabel("# classes")

            # Counting how many this is over 5 classes
            mse_reg = a_struct["val_reg"].reshape(
                -1,
            )
            mse_regcls = a_struct["val_regcls"].reshape(
                -1,
            )
            where = np.where(mse_reg > mse_regcls)
            prob = float(where[0].size) / (a_struct["val_reg"].size)
            count_samplings[a_struct["name"]] += prob / len(list(data_structs.keys()))

    print("Ratio of exceeding the baseline:", count_samplings)
    return ax


def plot_all_dataN(
    data_structs,
    outdir,
    settings,
    sampling,
    outfile,
    testname,
):
    plt.rcParams.update({"font.size": 6})

    fig, (ax) = plt.subplots(len(settings), len(sampling))
    fig.tight_layout(pad=5.0)
    ax = np.atleast_2d(ax).reshape(len(settings), len(sampling))

    # Loop over data settings
    for (i, aset) in enumerate(settings):
        ax[i, :] = plot_a_dataN(
            ax[i, :],
            data_structs[aset],
            idx=i,
            setting=aset,
            testname=testname,
        )

    leg = plt.legend(labels=["reg","untrained","reg_cls"])
    leg.legendHandles[0].set_color('red') 
    leg.legendHandles[0].set_alpha(1)                                                        
    leg.legendHandles[1].set_color('purple')    
    leg.legendHandles[1].set_alpha(.2)                                                        
    leg.legendHandles[2].set_color('green') 
    leg.legendHandles[2].set_alpha(1) 

    plt.savefig(
        os.path.join(outdir, outfile),
        format="pdf",
        pad_inches=-2,
        transparent=False,
        dpi=300,
    )
    plt.show()
    plt.close()


##################################################################################
##################################################################################
##################################################################################
def list_of_strings(arg):
    return arg.split(",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        default="logs",
        help="The path to the pickled results, and the output plot.",
    )
    parser.add_argument(
        "--outfile", type=str, default="plot.pdf", help="Plot file name."
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=5,
        help="Number of random seeds used: 5 for test, 3 for validation.",
    )
    parser.add_argument(
        "--num_cls", type=int, default=5, help="Number of classes used."
    )
    parser.add_argument(
        "--num_fct", type=int, default=10, help="Number of functions used."
    )
    parser.add_argument(
        "--sampling",
        type=list_of_strings,
        default=["uniform", "mild", "moderate", "severe"],
    )
    parser.add_argument(
        "--settings",
        type=list_of_strings,
        default=[
            "clean",
            "noisey_0.1",
            # "noisey_0.5",
            # "noisey_0.05",
            "ood",
        ],
    )
    args = parser.parse_args()
    testname = "weight-test"

    # Per data setting a best lambda value:
    setting_logfiles = {}
    setting_logfiles["clean"] = {
        "1e+2": "test_lambda_1e+2.pkl",
        # "1e+2 bal": "test_lambda_1e+2_bal.pkl",
    }
    setting_logfiles["noisey_0.1"] = {
        "1e+3": "test_lambda_1e+3.pkl",
        # "1e+3 bal": "test_lambda_1e+3_bal.pkl",
    }
    setting_logfiles["ood"] = {
        "1e+4": "test_lambda_1e+4.pkl",
        # "1e+4 bal": "test_lambda_1e+4_bal.pkl",
    }
    logfiles = setting_logfiles

    args.outfile = testname + "_" + args.outfile
    data_structs = {}

    # Loop over data settings
    for (i, aset) in enumerate(args.settings):
        data_structs[aset] = {}

        # Loop over lambda's
        for (j, lkey) in enumerate(list(logfiles[aset].keys())):

            data_struct = load_data(
                os.path.join(os.path.join("logs", aset), logfiles[aset][lkey]),
                sampling=args.sampling,
                num_cls=args.num_cls,
                num_seeds=args.num_seeds,
                num_fct=args.num_fct,
            )
            data_structs[aset][lkey] = data_struct
            print("data[", aset, "][", lkey, "]:", data_structs[aset][lkey])

    plot_all_dataN(
        data_structs,
        outdir=args.outdir,
        settings=args.settings,
        sampling=args.sampling,
        outfile=args.outfile,
        testname=testname,
    )
