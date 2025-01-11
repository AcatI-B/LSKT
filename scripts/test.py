import os
import json
from argparse import ArgumentParser
import torch
import tomlkit
from tqdm import tqdm
import sys
sys.path.append('/home/q22301200/current/KT_module/LSKT-main')
from LSKT.data import KTData
from LSKT.eval import Evaluator

DATA_DIR = "/home/q22301200/current/KT_module/LSKT-main/data"

# configure the main parser
parser = ArgumentParser()

# general options
parser.add_argument("--device", help="device to run network on", default="cuda")
parser.add_argument("-bs", "--batch_size", help="batch size", default=32, type=int)
parser.add_argument("-tbs", "--test_batch_size", help="test batch size", default=32, type=int)

# data setup
datasets = tomlkit.load(open(os.path.join(DATA_DIR, "datasets.toml")))
parser.add_argument("-d","--dataset",help="choose from a dataset",choices=datasets.keys(),required=True,default = "[assist09]")
parser.add_argument("-p", "--with_pid", help="provide model with pid", action="store_true")

# model setup
parser.add_argument("-m", "--model", default = "LSKT", help="choose model")
parser.add_argument("--d_model", help="model hidden size", type=int, default=128)
parser.add_argument("--n_layers", help="number of layers", type=int, default= 3)
parser.add_argument("--n_heads", help="number of heads", type=int, default=8)
parser.add_argument("--n_know", help="dimension of knowledge parameter", type=int, default=32)
parser.add_argument("--dropout", help="dropout rate", type=float, default=0.2)

# test setup
parser.add_argument("-n", "--n_epochs", help="training epochs", type=int, default=100)
parser.add_argument("--lambda", help="CL loss weight", type=float, default=0.1, dest="lambda_cl")
parser.add_argument("-emb", "--emb_method", default = "3pl", help="choose emb_method")

# snapshot setup
parser.add_argument("-o", "--output_dir", default="/home/q22301200/current/KT_module/LSKT-main/result", help="directory to save model files and logs")
parser.add_argument("-f", "--from_file", help="resume training from existing model file", default=None)

# training logic
def main(args):
    # prepare dataset
    dataset = datasets[args.dataset]
    seq_len = dataset["seq_len"] if "seq_len" in dataset else None
    valid_data = KTData(
        os.path.join(
            DATA_DIR, dataset["valid"] if "valid" in dataset else dataset["test"]
        ),
        dataset["inputs"],
        seq_len=seq_len,
        batch_size=args.test_batch_size,
    )

    # prepare logger and output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        config_path = os.path.join(args.output_dir, "config.json")
        json.dump(vars(args), open(config_path, "w"), indent=2)
    else:
        # TODO: no persistency warning
        pass

    # prepare model and optimizer
    from our_method.LSKT import LSKT
    model = LSKT(
        dataset["n_questions"],
        dataset["n_pid"],
        d_model=args.d_model,
        n_heads=args.n_heads,
        dropout=args.dropout,
        device = args.device,
        batch_size = args.batch_size,
    )


    for epoch in range(1, args.n_epochs + 1):
        # validation
        model.load_state_dict(torch.load(args.from_file, map_location=lambda s, _: s), strict=False)
        model.to(args.device)
        model.eval()
        evaluator = Evaluator()
        with torch.no_grad(): 
            it = tqdm(iter(valid_data))
            for batch in it:
                if args.with_pid:
                    q, s, pid = batch.get("q", "s", "pid")
                else:
                    q, s = batch.get("q", "s")
                    pid = None if seq_len is None else [None] * len(q)
                if seq_len is None:
                    q, s, pid= [q], [s], [pid]
                for q, s, pid in zip(q, s, pid):
                    q = q.to(args.device)
                    s = s.to(args.device)
                    if pid is not None:
                        pid = pid.to(args.device)
                    y, *_ = model.predict(q, s, pid)
                    evaluator.evaluate(s, torch.sigmoid(y))
        r = evaluator.report()
        print(r)
    return best_epoch, best


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    best_epoch, best = main(args)
    print(args)
    print("best epoch:", best_epoch)
    print("best result", {k: f"{v:.4f}" for k, v in best.items()})
