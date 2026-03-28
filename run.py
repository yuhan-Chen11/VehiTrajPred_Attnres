import torch
import pickle

from utils.datashell import HighDataset
from utils.instructor import EasyInstructor

from models.DLinear import DLinearAttnres
from models.FEDformer import FEDformerAttnres
from models.PatchTST import PatchTSTAttnRes
from models.TimeMixer import TimeMixerAttnres 


import argparse


def get_model(name, pred_len=5):
    name = name.lower()

    if name == "patchtst":
        return PatchTSTAttnRes(pred_len=pred_len)

    elif name == "fedformer":
        return FEDformerAttnres(pred_len=pred_len)

    elif name == "timemixer":
        return TimeMixerAttnres(pred_len=pred_len)

    elif name == "dlinear":
        return DLinearAttnres(pred_len=pred_len)

    else:
        raise ValueError(f"Unknown model: {name}")


def load_data(predict_len=5):
    # 数据存储路径
    train_path = "datasets/ngsim_train.pkl"
    test_path = "datasets/ngsim_test.pkl"

    with open(train_path, "rb") as f:
        train_raw = pickle.load(f)

    with open(test_path, "rb") as f:
        test_raw = pickle.load(f)

    train_ds = HighDataset(train_raw, predict_len=predict_len)
    train, val = train_ds.split_val()

    test_ds = HighDataset(test_raw, predict_len=predict_len)

    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="patchtst")
    parser.add_argument("--pred_len", type=int, default=5)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train, val, test = load_data(predict_len=args.pred_len)

    # 模型选择器
    model = get_model(args.model, pred_len=args.pred_len)

    trainer = EasyInstructor(
        net=model,
        train_dataset=train,
        val_dataset=val,
        test_dataset=test,
        device=device,
        epoch=30,
        lr=1e-4,
        batch_size=128,
    )

    trainer.fit()
    torch.save(model.state_dict(), f"trained_models/{args.model}.pth")

    metrics = EasyInstructor._test_model_(
        model,
        test,
        batch_size=128,
        device=device,
    )

    print("Test Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()


# python run.py --model patchtst