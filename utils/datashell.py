
import torch
from torch.utils.data import Dataset

class HighDataset(Dataset):
    def __init__(self, source, predict_len=5, ds=5, scale_factor=1.0):
        self.predict_len = int(predict_len)
        self.ds = int(ds)
        self.scale_factor = float(scale_factor)
        self.source = [
            [i[0], i[1][: self.predict_len * self.ds]] + list(i[2:])
            for i in source
            if len(i[1]) >= self.predict_len * self.ds
        ]

    def split_val(self):
        from random import shuffle
        shuffle(self.source)
        val_len = len(self.source) // 8
        val = HighDataset([], predict_len=self.predict_len, ds=self.ds, scale_factor=self.scale_factor)
        tra = HighDataset([], predict_len=self.predict_len, ds=self.ds, scale_factor=self.scale_factor)
        val.source = self.source[:val_len]
        tra.source = self.source[val_len:]
        return tra,val
    
    def __len__(self):
        return len(self.source)
    
    @torch.no_grad()
    def __getitem__(self, idx):
        xi, futi, yi, ii, *extra = self.source[idx]
        xi = torch.tensor(xi, dtype=torch.float32)
        futi = torch.tensor(futi, dtype=torch.float32)
        yi = torch.tensor(yi,dtype=torch.long)
        ii = torch.tensor(ii,dtype=torch.long)
        
        Ap = xi[:,-13:].reshape((-1,13)) # [T,13] -> [T,13,3]
        A = torch.zeros((Ap.size(0),13,3))
        for i in range(Ap.size(0)):
            loc = Ap[i] #13
            for j,l in enumerate(loc):
                if l < 0 or l >= 3:
                    continue
                A[i,j,l.int().item()] = 1

        state = xi[:, :6]
        social = xi[:, 6:58]
        targets = futi
        if self.scale_factor != 1.0:
            state = state / self.scale_factor
            social = social / self.scale_factor
            targets = targets / self.scale_factor

        res = {
            "S": state,
            "I": social,
            "A": A,
            "Y": targets,
            "LC": yi,
            "LI": ii,
        }

        return res
