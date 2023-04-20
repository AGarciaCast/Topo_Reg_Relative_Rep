
import torch
from collections import defaultdict

class DynamicDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, wrappee):
        super().__init__()
        self.wrappee = wrappee

    def __getattr__(self, name):
        return getattr(self.__dict__['wrappee'], name)

    def __len__(self):
        return len(self.wrappee)

    def __getitem__(self, idx):
        return self.wrappee[idx]


class DictDataset(DynamicDatasetWrapper):
    def __init__(self, ds, data_key="content", target_key="class"):
        
        self.wrappee = ds
        self.data_key = data_key
        self.target_key = target_key
    
    
    def __len__(self):
        return len(self.wrappee)

    def __getitem__(self, idx):
        it = self.wrappee[idx]
        return it[self.data_key], it[self.target_key]
    

class IntraLabelMultiDraw(DynamicDatasetWrapper):
    def __init__(self, ds, num_draws):

        self.wrappee = ds
        self.num_draws = int(num_draws)
        assert self.num_draws > 0

        self.indices_by_label = defaultdict(list)
        for i in range(len(ds)):
            _, y = ds[i]
            y = int(y)
            self.indices_by_label[y].append(i)

    def __getitem__(self, idx):
        x, y = self.wrappee[idx]

        N = len(self.indices_by_label[y])
        I = torch.randint(N, (self.num_draws - 1,))
        I = [self.indices_by_label[y][i] for i in I]
        x = [x]

        for i in I:
            x_i, _ = self.wrappee[i]
            x.append(x_i)

        return x, y

    def __len__(self):
        return len(self.wrappee)
