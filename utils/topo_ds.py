
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
 

class ClassAccumulationSampler():
    def __init__(self, ds, inbatch_size, accumulation=1, drop_last=True):
        self.accumulation = accumulation 

        self.indices_by_label = defaultdict(list)
        for i in range(len(ds)):
            _, y = ds[i]
            y = int(y)
            self.indices_by_label[y].append(i)
        
        self.samplers = {}
        self.batches_idx = []
        for k, v in self.indices_by_label.items():
          
          self.samplers[k] = DataLoader(v,
                                        batch_size=inbatch_size,
                                        drop_last=drop_last)
          
          self.batches_idx += [k]*len(self.samplers[k])
          
        
    def __iter__(self):
        random.shuffle(self.batches_idx)
        batch = []
        batch_iters = {k: iter(b) for k,b in self.samplers.items()}
        for i, k in enumerate(self.batches_idx):
          if i>0 and i%self.accumulation == 0:
            yield batch
            batch = []
          
          batch += next(batch_iters[k])
    
    def __len__(self) -> int:
      return len(self.batches_idx)//self.accumulation
