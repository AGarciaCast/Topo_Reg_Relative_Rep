"""
Modified from https://github.com/c-hofer/topologically_densified_distributions
"""

import torch
from collections import defaultdict
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
import random

class DictDataset(torch.utils.data.Dataset):
    def __init__(self, ds, data_key="content", target_key="class"):
        
        self.wrappee = ds
        self.data_key = data_key
        self.target_key = target_key
    
    
    def __len__(self):
        return len(self.wrappee)

    def __getitem__(self, idx):
        it = self.wrappee[idx]
        return it[self.data_key], it[self.target_key]


class IntraLabelMultiDraw(torch.utils.data.Dataset):
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
    def __init__(self, ds, batch_size, accumulation=1, drop_last=True, group_cls=True, indv=False, main_random=False):
        self.ds = ds
        self.main_random = main_random
        if main_random:
            self.ds_loader = BatchSampler(RandomSampler(ds),
                                          batch_size=batch_size,
                                          drop_last=drop_last)
        self.accumulation = accumulation 
        self.group_cls = group_cls
        self.inbatch_size = batch_size
        self.indv = indv

        self.indices_by_label = defaultdict(list)
        for i in range(len(ds)):
            _, y = ds[i]
            y = int(y)
            self.indices_by_label[y].append(i)

        self.samplers = {}
        self.max_lab = 0
        self.batches_idx = []
        self.num_cls = len(self.indices_by_label.keys())
        for k, v in self.indices_by_label.items():

            self.samplers[k] = DataLoader(v,
                                        batch_size=batch_size,
                                        drop_last=drop_last)

            self.batches_idx += [k]*len(self.samplers[k])

            if len(self.samplers[k]) > self.max_lab:
                self.max_lab = len(self.samplers[k]) 

        self.num_batches = None

    def __iter__(self):
        if self.group_cls:
            self.batches_idx = []
            for i in range(self.max_lab):
                aux_list = []
                for k in self.samplers.keys():
                    if i < len(self.samplers[k]):
                        aux_list.append(k)

                random.shuffle(aux_list)
                self.batches_idx += aux_list
        else:
            random.shuffle(self.batches_idx) 
        
        if self.main_random:
            random_sampler = iter(self.ds_loader)
        
        batch = []
        batch_iters = {k: iter(b) for k,b in self.samplers.items()}
        aux = 0
        for i, k in enumerate(self.batches_idx):
            if i>0 and i%self.accumulation == 0:
                if self.indv:
                    if self.main_random:
                        yield next(random_sampler)
                    else:
                        yield batch[aux*self.inbatch_size:(aux+1)*self.inbatch_size]

                for j in range(self.accumulation):
                    yield batch[j*self.inbatch_size:(j+1)*self.inbatch_size]
                
                aux =(aux+1)%self.num_cls
                batch = []

            batch += next(batch_iters[k]).tolist()

        if len(batch)==self.inbatch_size*self.accumulation:
            if self.indv:
                if self.main_random:
                    yield next(random_sampler)
                else:
                    yield batch[aux*self.inbatch_size:(aux+1)*self.inbatch_size]
                
            for j in range(self.accumulation):
                yield batch[j*self.inbatch_size:(j+1)*self.inbatch_size]

    def __len__(self) -> int:
        if self.num_batches is None:
            self.num_batches = sum(1 for _ in self.__iter__())
        return self.num_batches


class DoubleClassAccumulationSampler():
    def __init__(self, ds, batch_size, drop_last=True, main_random=False):
        self.ds = ds
        self.main_random = main_random
        if main_random:
            self.ds_loader = BatchSampler(RandomSampler(ds),
                                          batch_size=batch_size,
                                          drop_last=drop_last)
        self.batch_size = batch_size
        self.num_batches = None
        self.indices_by_label = defaultdict(list)
        for i in range(len(ds)):
            _, y = ds[i]
            y = int(y)
            self.indices_by_label[y].append(i)

        self.samplers = {}
        self.max_lab = 0
        self.batches_idx = []
        self.num_cls = len(self.indices_by_label.keys())
        for k, v in self.indices_by_label.items():

            self.samplers[k] = DataLoader(v,
                                          batch_size=batch_size,
                                          drop_last=drop_last)

            self.batches_idx += [k]*len(self.samplers[k])

            if len(self.samplers[k]) > self.max_lab:
                self.max_lab = len(self.samplers[k]) 


    def __iter__(self):

        self.batches_idx = []
        for i in range(self.max_lab):
            aux_list = []
            for k in self.samplers.keys():
                if i < len(self.samplers[k]):
                    aux_list.append(k)

            self.batches_idx += aux_list
        
        if self.main_random:
            random_sampler = iter(self.ds_loader)
            
        batch = []
        batch_iters = {k: iter(b) for k,b in self.samplers.items()}
        for i, k in enumerate(self.batches_idx):
            if i>0 and i%self.num_cls == 0:
                og_batch = batch.copy()
                random.shuffle(batch)
                for j in range(self.num_cls):
                    if self.main_random:
                        yield next(random_sampler)
                    else:
                        yield batch[j*self.batch_size:(j+1)*self.batch_size]
                        
                for j in range(self.num_cls):
                    yield og_batch[j*self.batch_size:(j+1)*self.batch_size]

                batch = []

            batch += next(batch_iters[k]).tolist()

        if len(batch)==self.batch_size*self.num_cls:
            og_batch = batch.copy()
            random.shuffle(batch)
            for j in range(self.num_cls):
                if self.main_random:
                    yield next(random_sampler)
                else:
                    yield batch[j*self.batch_size:(j+1)*self.batch_size]
                    
            for j in range(self.num_cls):
                yield og_batch[j*self.batch_size:(j+1)*self.batch_size]

    def __len__(self) -> int:
        if self.num_batches is None:
            self.num_batches = sum(1 for _ in self.__iter__())
        return self.num_batches


