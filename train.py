# +
import random
from pathlib import Path
import numpy as np
import pytorch_lightning as pl

import pandas as pd
from IPython.display import clear_output
import matplotlib.pyplot as plt
from utils.topo_ds import *
import seaborn as sns 
from utils.pershom import pers2fn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, GradientAccumulationScheduler
from pl_modules.pl_roberta import LitRelRoberta
from ruamel.yaml import YAML

from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error
from utils.sim_rel import compare_topo_models, topo_model, compare_CKA_models
from functools import partial

from utils.export_multilingual_results import process_df
from utils.multilingual_amazon_anchors import MultilingualAmazonAnchors
from typing import *

from modules.stitching_module import StitchingModule

from datasets import load_dataset, ClassLabel, Dataset, DatasetDict

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Tensorboard extension (for visualization purposes later)
# %load_ext tensorboard

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = Path("./data")
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = Path("./saved_models/rel_multi_topo")
CHECKPOINT_PATH_VANILLA = Path("./saved_models/rel_multi_vanilla")
RESULT_PATH = Path("./results/rel_multi_topo")
FIG_PATH =  Path("./Visualization")

PROJECT_ROOT = Path("./")

pd.options.display.max_columns = None
pd.options.display.max_rows = None

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

fine_grained: bool = True
target_key: str = "class"
data_key: str = "content"
anchor_dataset_name: str = "amazon_translated"  
ALL_LANGS = ("en", "fr")
num_anchors: int = 768
train_perc: float = 0.01
    
def get_dataset(lang: str, split: str, perc: float, fine_grained: bool):
    pl.seed_everything(42)
    assert 0 < perc <= 1
    
    # Amazon removed the public dataset :(
    # dataset = load_dataset("amazon_reviews_multi", lang)[split]
    
    # Using files from https://www.kaggle.com/datasets/mexwell/amazon-reviews-multi
    # Read CSV files into pandas DataFrames
    
    split_df = pd.read_csv(DATASET_PATH / f"multi/{split}.csv", engine='python')
    try:
        split_df = pd.read_csv(DATASET_PATH / f"multi/{split}.csv", engine='python')
    except:
        raise Exception("Download files from https://www.kaggle.com/datasets/mexwell/amazon-reviews-multi and add them to ./data/multi/")
    
    # Filter by language
    lang_split_df = split_df[split_df['language'] == lang]
    
    # Create Hugging Face datasets
    dataset = Dataset.from_pandas(lang_split_df)
    dataset = dataset.remove_columns(['Unnamed: 0', '__index_level_0__'])

    if not fine_grained:
        dataset = dataset.filter(lambda sample: sample["stars"] != 3)

    # Select a random subset
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[: int(len(indices) * perc)]
    dataset = dataset.select(indices)

    def clean_sample(sample):
        title: str = sample["review_title"].strip('"').strip(".").strip()
        body: str = sample["review_body"].strip('"').strip(".").strip()

        if body.lower().startswith(title.lower()):
            title = ""

        if len(title) > 0 and title[-1].isalpha():
            title = f"{title}."

        sample["content"] = f"{title} {body}".lstrip(".").strip()
        if fine_grained:
            sample[target_key] = str(sample["stars"] - 1)
        else:
            sample[target_key] = sample["stars"] > 3
        return sample

    dataset = dataset.map(clean_sample)
    dataset = dataset.cast_column(
        target_key,
        ClassLabel(num_classes=5 if fine_grained else 2, names=list(map(str, range(5) if fine_grained else (0, 1)))),
    )

    return dataset

def _amazon_translated_get_samples(lang: str, sample_idxs):
    anchor_dataset = MultilingualAmazonAnchors(split="train", language=lang)
    anchors = []
    for anchor_idx in sample_idxs:
        anchor = anchor_dataset[anchor_idx]
        anchor[data_key] = anchor["data"]
        anchors.append(anchor)
    return anchors

train_datasets = {
    lang: get_dataset(lang=lang, split="train", perc=train_perc, fine_grained=fine_grained) for lang in ALL_LANGS
    }

test_datasets = {
    lang: get_dataset(lang=lang, split="test", perc=1, fine_grained=fine_grained) for lang in ALL_LANGS
    }

val_datasets = {
    lang: get_dataset(lang=lang, split="validation", perc=1, fine_grained=fine_grained) for lang in ALL_LANGS
    }

num_labels = len(np.unique(train_datasets["en"][target_key], return_counts=True)[0])
print("Num labels:", num_labels)

assert len(set(frozenset(train_dataset.features.keys()) for train_dataset in train_datasets.values())) == 1
class2idx = train_datasets["en"].features[target_key].str2int

train_datasets["en"].features

anchor_dataset2num_samples = 1000
anchor_dataset2first_anchors = [
        776,
        507,
        895,
        922,
        33,
        483,
        85,
        750,
        354,
        523,
        184,
        809,
        418,
        615,
        682,
        501,
        760,
        49,
        732,
        336,
    ]


assert num_anchors <= anchor_dataset2num_samples

pl.seed_everything(42)
anchor_idxs = list(range(anchor_dataset2num_samples))
random.shuffle(anchor_idxs)
anchor_idxs = anchor_idxs[:num_anchors]

assert anchor_idxs[:20] == anchor_dataset2first_anchors  # better safe than sorry
lang2anchors = {
    lang: _amazon_translated_get_samples(lang=lang, sample_idxs=anchor_idxs) for lang in ALL_LANGS
}

def collate_fn(batch, tokenizer, cls=True):
    data = []
    labels = []
    
    for x, y in batch:
        data.append(x)
        if cls:
            labels.append(y)

    encoding = tokenizer(
        data,
        return_tensors="pt",
        return_special_tokens_mask=True,
        truncation=True,
        max_length=512,
        padding=True,
    )
    del encoding["special_tokens_mask"]

    if cls:
        result = (encoding, torch.tensor(labels))
    else:
        result = encoding

    return  result


def multi_draw_collate_fn(batch, tokenizer, cls=True):
    data = []
    labels = []    
    for x, y in batch:
        data += x 
        if cls:
            labels += [y]*len(x)

    encoding = tokenizer(
        data,
        return_tensors="pt",
        return_special_tokens_mask=True,
        truncation=True,
        max_length=512,
        padding=True,
    )
    del encoding["special_tokens_mask"]

    if cls:
        result = (encoding, torch.tensor(labels))
    else:
        result = encoding

    return  result

lang2transformer_name = {
    "en": "roberta-base",
    "fr": "ClassCat/roberta-base-french",
    #"es": "PlanTL-GOB-ES/roberta-base-bne",
    #"ja": "nlp-waseda/roberta-base-japanese",
}
assert set(lang2transformer_name.keys()) == set(ALL_LANGS)

train_lang2dataloader = {}
test_lang2dataloader = {}
val_lang2dataloader = {}
anchors_lang2dataloader = {}

for lang in ALL_LANGS:
    transformer_name = lang2transformer_name[lang]
    print(transformer_name)
    lang_tokenizer = AutoTokenizer.from_pretrained(transformer_name)
    
    aux_train_ds = DictDataset(train_datasets[lang], data_key, target_key)    
    train_lang2dataloader[lang] = DataLoader(aux_train_ds,
                                       num_workers=4,
                                       collate_fn=partial(collate_fn, tokenizer=lang_tokenizer),
                                       pin_memory=True,
                                       #persistent_workers= True,
                                       batch_sampler=ClassAccumulationSampler(aux_train_ds,
                                                       batch_size=16,
                                                       drop_last=True,
                                                       accumulation=num_labels,
                                                       indv=True,
                                                       main_random=True
                                                      )
                                            )
                                  
    
    aux_test_ds = DictDataset(test_datasets[lang], data_key, target_key)
    ds_test_multi = IntraLabelMultiDraw(aux_test_ds, 16)
    test_lang2dataloader[lang] = (DataLoader(aux_test_ds,
                                       num_workers=4,
                                       collate_fn=partial(collate_fn, tokenizer=lang_tokenizer),
                                       batch_size=32,
                                       pin_memory=True
                                       ),
                                  DataLoader(ds_test_multi,
                                       num_workers=4,
                                       collate_fn=partial(multi_draw_collate_fn, tokenizer=lang_tokenizer, cls=False),
                                       batch_size=1,
                                       pin_memory=True
                                       )
                                 )
    
    
    aux_val_ds = DictDataset(val_datasets[lang], data_key, target_key)
    
    
    val_lang2dataloader[lang] = DataLoader(aux_val_ds,
                                       num_workers=4,
                                       collate_fn=partial(collate_fn, tokenizer=lang_tokenizer),
                                       batch_size=32,
                                       pin_memory=True
                                       )
    
    aux_anc_ds = DictDataset(lang2anchors[lang], data_key, target_key)
     
    anchors_lang2dataloader[lang] = DataLoader(aux_anc_ds,
                                       num_workers=4,
                                       pin_memory=False,
                                       collate_fn=partial(collate_fn, tokenizer=lang_tokenizer, cls=False),
                                       batch_size=16,
                                       )
    
    
from pl_modules.pl_topo_roberta import LitTopoRelRoberta

def test_model(model, dataloader, title=""):
    preds = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        batch_idx = 0
        for batch, _ in tqdm(dataloader, position=0, leave=True, desc="Computing"+title):
            batch.to(device)
            batch_latents = model(batch_idx=batch_idx, **batch)["prediction"].argmax(-1)
            preds.append(batch_latents)
            batch_idx = 1

    preds = torch.cat(preds, dim=0).detach().cpu().numpy()
    test_y = np.array(test_datasets["en"][target_key])

    precision, recall, fscore, _ = precision_recall_fscore_support(test_y, preds, average="weighted")
    mae = mean_absolute_error(y_true=test_y, y_pred=preds)
    acc = (preds == test_y).mean()
    return precision, recall, acc, fscore, mae


torch.autograd.set_detect_anomaly(True)

EPOCHS = 40 if fine_grained else 3


def train_network(lang, mode="relative", seed=24, test=False, topo=("pre", "L_2", 7, 0.1, "L_1")):
    
    # Create a PyTorch Lightning trainer with the generation callback
    aux = 'fine_grained' if fine_grained else 'coarse_grained'
    if topo is None:
        aux = f"{aux}/vanilla_linear_{lang}_{mode}_seed{seed}"
    else:
        print(topo[2])

        if  mode == "relative":
            aux = f"{aux}/topo_linear_{lang}_{mode}_{topo[0]}_seed{seed}"
        else:
            aux = f"{aux}/topo_linear_{lang}_{mode}_seed{seed}"
        
    title = CHECKPOINT_PATH /  aux
    trainer = pl.Trainer(default_root_dir=title, 
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         gradient_clip_val=1,
                         accumulate_grad_batches=num_labels+1,
                         max_epochs=EPOCHS, 
                         reload_dataloaders_every_n_epochs=1,
                         callbacks=[ModelCheckpoint(save_weights_only=True,
                                                   mode="max", monitor="val_acc",
                                                   filename="best"
                                                   ),
                                    ModelCheckpoint(save_weights_only=True,
                                                    filename="full"
                                                   ),
                                    LearningRateMonitor(logging_interval='step'),
                                    ]
                         )
    
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
    
    transformer_model = lang2transformer_name[lang]
    
    anchor_loader = None
    if mode == "relative":
        anchor_loader = anchors_lang2dataloader[lang]
    
    train_loader = train_lang2dataloader[lang]
    steps = EPOCHS*len(train_loader)
    power = 10
    freq_anchors = power*(num_labels+1)
    model = LitTopoRelRoberta(
                          num_labels=num_labels,
                          transformer_model=transformer_model,
                          anchor_dataloader=anchor_loader,
                          topo_par=topo,
                          hidden_size=num_anchors,
                          # normalization_mode="batchnorm",
                          normalization_mode=None,
                          output_normalization_mode=None,
                          dropout_prob=0.1,
                          seed=seed,
                          steps=steps,
                          weight_decay=0.01, 
                          head_lr=1e-3/(num_labels),
                          encoder_lr=1.75e-4/(num_labels),
                          layer_decay=0.65,
                          scheduler_act=True,
                          freq_anchors=freq_anchors,
                          device=device,
                          fine_tune=False,
                          linear=True
                          )
    
    val_loader = val_lang2dataloader[lang]
    trainer.fit(model, train_lang2dataloader[lang], val_dataloaders=val_loader)
    
    if test:
        
        res = test_model(model.net, test_lang2dataloader[lang][0])
        with open("res.txt", "a") as f:
            f.write(f"-------------{topo}-------------\n")
            f.write(f"precision {res[0]}\n")
            f.write(f"recall {res[1]}\n")
            f.write(f"acc {res[2]}\n")
            f.write(f"fscore {res[3]}\n")
            f.write(f"mae {res[4]}\n")
            
        print("precision", res[0])
        print("recall", res[1])
        print("acc", res[2])
        print("fscore", res[3])
        print("mae", res[4])
        aux = "rel_topo" if topo is not None else "rel_vanilla"
        
        title = f"{lang.upper()} {mode}: VR " + r"$H_0$" + " pers w/ " + r"$L^2$" 
        if topo is not None:
            if topo[1]=="L_inf":
                title += " and " +r"$L^{\infty}$"
                
            if type(topo[2]) is tuple:
                title+= f" ({topo[0]}, "+ r"$\lambda={}$, ".format(topo[3]) + r"$\beta_{pre}$" +r"$={}$, ".format(topo[2][0]) +\
                    r"$\beta_{post}$" +r"$={}$)".format(topo[2][1]) 
            else:
                title+= f" ({topo[0]}, "+ r"$\lambda={}$, ".format(topo[3]) + r"$\beta={}$)".format(topo[2])
                
        if topo is not None:
            path =  f"{'fine_grained' if fine_grained else 'coarse_grained'}/full_topo_dist_linear/{lang}_{mode}"
            if mode == "relative":
                path += f"_{topo[0]}"
            if type(topo[2]) is tuple:
                path += f"_{topo[2][0]}_{topo[2][1]}_seed{seed}.png"
            else:
                path += f"_{topo[2]}_seed{seed}.png"
            
        else:
            path =  f"{'fine_grained' if fine_grained else 'coarse_grained'}/full_topo_dist_linear/{lang}_{mode}_seed{seed}.png"

        topo_model(model.net, device, test_lang2dataloader[lang][1],
                 FIG_PATH / aux / path,
                 title=title,
                 pers=("L_2", "L_inf"), plot_topo=True, relative=mode=="relative");
       
    
    model.to("cpu")
    del model
    
train_network("en", mode="relative", seed=0, test=True, topo=None)
#train_network("en", mode="relative", seed=1, test=True, topo=("both", "L_inf", (3, grad2dif(25)), 0.1, "L_1"))
#train_network("fr", mode="relative", seed=0, test=True, topo=("both", "L_inf", (3, grad2dif(25)), 0.1, "L_1"))
#train_network("fr", mode="relative", seed=1, test=True, topo=("both", "L_inf", (3, grad2dif(25)), 0.1, "L_1"))
