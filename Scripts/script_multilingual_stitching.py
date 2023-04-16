# -*- coding: utf-8 -*-

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import argparse

# Commented out IPython magic to ensure Python compatibility.
import random
from pathlib import Path
import numpy as np
import pytorch_lightning as pl

import pandas as pd

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error

from functools import partial

from aux_relative_text.multilingual_amazon_anchors import MultilingualAmazonAnchors
from typing import *

from modules.stitching_module import StitchingModule

from pl_modules.pl_roberta import LitRelRoberta

from datasets import load_dataset, ClassLabel

# Tensorboard extension (for visualization purposes later)
# %load_ext tensorboard

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = Path("./data")
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = Path("./saved_models/rel_multi_vanilla")
RESULT_PATH = Path("./results/rel_multi_vanilla")

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


target_key: str = "class"
data_key: str = "content"
anchor_dataset_name: str = "amazon_translated"  
ALL_LANGS = ("en", "es", "fr")
num_anchors: int = 768
train_perc: float = 0.25


"""# Data"""

def get_dataset(lang: str, split: str, perc: float, fine_grained: bool):
    pl.seed_everything(42)
    assert 0 < perc <= 1
    dataset = load_dataset("amazon_reviews_multi", lang)[split]

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
        ClassLabel(num_classes=5 if fine_grained else 2, names=list(map(str, range(1, 6) if fine_grained else (0, 1)))),
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


def collate_fn(batch, tokenizer, cls=True):
    data = []
    labels = []
    for sample in batch:
        data.append(sample[data_key])
        if cls:
            labels.append(sample[target_key])

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

"""# Train"""


def train_network(lang, epochs, mode="relative", seed=24, fine_tune=False):
    
    # Create a PyTorch Lightning trainer with the generation callback
    
    if fine_grained:
        title = CHECKPOINT_PATH / 'fine_grained' 
    else:
        title = CHECKPOINT_PATH / 'coarse_grained' 
    
    if fine_tune:
        title = title / f"finetune_{lang}_{mode}_seed{seed}"
    else:
        title = title / f"full_{lang}_{mode}_seed{seed}"
    
    trainer = pl.Trainer(default_root_dir=title, 
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         accumulate_grad_batches=num_labels,
                         max_epochs=epochs, 
                         callbacks=[ModelCheckpoint(save_weights_only=True),
                                    LearningRateMonitor(logging_interval='step')
                                    ])
    
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
    
    transformer_model = lang2transformer_name[lang]
    
    anchor_loader = None
    if mode == "relative":
        anchor_loader = anchors_lang2dataloader[lang]
    
    
    train_loader = train_lang2dataloader[lang]
    
    if fine_tune:
        freq_anchors = len(train_loader)
    else:
        freq_anchors = 100*num_labels
    
    model = LitRelRoberta(num_labels=num_labels,
                          transformer_model=transformer_model,
                          anchor_dataloader=anchor_loader,
                          hidden_size=num_anchors,
                          normalization_mode="batchnorm",
                          output_normalization_mode=None,
                          dropout_prob=0.1,
                          seed=seed,
                          steps=epochs*len(train_loader),
                          weight_decay=0.01, 
                          head_lr=1e-3/num_labels,
                          encoder_lr=1.75e-4/num_labels,
                          layer_decay=0.65,
                          scheduler_act=True,
                          freq_anchors=freq_anchors,
                          device=device,
                          fine_tune=fine_tune
                          )
    
    val_loader = val_lang2dataloader[lang]
   
    trainer.fit(model, train_loader, val_loader)
        
    return model

"""# Results"""

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



if __name__ == '__main__':
    
    # options for training
    parser = argparse.ArgumentParser()

    # default parameter setting for Vit-B
    parser.add_argument('--finegrain', action='store_true')
    parser.add_argument('--coarse', dest='finegrain', action='store_false')
    parser.set_defaults(finegrain=True)
    opt = parser.parse_args()
    
    fine_grained: bool = opt.finegrain
    
        
    train_datasets = {
        lang: get_dataset(lang=lang, split="train", perc=train_perc, fine_grained=fine_grained) for lang in ALL_LANGS
        }

    test_datasets = {
        lang: get_dataset(lang=lang, split="test", perc=1, fine_grained=fine_grained) for lang in ALL_LANGS
        }

    val_datasets = {
        lang: get_dataset(lang=lang, split="validation", perc=1, fine_grained=fine_grained) for lang in ALL_LANGS
        }

    num_labels = list(train_datasets.values())[0].features[target_key].num_classes

    train_datasets["es"][5]

    assert len(set(frozenset(train_dataset.features.keys()) for train_dataset in train_datasets.values())) == 1
    class2idx = train_datasets["en"].features[target_key].str2int

    train_datasets["en"].features

    """Get pararel anchors"""

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
    

    lang2transformer_name = {
        "en": "roberta-base",
        "es": "PlanTL-GOB-ES/roberta-base-bne",
        "fr": "ClassCat/roberta-base-french",
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
        train_lang2dataloader[lang] = DataLoader(train_datasets[lang],
                                        num_workers=4,
                                        collate_fn=partial(collate_fn, tokenizer=lang_tokenizer),
                                        shuffle=True,
                                        pin_memory=True,
                                        drop_last=True,
                                        batch_size=16,
                                        )
        
        test_lang2dataloader[lang] = DataLoader(test_datasets[lang],
                                        num_workers=4,
                                        collate_fn=partial(collate_fn, tokenizer=lang_tokenizer),
                                        batch_size=32,
                                        )
        
        val_lang2dataloader[lang] = DataLoader(val_datasets[lang],
                                        num_workers=4,
                                        collate_fn=partial(collate_fn, tokenizer=lang_tokenizer),
                                        batch_size=32,
                                        )
        
        anchors_lang2dataloader[lang] = DataLoader(lang2anchors[lang],
                                        num_workers=4,
                                        pin_memory=True,
                                        collate_fn=partial(collate_fn, tokenizer=lang_tokenizer, cls=False),
                                        batch_size=48,
                                        )
    
    SEEDS = [0, 2, 3, 4]
    EPOCHS = 5 if fine_grained else 3

    for seed in tqdm(SEEDS, leave=False, desc="seed"):
        for fine_tune in tqdm([True, False], leave=False, desc="fine_tune"):
            for embedding_type in tqdm(["absolute", "relative"], leave=False, desc="embedding_type"):
                for train_lang in tqdm(ALL_LANGS, leave=False, desc="lang"):
                    train_network(train_lang, mode=embedding_type, seed=seed, fine_tune=fine_tune, epochs=EPOCHS)
    
        
    models = {
        train_mode: {
            seed: {
                embedding_type: {
                    train_lang: LitRelRoberta.load_from_checkpoint(
                                CHECKPOINT_PATH / 
                                f"{'fine_grained' if fine_grained else 'coarse_grained'}/{train_mode}_{train_lang}_{embedding_type}_seed{seed}" /
                                f"lightning_logs/version_0/checkpoints/{'epoch=4-step=3125.ckpt' if fine_grained else 'epoch=2-step=3750.ckpt'}" )

                    for train_lang in ALL_LANGS
                }
                for embedding_type in ["absolute", "relative"]
            }
            for seed in [1]
        }
        for train_mode in tqdm(["finetune", "full"], leave=True, desc="mode")
    }

    numeric_results = {
        "finetune": {
            "seed": [],
            "embed_type": [],
            "enc_lang": [],
            "dec_lang": [],
            "precision": [],
            "recall": [],
            "acc": [],
            "fscore": [],
            "mae": [],
            "stitched": []
        },
        "full": {
            "seed": [],
            "embed_type": [],
            "enc_lang": [],
            "dec_lang": [],
            "precision": [],
            "recall": [],
            "acc": [],
            "fscore": [],
            "mae": [],
            "stitched": []
        },
    }

    for mode in ["finetune", "full"]:
        for seed in [1]:
            for embed_type in ["absolute", "relative"]:
                for enc_lang  in ALL_LANGS:
                    for dec_lang  in ALL_LANGS:
                        
                        model = models[mode][seed][embed_type][enc_lang].net
                        if embed_type == "relative":
                            model.anchor_dataloader = anchors_lang2dataloader[enc_lang]
                            
                        if enc_lang != dec_lang:
                            model_dec = models[mode][seed][embed_type][dec_lang].net
                            model = StitchingModule(model, model_dec)
                        
                            
                        # The data is paired with its encoder
                        test_loader = test_lang2dataloader[enc_lang]
                        title = f" {mode}_seed{seed}_{embed_type}_{enc_lang}_{dec_lang}"

                        precision, recall, acc, fscore, mae = test_model(model, test_loader, title)
                        numeric_results[mode]["embed_type"].append(embed_type)
                        numeric_results[mode]["enc_lang"].append(enc_lang)
                        numeric_results[mode]["dec_lang"].append(dec_lang)
                        numeric_results[mode]["precision"].append(precision)
                        numeric_results[mode]["recall"].append(recall)
                        numeric_results[mode]["acc"].append(acc)
                        numeric_results[mode]["fscore"].append(fscore)
                        numeric_results[mode]["stitched"].append(enc_lang != dec_lang)
                        numeric_results[mode]["mae"].append(mae)
                        numeric_results[mode]["seed"].append(seed)

    for mode in ["finetune", "full"]:
        df = pd.DataFrame(numeric_results[mode])
        df.to_csv(
            RESULT_PATH / f"nlp_multilingual-stitching-amazon-{'fine_grained' if fine_grained else 'coarse_grained'}-{mode}-{train_perc}.tsv",
            sep="\t",
            index=False
        )