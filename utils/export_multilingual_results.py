"""
Modified from https://openreview.net/attachment?id=SrC-nwieGJ&name=supplementary_material
"""
import pandas as pd
import numpy as np

COLUMNS_TO_DROP = ["precision", "recall", "CKA_pre", "CKA_post"]
MEAN_STD_FORMAT = r"${:.2f} \pm {:.2f}$"

def read_df(fine_grained, mode, train_perc, best, path):
    aux = f"nlp_multilingual-stitching-amazon-{'fine_grained' if fine_grained else 'coarse_grained'}-{mode}-{train_perc}"
    
    if best:
        aux+="-best"
    aux += ".tsv"
    
    full_df = pd.read_csv(
        path / aux,
        sep="\t"
    )
    return full_df


def rearrange_embedtype_as_column(mydf):
    relative_out = mydf[mydf[("embed_type")] == "relative"]
    relative_out.columns = pd.MultiIndex.from_tuples(
        [
            ("seed",  ""),
            ("embed_type",  ""),
            ("enc_lang", ""),
            ("dec_lang", ""),
            ("Relative", "acc"),
            ("Relative", "fscore"),
            ("Relative", "mae"),
            ("stitched", ""),
        ],
    )
    absolute_out = mydf[mydf[("embed_type")] == "absolute"]
    absolute_out.columns = pd.MultiIndex.from_tuples(
        [
            ("seed", ""),
            ("embed_type", ""),
            ("enc_lang", ""),
            ("dec_lang", ""),
            ("Absolute", "acc"),
            ("Absolute", "fscore"),
            ("Absolute", "mae"),
            ("stitched", ""),
        ],
    )
    return pd.merge(
        relative_out.drop(columns=["embed_type"]),
        absolute_out.drop(columns=["embed_type"]),
        on=[
            ("enc_lang",""),
            ("dec_lang", ""),
            ("seed", ""),
            ("stitched", ""),
        ],
    )

def to_latex(df):
    return df.to_latex(
        escape=False,
        multirow=True,
        sparsify=True,
        multicolumn_format="c",
    )


def process_df(fine_grained, mode, path, train_perc=0.25, best=False):

    full_in_domain = read_df(fine_grained=fine_grained, mode=mode, train_perc=train_perc, best=best, path=path)
    full_in_domain = full_in_domain.drop(columns=COLUMNS_TO_DROP)
    full_in_domain["fscore"] = full_in_domain["fscore"] * 100
    full_in_domain["acc"] = full_in_domain["acc"] * 100
    full_in_domain["mae"] = full_in_domain["mae"] * 100


    df = rearrange_embedtype_as_column(full_in_domain)

    full_df = df.drop(
        columns=[
            ( "seed","" ),
            ("stitched", ""),
        ]
    )


    enc_lang = "Encoder"
    dec_lang = "Decoder"
    full_df = full_df.rename(columns={"enc_lang": enc_lang, "dec_lang": dec_lang})
    full_df = full_df[
        [
            (dec_lang, ""),
            (enc_lang,  ""),
            ("Absolute",  "acc"),
            ("Absolute",  "fscore"),
            ("Absolute",  "mae"),
            ("Relative",  "acc"),
            ("Relative",  "fscore"),
            ("Relative",  "mae"),
        ]
    ]
    
    df = (
        full_df.groupby(
            [dec_lang,enc_lang],
        )
        .agg([np.mean, np.std])
        .round(2)
    )
    
    df_res = df.copy()
    for embed in ("Absolute", "Relative"):
        for metric, new_name in (("acc", "Acc"), ("fscore", "FScore"), ("mae", "MAE")):
            df[(embed, new_name, "")] = df.apply(
                lambda row: MEAN_STD_FORMAT.format(
                    row[(embed, metric, "mean")], row[(embed, metric, "std")]
                ),
                axis=1,
            )
            for agg in ("mean", "std"):
                df = df.drop(columns=[(embed, metric, agg)])

    tex_table = to_latex(df)
        
    header = '\\begin{table}[ht]\n\\centering\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{clcccccc}\n\\toprule\n   &    & \\multicolumn{3}{c}{Absolute} & \\multicolumn{3}{c}{Relative} \\\\\n \\cmidrule(lr){3-5} \n \\cmidrule(lr){6-8} \n Decoder & Encoder & Acc & FScore & MAE & Acc & FScore & MAE \\\\\n'
    caption = ('Fine grained' if fine_grained else 'Coarse grained') + f": {mode}"
    bottom = "}\n\\caption{" +\
              caption +\
              "}\n\\label{" +\
              f'tab:multilingual-{mode}-{"fine" if fine_grained else "coarse"}-grained' +\
              "}\n\\end{table}"

    tex_res = "".join([header, tex_table[tex_table.find('\\midrule'):], bottom]).replace("\\cline{1-8}", "&\\\\")


    return df_res, tex_res
