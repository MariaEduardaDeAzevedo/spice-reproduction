import json
from aac_metrics import evaluate
import pandas as pd
import os
import re
import torch
from spice import SPICE
from aac_metrics.functional import cider_d
from aac_metrics.utils.tokenization import preprocess_mono_sents, preprocess_mult_sents


def tensor_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: tensor_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_serializable(v) for v in obj]
    elif isinstance(obj, torch.Tensor):
        # Se for tensor com um único elemento, converte para float/int
        if obj.numel() == 1:
            return obj.item()
        else:
            return obj.tolist()
    else:
        return obj


annotations = "res.csv"
directory = "metrics_new_test"

df = pd.read_csv(annotations)
df.replace(r"[\n\r\t]|[^\x00-\x7F]+", " ", regex=True, inplace=True)

metrics_to_use = ["bleu", "meteor", "spice"]

res = {"qwen": [], "llava": []}

spice = SPICE(lang="pt", wn_model="own-pt:1.0.0")

if not os.path.exists(directory):
    os.mkdir(directory)

for i in range(df.shape[0]):

    row = df.iloc[i]

    print(f"Running {i}")
    references = [row[row.index.str.startswith("annotation")].values.tolist()]

    qwen_caption = preprocess_mono_sents([row["qwen_caption"]])
    llava_caption = preprocess_mono_sents([row["llava_caption"]])
    references = preprocess_mult_sents(references)

    cider_sents = [qwen_caption[0], llava_caption[0]]
    _, cider_scores = cider_d(cider_sents, [references[0], references[0]])
    cider_scores = tensor_to_serializable(cider_scores)

    try:
        result_qwen = evaluate(qwen_caption, references, metrics=metrics_to_use)
        result_llava = evaluate(llava_caption, references, metrics=metrics_to_use)
    except Exception as e:
        print(f"Erro em {i}:", e)
        continue

    result_qwen = tensor_to_serializable(result_qwen[0])
    result_llava = tensor_to_serializable(result_llava[0])

    result_qwen["cider-d"] = cider_scores["cider_d"][0]
    result_llava["cider-d"] = cider_scores["cider_d"][1]

    # references = list(map(lambda x: x[0], references))

    result_qwen["spice-pt"] = spice.compute_score(
        str(row["qwen_caption"]), references[0]
    )["f1"]

    result_llava["spice-pt"] = spice.compute_score(
        str(row["llava_caption"]), references[0]
    )["f1"]

    r_qwen = {"resultado": result_qwen, "image_id": row["id"]}

    r_llava = {"resultado": result_llava, "image_id": row["id"]}

    res["llava"].append(r_llava)
    res["qwen"].append(r_qwen)

    with open(f"{directory}/{row['id']}.json", "w") as f:
        r = {"llava": result_llava, "qwen": result_qwen}
        json.dump(r, f, indent=4)

print(res)

with open(f"{directory}/metrics.json", "w") as f:
    json.dump(res, f, indent=4)

registros = []

for modelo, entradas in res.items():
    for entrada in entradas:

        image_id = entrada.get("image_id")

        resultado = entrada.get("resultado", {})

        registro = {
            "modelo": modelo,
            "image_id": image_id,
            **resultado,
        }
        registros.append(registro)


df = pd.DataFrame(registros)

df.to_csv(f"{directory}/dataset.csv")
