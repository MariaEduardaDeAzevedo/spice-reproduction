import pandas as pd

df = pd.read_csv("respostas_form.csv")
df_opt = df.loc[:, df.columns.str.startswith("escolha")]

df_opt.columns = [col.split(".")[-1] for col in df_opt.columns]

opt = {"[OPÇÃO A]": "qwen", "[OPÇÃO B]": "llava"}

df_opt = df_opt.map(lambda x: opt[x.split("]")[0] + "]"])

scores = {"id": [], "modelo": [], "score": []}

for c in df_opt.columns:
    col = df_opt[c]
    l = len(col)
    vc = col.value_counts()

    scores["id"].append(c)
    scores["modelo"].append("qwen")
    try:
        scores["score"].append(vc["qwen"] / l)
    except:
        scores["score"].append(0)

    scores["id"].append(c)
    scores["modelo"].append("llava")
    try:
        scores["score"].append(vc["llava"] / l)
    except:
        scores["score"].append(0)

df_result = pd.DataFrame(scores)

df_result.to_csv("human_scores.csv")
