import pandas as pd

df_annotations = pd.read_csv("annotations_new.csv")
df_references = pd.read_csv("respostas_form.csv")

df_references_annotations = df_references.loc[
    :, df_references.columns.str.startswith("proposta")
]

df_references_annotations.columns = [
    col.split(".")[-1] for col in df_references_annotations.columns
]

df_annotations.rename(columns={"annotation": "annotation_0"}, inplace=True)

columns = ["image_id", "qwen_caption", "llava_caption", "annotation_0"]
df_annotations = df_annotations[columns]

df_references_annotations = df_references_annotations.T
df_annotations = df_annotations.set_index("image_id")

comb = df_references_annotations.join(df_annotations, how="outer")
comb.columns = [f"annotation_{c + 1}" if type(c) is int else c for c in comb.columns]

comb = comb[sorted(comb.columns)]
comb.rename_axis("id", inplace=True)
comb.to_csv("res.csv")
