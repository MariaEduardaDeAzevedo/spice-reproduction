from transformers import M2M100Config, M2M100ForConditionalGeneration, M2M100Tokenizer
import pandas as pd

df = pd.read_csv("res.csv")

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained(
    "facebook/m2m100_418M", src_lang="pt", tgt_lang="en"
)

d = {}

for i in range(df.shape[0]):
    print(f"Running {i}")
    row = df.iloc[i]

    for c in list(row.index):
        if c not in d:
            d[c] = []

        if c.startswith("annotation") or c.endswith("caption"):
            text = row[c]
            encoded_text = tokenizer(text, return_tensors="pt")
            generated_tokens = model.generate(
                **encoded_text, forced_bos_token_id=tokenizer.get_lang_id("en")
            )
            translated_text = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )

            d[c].append(translated_text[0])

        else:
            d[c].append(row[c])


df = pd.DataFrame(d)
df.to_csv("res-eng.csv")
