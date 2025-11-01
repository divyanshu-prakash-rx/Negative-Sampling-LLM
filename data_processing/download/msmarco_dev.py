import os
from datasets import load_dataset

os.makedirs("data/msmarco", exist_ok=True)

# Download training triples
print("=== Downloading MSMARCO ===")
print("Downloading MSMARCO training triples...")
train_dataset = load_dataset("thilina/negative-sampling")["train"]
train_df = train_dataset.to_pandas()

print("Sample train triples:")
print(train_df.head())

# Download qrels from BeIR
print("Loading dev qrels from BeIR...")
qrels = load_dataset("BeIR/msmarco-qrels")["validation"]
qrels_df = qrels.to_pandas()

# Load msmarco corpus - specify the split (usually 'corpus')
print("Loading MSMARCO corpus dataset...")
corpus_dataset = load_dataset("BeIR/msmarco", "corpus")["corpus"]  # Select 'corpus' split
corpus_df = corpus_dataset.to_pandas().rename(columns={"_id": "corpus_id", "text": "passage"})

# Load msmarco queries - specify the split (usually 'queries')
print("Loading MSMARCO queries dataset...")
queries_dataset = load_dataset("BeIR/msmarco", "queries")["queries"]  # Select 'queries' split
queries_df = queries_dataset.to_pandas().rename(columns={"_id": "query_id", "text": "query_text"})

print("Saving dataset to disk...")
train_df.to_csv("data/msmarco/msmarco-train.tsv", sep="\t", index=False)
qrels_df.to_csv("data/msmarco/devs.tsv", sep="\t", index=False)
corpus_df.to_csv("data/msmarco/corpus.tsv", sep="\t", index=False)
queries_df.to_csv("data/msmarco/queries.tsv", sep="\t", index=False)

print("Done.")
print("=== MSMARCO download complete ===")
