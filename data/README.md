# Training datasets

Put your downloaded datasets here, then pass the path to the CLI or the
Training Studio dashboard (paths are relative to the **project root**).

Provide **one** dataset source per training run.

## 1. Text dataset (CSV)

File: `data/text.csv` — columns `text`, `label`:

```csv
text,label
I feel hopeless and empty,depressed
what a great day,non_depressed
everything is dark,1
meh,0
```

Train with: `python train.py --text-csv data/text.csv`

Labels may be `0/1`, `depressed/non_depressed`, `positive/negative`,
`yes/no`, ... (case-insensitive).

## 2. Image dataset (folder)

Folder `data/images/` with **exactly two** class subfolders
(any two of these names work, or use `class_0/class_1`):

```
data/images/
├── depressed/       # class label = 1
│   ├── 001.jpg
│   └── 002.jpg
└── non_depressed/   # class label = 0
    ├── 003.jpg
    └── 004.jpg
```

Train with: `python train.py --image-dir data/images`

## 3. Image dataset (CSV)

File: `data/images.csv` — columns `image`, `label`. Image paths are
resolved relative to the CSV file:

```csv
image,label
images/depressed/001.jpg,depressed
images/non_depressed/003.jpg,non_depressed
```

Train with: `python train.py --image-csv data/images.csv`

## 4. Multimodal dataset (CSV)

File: `data/pairs.csv` — columns `text`, `image`, `label`. Trains the
cross-attention fusion on both modalities together:

```csv
text,image,label
feeling hopeless,images/depressed/001.jpg,depressed
great day,images/non_depressed/003.jpg,non_depressed
```

Train with: `python train.py --multimodal-csv data/pairs.csv`

## Tips

- Use **absolute paths** (e.g. `C:/Users/you/Downloads/dataset.csv`) if
  your data lives outside the project — this also works in the dashboard.
- After training, weights are saved to `checkpoints/latest/` and are
  automatically used for inference by the app.
- Fine-tuning keeps the DistilBERT/ResNet encoders frozen by default
  (fast, works on small datasets). Add `--unfreeze` to fine-tune them too.
