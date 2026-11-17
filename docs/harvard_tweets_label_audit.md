# Harvard English Depression Tweets — label mapping audit

Dataset: `Depression dataset for English tweets classified binary`
DOI: [10.7910/DVN/ISZCSA](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ISZCSA)
Licence: CC0 1.0 · Version audited: 1.1 · Audit date: 2026-11-14
Local file: `data/raw/text/harvard_tweets/BinaryLabeled_Depression_English_99590.xlsx`
(4,678,835 B, SHA-256 `e566fa96…f9ba`, MD5 `c925e2fb…` identical to the Dataverse record)

## 1. Question this audit answers

Does the released file, or the depositor's own documentation, state **which
numeric code means "depressed"** — `0` or `1`?

**Answer: no. The mapping is unresolved and must not be assumed.**

## 2. What the file actually contains — verified

Read from the workbook itself (`scripts/datasets/validate_harvard_tweets.py`):

| Item | Value |
|---|---|
| Sheets | 1 (`Sheet1`) |
| Columns | `text`, `label`, *(unnamed side column)*, `column_4` |
| Records | 99,590 |
| Rows with empty text | 0 |
| Exact duplicate texts | 0 |
| Duplicate texts after whitespace/case normalisation | 264 |
| Near-duplicate texts with **conflicting** labels | 0 |
| Distinct label values | exactly `0` and `1` |
| Rows with code `0` | **58,405** |
| Rows with code `1` | **41,185** |
| User / tweet / session identifier columns | **none** |

The two extra columns are not data: they hold an unlabelled legend

```
C3 Class          D3 Binary classification data balance
C4 Depressed      D4 58405
C5 Non-depressed  D5 41185
C6 (blank)        D6 99590
```

## 3. What the depositor states — verified

The Dataverse record description (retrieved from the Dataverse API, version 1.1)
reads, in full:

> The dataset contains 99,590 english tweets extracted from twitter as a
> depression dataset. The tweets classified either depressed or non-depressed.
> Depressed 58,405 tweets and Non_depressed 41,185 tweets

It states **class counts only**. No field in the record — title, description,
subject, keyword, file description — maps a numeric code to a class. The single
file ships with an empty description field.

So the only factual constraints are:

- Depressed = 58,405 rows, Non-depressed = 41,185 rows (**[verified]**).
- Exactly one label column with values `0`/`1` and those two counts.

## 4. The ambiguity, stated exactly

Two readings fit the evidence, and they disagree:

**Reading A — `0` = depressed (positional).** The legend lists `Depressed` above
`Non-depressed`, and the counts line up with the codes in that same order:
58,405 rows carry `0`, 41,185 carry `1`. If the depositor sorted the legend the
way the encoder assigned codes, `0` is depressed.

**Reading B — `1` = depressed (conventional).** A `0/1` binary is conventionally
`1` = positive class, and the dataverse description names the positive class
first. Under the usual sklearn/PyTorch convention the positive class is `1`.

**Reading C — `1` = depressed (lexical).** Rate of rows containing a
depression-related term (depress, anxiety, suicid, hopeless, worthless, lonely,
sad, cry, kill, therapy, stress, pain, empty, hate):

| Code | Rows | Rows containing a depression term | Rate |
|---|---|---|---|
| `0` | 58,405 | 10,781 | **18.5 %** |
| `1` | 41,185 | 27,185 | **66.0 %** |

A corpus built by keyword collection would show exactly this pattern: the
"depressed" class is where the collection terms cluster. This is the strongest
single signal, and it points at `1` = depressed — the opposite of Reading A.

**Reading C is evidence, not proof.** It is a bag-of-words count over a
10-token, function-word-stripped text (see §6), it can be confounded by negative
phrases ("i am not depressed"), and it is exactly the kind of lexical from-text
inference that a label audit must not use to *decide* the mapping. It is
recorded because it contradicts the positional reading, which is why the mapping
cannot be settled locally.

## 5. Decision

Per the phase-4 rules, the original codes are **preserved unchanged** and the
semantics are marked **unresolved**. No relabelling, no inversion, no
`depressed = (label == 0)` shortcut anywhere in the codebase.

Consequences, enforced from this audit onward:

- **Blocked:** supervised depression classification on this dataset; any
  cross-dataset comparison of this dataset's scores with EATD/AVDD-TCMI.
- **Allowed:** unsupervised/structural work (tokenisation, length statistics,
  vocabulary analysis), and reporting the dataset as `label ∈ {0,1}` with
  counts, provided the mapping is stated as unresolved in the same table.
- If a binary metric is ever reported, the *assumed* mapping must be printed
  next to it, and the complementary-mapping run should be reported alongside it
  so a reader can see the label is doing the work, not the model.

The machine-readable form lives in
`data/reports/harvard_tweets_validation.json` → `label_code_mapping`
(`ambiguous: true`, `declared_alignment.counts_agree: false`).

## 6. Secondary finding: the text is not raw tweet text

Verified from the file, independent of the label question:

- mean 10.09 tokens per row; 54 % of rows contain **no** common stopword;
- apostrophes replaced by the literal token `xexx` in 21,996 rows (22.1 %);
- no URLs, no `@mentions`, no emoji, all lower-case;
- word order is **not** destroyed (only 6.3 % of rows are alphabetically sorted,
  i.e. chance level) — an earlier internal note claiming the tokens were sorted
  was wrong and has been corrected.

Any model trained on this data learns from a short, pre-processed keyword
sequence, not from natural tweets. That is a separate limitation from the label
ambiguity, and it further limits what results on this dataset can be claimed to
show about "depression detection from social media".

## 7. Question to send the depositor

> Subject: Binary label coding in "Depression dataset for English tweets
> classified binary" (doi:10.7910/DVN/ISZCSA)
>
> Dear authors,
>
> Thank you for releasing this dataset under CC0. I am using it as an auxiliary
> text corpus in an academic study and want to describe it accurately.
>
> The file `BinaryLabeled_Depression_English_99590.xlsx` contains a `label`
> column with values `0` and `1` (58,405 and 41,185 rows), and the workbook holds
> a legend "Depressed 58,405 / Non-depressed 41,185". The dataset description
> states the same two counts but does not say which numeric code corresponds to
> which class.
>
> Could you confirm: **does `label = 0` mean "depressed" or "non-depressed"?**
> A one-line reply is enough, and I will cite it in the dataset description.
>
> Thank you,
> *(project owner)*

## 8. What would resolve this

Any one of the following is sufficient:

1. A reply from the depositor stating the code → class mapping.
2. A released script or notebook from the authors showing `label == 1` being
   used as the positive class.
3. A second released file or split with per-class row indices.

Until one of these exists, the status is **unresolved** and the dataset stays
out of supervised experiments.
