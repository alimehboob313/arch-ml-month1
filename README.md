# Email Spam Classification

Machine Learning internship at Arch Technologies, Month 1, Task 1.

A machine learning pipeline that classifies emails as spam or not spam. Email text is cleaned, converted to TF-IDF features, and used to train two models, Multinomial Naive Bayes and Logistic Regression, which are then compared.

## Dataset

- Source: [Spam Email Classification Dataset/purusinghvi/email-spam-classification-dataset) (Kaggle)
- 83,448 emails with two columns: `label` (1 = spam, 0 = not spam) and `text`
- Nearly balanced classes: 52.62% spam and 47.38% not spam
- No missing values. The text was already lowercase and free of punctuation, with every number replaced by an `escapenumber` placeholder

The dataset is not included in this repository. Download it from the link above and place the CSV at `data/REPLACE_WITH_FILE_NAME.csv`.

## Approach

1. **Explore:** checked the shape, missing values, class balance, duplicates, and email length (median about 152 words).
2. **Clean:** removed the 2 duplicated texts (all copies), removed the `escapenumber` placeholder, lowercased, kept letters only, and collapsed extra spaces. Emails left empty after cleaning were dropped, leaving 83,382 emails.
3. **Split:** 80% training (66,705 emails) and 20% test (16,677 emails), stratified so both sets are 52.59% spam, with `random_state=42`.
4. **Extract features:** TF-IDF with English stop words removed, a 50,000-word vocabulary, `min_df=5`, and sublinear term frequency. The vectorizer was fitted on the training set only, to avoid data leakage.
5. **Train:** Multinomial Naive Bayes and Logistic Regression (`max_iter=1000`).
6. **Evaluate:** accuracy, precision, recall, F1-score, and confusion matrices on the unseen test set.

## Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Naive Bayes | 0.9707 | 0.9836 | 0.9602 | 0.9718 |
| Logistic Regression | **0.9868** | 0.9808 | **0.9944** | **0.9875** |

Confusion matrices on the 16,677 test emails:

| | NB: predicted not spam | NB: predicted spam | LR: predicted not spam | LR: predicted spam |
|---|---|---|---|---|
| **Actual not spam** | 7,766 | 140 | 7,735 | 171 |
| **Actual spam** | 349 | 8,422 | 49 | 8,722 |

## Key findings

- Logistic Regression made 220 mistakes against 489 for Naive Bayes, and missed far less spam (49 emails against 349).
- Naive Bayes was slightly more careful with real emails (140 false alarms against 171) and trained faster, though both train in under a second.
- A check on 4 hand-written emails (2 spam, 2 normal) classified all of them correctly.

## Limitations

The words with the strongest "not spam" weights include `enron`, `vince`, `samba`, and `perl`. These come from the sources of the dataset's real emails (a company email collection and technical mailing lists), so the model partly recognises where an email came from, as well as what spam looks like. The scores above describe performance on this dataset, and results on emails from other sources may be lower.

## Project structure

```
arch-ml-month1/
├── task1_spam.ipynb    # full workflow: exploration, cleaning, features, models, evaluation
├── requirements.txt    # library versions
├── .gitignore
└── README.md
```

## How to run

```bash
python -m venv venv
source venv/Scripts/activate    # Git Bash on Windows
pip install -r requirements.txt
```

Download the dataset into a `data/` folder as described above, then open `task1_spam.ipynb` in VS Code and run the cells in order.

## Tools

Python 3.14, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter, VS Code.

## Author

Ali ([@alimehboob313](https://github.com/alimehboob313))