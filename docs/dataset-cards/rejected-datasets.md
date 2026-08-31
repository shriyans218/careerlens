# Dataset Card: Rejected Datasets

| Dataset | File | Verdict | Reason |
|---|---|---|---|
| CPP | `data/cpp.csv` | Rejected | All three models scored at random-chance level (~1%); labels don't correlate with features. |
| CPD | Data_final.csv (not shipped) | Rejected | 105 rows, 104 unique career labels — ~1 sample/class. |
| AICPT | personality.csv (not shipped) | Rejected | 2,527 rows, every row a unique career title — zero repeated classes. |
