# Results: does traversal order change LLM scores?

Designed matrix: mean inter-criterion correlation -0.18, spread 1.45, totals A 15, B 15, C 15, D 18, E 15, F 17, G 12, H 13.

| model | traversal | complete | as instructed | mean | MAE vs design | halo r | spread | rank tau within | winner |
|---|---|---|---|---|---|---|---|---|---|
| haiku | row | 6/6 | 6/6 | 2.96 | 0.21 | -0.15 | 1.39 | 0.88 | D×6 |
| haiku | column | 6/6 | 6/6 | 2.96 | 0.23 | -0.15 | 1.40 | 0.83 | D×6 |
| haiku | table | 6/6 | 6/6 | 2.98 | 0.20 | -0.16 | 1.38 | 0.83 | D×6 |
| haiku | cell | 3/3 | 3/3 | 2.98 | 0.61 | -0.11 | 1.07 | 0.87 | D×2, B×1 |
| sonnet | row | 6/6 | 6/6 | 3.08 | 0.28 | -0.14 | 1.42 | 0.93 | D×6 |
| sonnet | column | 6/6 | 6/6 | 3.12 | 0.28 | -0.12 | 1.42 | 0.94 | D×6 |
| sonnet | table | 6/6 | 6/6 | 3.12 | 0.27 | -0.12 | 1.39 | 0.91 | D×6 |
| sonnet | cell | 3/3 | 3/3 | 3.04 | 0.54 | -0.14 | 1.14 | 0.93 | B×3 |
| mistral | row | 5/6 | 4/5 | 3.08 | 1.31 | 0.30 | 0.89 | 0.43 | F×4, D×1 |
| mistral | column | 3/6 | 2/3 | 2.93 | 1.27 | 0.22 | 0.77 | 0.27 | B×2, F×1 |
| mistral | table | 5/6 | 5/5 | 2.90 | 1.20 | 0.22 | 0.87 | 0.45 | F×2, D×1, E×1 |
| mistral | cell | 3/3 | 3/3 | 2.90 | 1.43 | 0.11 | 1.26 | -0.02 | F×2, C×1 |
| llama3.2 | row | 6/6 | 6/6 | 3.07 | 1.32 | 0.25 | 1.14 | 0.37 | C×2, D×2, E×1 |
| llama3.2 | column | 6/6 | 1/6 | 2.93 | 1.16 | 0.10 | 1.14 | 0.39 | D×3, H×2, A×1 |
| llama3.2 | table | 6/6 | 6/6 | 3.15 | 1.25 | 0.17 | 1.08 | 0.41 | C×2, D×2, H×1 |
| llama3.2 | cell | 3/3 | 3/3 | 3.66 | 1.44 | -0.01 | 0.50 | 0.17 | A×3 |

## Row against column (permutation test over matrices, two-sided)

| model | halo r row | halo r column | p | spread row | spread column | p | MAE row | MAE column | p | tau(row mean, column mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| haiku | -0.15 | -0.15 | 0.819 | 1.39 | 1.40 | 0.630 | 0.21 | 0.23 | 0.583 | 0.91 |
| sonnet | -0.14 | -0.12 | 0.300 | 1.42 | 1.42 | 0.832 | 0.28 | 0.28 | 1.000 | 0.84 |
| mistral | 0.30 | 0.22 | 0.622 | 0.89 | 0.77 | 0.329 | 1.31 | 1.27 | 0.748 | 0.33 |
| llama3.2 | 0.25 | 0.10 | 0.329 | 1.14 | 1.14 | 0.952 | 1.32 | 1.16 | 0.247 | 0.76 |

## Agreement between traversals (Kendall tau of mean totals) and with the design

- **haiku**: row-column 0.91; row-table 0.93; row-cell 0.55; column-table 0.98; column-cell 0.67; table-cell 0.62; row-design 0.81; column-design 0.86; table-design 0.81; cell-design 0.70
- **sonnet**: row-column 0.84; row-table 0.91; row-cell 0.49; column-table 0.93; column-cell 0.52; table-cell 0.59; row-design 0.78; column-design 0.81; table-design 0.89; cell-design 0.54
- **mistral**: row-column 0.33; row-table 0.62; row-cell 0.40; column-table 0.36; column-cell 0.14; table-cell 0.21; row-design 0.53; column-design 0.32; table-design 0.48; cell-design 0.48
- **llama3.2**: row-column 0.76; row-table 0.71; row-cell 0.11; column-table 0.69; column-cell 0.22; table-cell 0.11; row-design 0.24; column-design 0.12; table-design 0.40; cell-design 0.12

## Calls and cost

| model | calls | cost (USD, as reported) |
|---|---|---|
| haiku | 138 | 0.188 |
| sonnet | 138 | 3.482 |
| mistral | 138 | 0.000 |
| llama3.2 | 138 | 0.000 |
