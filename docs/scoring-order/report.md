# Does the order an LLM fills a scoring matrix in change its scores?

**Short answer, from one controlled experiment:** for the two Claude models tested, scoring row by row (one option on every criterion, then the next option) or column by column (every option on one criterion, then the next criterion) made no measurable difference to the scores, to halo or to the ranking. What did change things was scoring each cell in its own isolated call: the error against the designed scores roughly doubled, scores bunched toward the middle, and one model picked a different winner. For the two small local models, every traversal was noisy, and halo appeared in all of them.

So the practical default is: **score the options together, in one context, in whichever order is convenient; do not score cells in isolation.** If you must isolate calls, give each call the other options as a frame of reference.

This answers the question in thorwhalen/essays#28 for one task shape. The limits are below, and they matter.

## The question

When an LLM fills in a criteria × alternatives matrix, it can go row-wise (all criteria for one alternative), column-wise (all alternatives on one criterion), produce the whole table in one pass, or score each cell in an isolated call. In human raters the choice matters: rating one dimension at a time across all ratees is a standard way to reduce halo, the tendency to let an overall impression colour every dimension [4], [5]. Does that transfer to LLM raters?

## What the literature says (a short scan, not a review)

- **LLM judges show halo and order effects, but nobody has compared traversals.** A pre-registered audit of LLM essay scoring measured mean residual halo of r = .26 across judges, close to high-halo human raters, with wide variation between judges [1]. Rubric-based judging shows position bias that depends on the order of the rubric's options [2], and a prior score shown to the judge anchors its next one [3]. None of these vary row-wise against column-wise traversal of a matrix.
- **In humans the effect is real but modest, and its measurement is contested.** One study found that the rating format had little effect on overall ratings but a large effect on halo, which was lowest when raters went dimension by dimension; accuracy gains were small and inconsistent [4]. Format effects depend on the job, and no format wins everywhere [5]. Halo indices themselves confound true correlation between dimensions with rater error, which led some authors to call for a moratorium on using them as outcomes [6], [7].

That gap is why an experiment was worth running. It is also why this one uses a matrix whose true correlations are known by construction (below): it sidesteps the true-versus-illusory halo problem of [6].

## The experiment

**Task.** Eight candidate use cases for a fictional graph-analytics startup, five criteria (market pain, technical fit, reusability, defensibility, time to signal), each scored 1 to 5. Each description carries one piece of evidence per criterion, written to a designed level. The designed matrix has a mean inter-criterion correlation of −0.18, so a rater with no halo should not see criteria rise and fall together, and the designed totals are close (12 to 18), so rankings can flip. The designed winner is D (18), then F (17). Code and data: [`experiment/`](experiment/).

**Traversals.**

- *row*: one generation, scores written one line each, option by option;
- *column*: one generation, criterion by criterion;
- *table*: one generation, the whole matrix as a markdown table;
- *cell*: 40 isolated calls, each seeing one option and one criterion.

In each repetition the options and criteria were shown in a shuffled order, the same for every traversal of that repetition, so presentation order is not confounded with traversal.

**Models.** Claude Haiku and Claude Sonnet through the Claude Code subscription (`claude -p`, default temperature), and two small local models through Ollama: Mistral 7B Instruct and Llama 3.2 3B (temperature 1). Six repetitions per traversal, three for *cell* (552 calls in all).

**Measures.** Mean absolute error against the designed scores; halo as the mean correlation between criteria across options (higher than −0.18 means halo) and as the spread of one option's scores across criteria (lower means halo); rank stability between repetitions (Kendall's tau of totals); the winner. Row against column was tested by permutation over matrices.

## Results

Full tables: [`experiment/results.md`](experiment/results.md).

| model | traversal | error vs design | halo r (design −0.18) | spread (design 1.45) | rank tau between reps | winners |
|---|---|---|---|---|---|---|
| Haiku | row | 0.21 | −0.15 | 1.39 | 0.88 | D×6 |
| Haiku | column | 0.23 | −0.15 | 1.40 | 0.83 | D×6 |
| Haiku | table | 0.20 | −0.16 | 1.38 | 0.83 | D×6 |
| Haiku | cell | **0.61** | −0.11 | **1.07** | 0.87 | D×2, B×1 |
| Sonnet | row | 0.28 | −0.14 | 1.42 | 0.93 | D×6 |
| Sonnet | column | 0.28 | −0.12 | 1.42 | 0.94 | D×6 |
| Sonnet | table | 0.27 | −0.12 | 1.39 | 0.91 | D×6 |
| Sonnet | cell | **0.54** | −0.14 | **1.14** | 0.93 | **B×3** |
| Mistral 7B | row | 1.31 | 0.30 | 0.89 | 0.43 | F×4, D×1 |
| Mistral 7B | column | 1.27 | 0.22 | 0.77 | 0.27 | B×2, F×1 |
| Mistral 7B | table | 1.20 | 0.22 | 0.87 | 0.45 | F×2, D×1, E×1 |
| Mistral 7B | cell | 1.43 | 0.11 | 1.26 | −0.02 | F×2, C×1 |
| Llama 3.2 3B | row | 1.32 | 0.25 | 1.14 | 0.37 | C×2, D×2, E×1 |
| Llama 3.2 3B | column | 1.16 | 0.10 | 1.14 | 0.39 | D×3, H×2, A×1 |
| Llama 3.2 3B | table | 1.25 | 0.17 | 1.08 | 0.41 | C×2, D×2, H×1 |
| Llama 3.2 3B | cell | 1.44 | −0.01 | 0.50 | 0.17 | A×3 |

**Row against column: no difference for either Claude model.** Halo, spread and error are within noise (permutation p from 0.30 to 1.0), the rankings agree (tau 0.91 for Haiku, 0.84 for Sonnet) and the winner is D in all 24 matrices. Neither model showed halo at all: the correlation between criteria stayed at the designed level.

**Isolated cells are the outlier.** For both Claude models, cell-by-cell scoring doubled or tripled the error, compressed the spread of each option's scores toward the middle, and lowered agreement with the design (tau of totals with the design 0.70 and 0.54, against 0.78 to 0.89 for the joint traversals). Sonnet picked B in all three cell-wise runs and D in all eighteen joint ones. The likely reason is the missing frame of reference: the descriptions were written relative to each other, and a single option seen alone gets an absolute judgement with nothing to calibrate it against. Repetitions still agreed with each other, so this is a consistent bias, not noise.

**Small models: halo everywhere, and too noisy to separate traversals.** Both showed halo (inter-criterion correlation 0.10 to 0.30 against a designed −0.18), and their row-wise runs had the highest halo (0.30 and 0.25, against 0.22 and 0.10 column-wise), which is the direction human research predicts, but six repetitions cannot tell that from noise (p 0.62 and 0.33). Their rankings barely agreed between repetitions (tau 0.27 to 0.45) and they followed the format poorly: Llama answered the column-wise instruction column-wise once in six; Mistral answered several line-format requests with a table, and produced incomplete matrices in 5 of 18 joint runs.

**Cost.** Through the subscription, the CLI reported $0.19 for Haiku's 138 calls and $3.48 for Sonnet's (notional; the subscription is not billed per call). Most of it is the isolated cells: each call carries the CLI's own overhead, which a direct API call would not. The local models cost nothing but time.

## What this does and does not show

- **One task, explicit evidence.** Each description states its evidence plainly, sentence by sentence. Halo in humans thrives on ambiguous evidence, where a global impression fills the gaps. The Claude models' zero halo here may not hold when the evidence is vague; that is the obvious next experiment.
- **The designed scores are one author's judgement**, and that author was itself a Claude model. Agreement with the design partly measures agreement with that model family.
- **Small samples.** Six repetitions detect large effects only. A modest row/column effect in the small models would need many more runs.
- **No reasons were asked for.** Raters wrote scores only. Asking for a justification before each score lengthens the generated context between scores, and may strengthen order effects.
- **Two model families of different size, not matched.** The Claude-versus-local contrast mixes family and scale.

## Recommendation

1. Score options together in one context. Row, column or a single table: pick for convenience; with capable models it did not matter here.
2. Do not score cells in isolation without a frame of reference. If calls must be separate (for cost or parallelism), include the other options, or anchor descriptions for each level of the scale.
3. With small models, average several traversals or repetitions and check inter-criterion correlation against what you expect: they showed halo in every traversal.

## Reproduce

`experiment/run.py` runs the experiment (Claude models through `claude -p` on a Claude Code subscription, local models through Ollama via the `aix` package) and appends to `results.jsonl`, so a rerun resumes; `experiment/analyze.py` rewrites `results.md`. Both import `matrix.py`, which holds the task and the designed scores.

## References

1. [LLM Judges as Raters: A Pre-Registered Audit of Severity, Halo, Reliability, and Version Instability in LLM Essay Scoring on Public Corpora](https://arxiv.org/abs/2608.29517). arXiv:2608.29517, 2026.
2. [Am I More Pointwise or Pairwise? Revealing Position Bias in Rubric-Based LLM-as-a-Judge](https://arxiv.org/abs/2602.02219). arXiv:2602.02219, 2026.
3. [Anchoring Bias in LLM-as-a-Judge Systems: Prior Scores Compromise Evaluation Independence](https://arxiv.org/abs/2608.25869). arXiv:2608.25869, 2026.
4. [Michigan State University doctoral dissertation on rating method and halo](https://d.lib.msu.edu/etd/22772). MSU Electronic Theses and Dissertations, record 22772.
5. Borman WC. Format and training effects on rating accuracy and rater errors. *Journal of Applied Psychology*, 1979. [Record](https://digitalcommons.usf.edu/psy_facpub/1136).
6. Murphy KR, Jako RA, Anhalt RL. [Nature and consequences of halo error: a critical analysis](https://pure.ul.ie/en/publications/nature-and-consequences-of-halo-error-a-critical-analysis/). *Journal of Applied Psychology*, 1993.
7. Saal FE, Downey RG, Lahey MA. [Rating the ratings: assessing the psychometric quality of rating data](https://gwern.net/doc/statistics/order/comparison/1980-saal.pdf). *Psychological Bulletin*, 1980.

The literature entries were read at the level of abstracts and summaries; the experiment is the primary contribution.
