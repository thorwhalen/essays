# The documentation tax: drift, reading habits, and the long search for documents that stay true

Research notes for the essay draft in [thorwhalen/essays#31](https://github.com/thorwhalen/essays/issues/31). Compiled 2026-10-08. Every claim carries a numbered source; "[unverified]" marks things I could not check against the original, "[estimate]" marks my own arithmetic or inference.

## Summary for the blog draft

1. Drift is real but the draft overstates it. In the best-known study, 68% of 45 engineers agreed (44% somewhat, 24% strongly) that "documentation is always outdated relative to the current state of a software system", yet 81% (53% + 28%) agreed documentation "can be useful, even though it might not always be the most up-to-date" [1]. So "worse than useless" needs softening to "stale, still often useful, and trusted less the closer it sits to the code".
2. What stays true is a pattern, not luck: short, local, cheap-to-edit, or executable things (inline comments, test cases, bug-tracker notes) stayed current in the 2003 study; big, structured, separate specifications did not [1]. Every durable remedy since (doctests, executable specifications, fitness functions, ArchUnit) works by making a machine check the document against the code [20][21][22][23].
3. "Nobody read them" is supported by one small observation (docs were 12 of 357 logged events, 3%, at one telecom company) and contradicted by self-report and by agile teams themselves, who said they had too little documentation [1][29]. Better phrasing: developers read short, task-relevant, trusted docs and ask colleagues for the rest [1][27].
4. Comprehension is the dominant cost: about 58% of working time in a 78-developer field study [3], about 70% of IDE time in an earlier IDE-only study [4]. But the "developers code 52 minutes a day" claim is vendor telemetry from a self-selected group [36]; Microsoft's survey of 5,971 developers gives 84 minutes a day of "coding (reading or writing code and tests)" in a 9-hour day, and 2% (9 minutes) on documentation [5].
5. The agile authors did not say "no documentation". They said: working software over comprehensive documentation, "while there is value in the items on the right"; code as the primary documentation; document stable things, only when needed, "just barely good enough"; prefer executable specifications [13][14][15][16].
6. Model-driven engineering and UML round-tripping failed mostly on synchronisation and organisational fit, not on the idea of models: in Petre's 50 interviews, 35 used no UML and 0 used it wholeheartedly; keeping models consistent with code was a stated reason [9]. Notably, the MDE practitioners' main benefit was documenting architecture, not code generation [10].
7. Corrections to the draft's AI claims: (a) "the agent reads every word" is half right: agents follow context-file instructions closely, but a 2026 study found repository context files gave little or no benefit and raised cost by over 20% [38]; another small study found a 28.6% median runtime reduction with an AGENTS.md [39]. The evidence is mixed. (b) Models also have a version of the skimming problem: accuracy drops for information in the middle of long inputs [40]. (c) The "quadratically cheaper" product-of-two-factors claim has no empirical support I could find; treat it as an intuition and label it so. (d) Drift detection with LLMs exists mainly as grey-literature tools, not evaluated studies.
8. A point in the draft's favour: developers themselves want more time on documentation. In a 484-developer Microsoft survey, documentation was among the activities they wished to do more of [34]; and DORA reports documentation quality amplifies the effect of technical practices (correlational survey data) [35].

---

## 1. Documentation goes stale, and how developers use it

### Lethbridge, Singer and Forward (2003), the foundational study

Three studies: interviews at 12 corporate sites and one government site; a telecom study (a Web questionnaire, one engineer followed for 14 weeks, nine engineers shadowed for an hour each, tool-use statistics); and a 50-question survey of 48 respondents (the 2002 Forward and Lethbridge survey [2]) [1]. Key numbers, all from the paper [1]:

- Only 25 of 48 survey respondents answered the "how long until documentation is updated" question. Apart from testing and quality documents, engineers "rarely" update documents, and when they do it is "several weeks" after the code change.
- Of 45 respondents, 44% somewhat agreed and 24% strongly agreed that "Documentation is always outdated relative to the current state of a software system".
- Of 45 respondents, 53% somewhat agreed and 28% strongly agreed that "Software documentation can be useful, even though it might not always be the most up-to-date".
- Correlation between perceived accuracy and consultation frequency: testing/quality 0.67, low-level design 0.58, requirements 0.43, architectural 0.41, detailed design 0.39, specifications 0.03. The authors read this as "the closer you get to the real code, the more accurate the documentation must be for SEs to use it".
- Interviewees were more likely to trust abstract documents: "The more abstract a piece of documentation, the more likely SEs were to consider it accurate and useful."
- Perception versus behaviour: at the telecom company 6% said they spent considerable time reading documentation and 50% said they spent considerable time consulting source code, but in observation engineers "consulted the documentation only 3 percent of the time: 12 times over 357 logged events".
- Why some documentation stays current: bug-tracker comments ("requires little effort, and maintenance is semiautomatic"), code comments ("short and 'right there'"), test cases ("simple structure and obvious operational value"). "Specifications and requirements, however, are big, complex, and of varied structure."
- The authors reject the laziness explanation: "most SEs aren't lazy; they have discipline of a different sort. They consciously or subconsciously make value judgments".
- Their "bad" list includes "Much mandated documentation is so time consuming to create that its cost can outweigh its benefits" and "Systems often have too much documentation".

Caveats: small samples, 2003 vintage, one telecom company for the observation, and "3% of logged events" is not 3% of time. Do not state it as a universal.

### Aghajani et al. (ICSE 2019, ICSE 2020)

- ICSE 2019 mined 878 documentation-related artifacts (mailing lists, Stack Overflow, issue trackers, pull requests) into a taxonomy of documentation issues [6]. I could not retrieve the paper's frequency table [unverified frequencies].
- ICSE 2020 validated the taxonomy with 146 practitioners (125 from one automation-technology multinational, 21 recruited online), so it is not a random sample [7]. Selected "% of respondents rating the issue important" figures: missing documentation for a new feature or component 69%, outdated or obsolete references 64%, code-documentation inconsistency 59%, outdated installation instructions 54%, outdated examples 51% [7]. "Lack of time to write documentation" was rated important by 65% and was the only process/tool issue most participants called important [7]. Readability's "clarity" was the top individual issue at 88% [7]. I read the table's percentages as "important" ratings from the layout of the extracted text; check against the paper before printing [7].

### Code-comment inconsistency

- Wen et al. (ICPC 2019) mined 1.3 billion AST-level changes across 1,500 Java systems and manually examined 500 commits. Between 13% and 20% of code changes trigger comment updates, which they say "does not imply that in the remaining ~80% of cases code-comment inconsistencies are introduced, but they represent a possibility" [8].
- Fluri et al. (WCRE 2007): in three open-source systems, 97% of comment changes happen in the same revision as the associated code change [reported in secondary form, 41]. Their 2009 extension reports about 90% for six of eight systems [reported in secondary form, 41]. Both numbers measure co-change when a comment is touched, not how many stale comments exist, so they do not contradict the drift claim.

### Architecture and design documents specifically

- Rost, Naab, Lima and Flach Chavez (Fraunhofer IESE report) surveyed 147 industrial participants and reported that "architecture documentation is most frequently outdated, updated only with strong delays, and inconsistent in detail and form" [28]. Sample: 59% Germany, 23% Brazil, 13% Finland; 46% developers; 60.31% response rate but 25.7% completion rate [28]. Their remedy list is "Centralization with powerful tooling and automated generation" and a "single source of architecture documentation" [28].
- Stettina and Heijstek (SIGDOC 2011): 79 professionals in 8 agile teams across 13 countries; more than half rated documentation important or very important, yet felt their projects had too little of it, and did not seem to accept that face-to-face conversation is the most effective way to convey information [29, secondary summary, [unverified] in the original].
- Ko, DeLine and Venolia (ICSE 2007): in observation of 17 developers, questions about design and program behaviour (why code was written a particular way, what it was supposed to do) were the ones most often put off, and work stalled when the only source was an absent coworker [27, secondary summary].
- Roehm et al. (ICSE 2012, extended in TOSEM 2014; 28 observed developers and 1,477 survey respondents): developers use pragmatic strategies, try to avoid comprehension where possible, and share knowledge in informal comments [26, secondary summary]. I could not confirm a documentation-specific percentage [unverified].
- I found no study that measures how often design or architecture documents are opened or how often they are updated through repository telemetry. This gap is worth stating in the essay as such: the claim "the folder no living person has opened twice" is anecdote, not data.

### Why the 2003 finding matters for the draft

The draft says a stale spec "becomes worse than useless, because now you have to check which one is lying". The data say something subtler: engineers kept using out-of-date documents, especially high-level ones, as orientation, and trusted them less the lower the level of detail [1]. The accurate claim is that drift destroys trust in low-level documents faster than in high-level ones, and that the cost is the verification step, not uselessness.

## 2. Reading and understanding versus writing

- Xia et al. (TSE 2018): 78 professional developers, seven projects, 3,148 working hours, instrumented across applications, not just the IDE; developers spend about 58% of their time on program comprehension, and they frequently use web browsers and document editors to do it [3].
- Minelli, Mocci and Lanza (ICPC 2015): IDE interaction data, 738 sessions in the dataset (the paper's text; one secondary source says 740), program understanding reaching "roughly 70%" of IDE time, with roughly 17% on UI interactions and about 5% each on editing and navigation [4]. The two figures are not comparable: one is IDE-only, one is all-application and classified differently [3][4].
- Meyer et al. (TSE 2021; 5,971 Microsoft developers; self-reported minutes for the previous workday; mean day just over 9 hours including 44 minutes of lunch and breaks) [5]. Table 2, all developers: coding (reading or writing code and tests) 15% / 84 min; bug fixing 14% / 74 min; testing 8% / 41 min; specification 4% / 20 min; code review 5% / 25 min; documentation 2% / 9 min; meetings 15% / 85 min; email 10% / 53 min; learning 3% / 17 min; administrative tasks 2% / 12 min; breaks 8% / 44 min [5]. On good days coding is 96 min and on bad days 66 min [5]. The six development-heavy rows I listed sum to 48% [estimate, my arithmetic on the rounded figures].
- Wider spread: the same paper cites earlier work finding about 50% of time writing code (Perry 1994), only about 9% (Goncalves 2011, with 45% collaborating and 32% information seeking), and 61% (Astromskis) [5, citing others]. So "developers only code N hours a day" has no stable N; it depends entirely on the taxonomy and sample.
- Verified caveat on the popular "52 minutes" claim: it is a median of editor telemetry from a vendor's self-selected user community, covering only time in an editor, with an extra 41 minutes a day of "other work in editors" such as reading code and reviewing; methodology details are thin [36]. I would not use it in print, or only with those caveats.
- Newer data: Kumar et al. (ICSE SEIP 2025, Microsoft, 484 complete responses of ~6,000 invited, 8.06% response rate, self-reported): actual week averages roughly 12% communication and meetings, 11% coding, 9% debugging, 6% architecting and designing new systems, 5% code review; the ideal week is roughly 20% coding and 15% architecting and designing; developers wanted less debugging and more code review, documentation and mentoring [34]. Percentages are read off the paper's text; the sample is small and from one company [34].

### Cognitive-load angle

- Miller (1956) is the classic "seven, plus or minus two" [30]; Cowan (2001) argues the working-memory capacity limit is closer to four chunks [31]. Both are about short-term retention of chunks, which supports the principle that nobody can hold a 60-page document in mind while coding.
- I found no study that directly tests "read a long specification, then code" against alternatives. The argument that it fails is an inference from working-memory limits plus the Lethbridge observation that engineers do not read much documentation [1][31] [estimate: inference, not a finding]. The draft should present it as reasoning, not as evidence.
- Not researched in this pass: cognitive load theory proper (Sweller) and program-comprehension neuroimaging; add if the essay leans on the cognitive-load argument.

## 3. The "code is the documentation" stance in agile literature

- Agile Manifesto: "Working software over comprehensive documentation", then "That is, while there is value in the items on the right, we value the items on the left more." Seventeen signatories including Beck, Fowler, Cunningham, Martin and Sutherland [13]. Principles include "The most efficient and effective method of conveying information to and within a development team is face-to-face conversation" (6), "Continuous attention to technical excellence and good design enhances agility" (9), "Simplicity--the art of maximizing the amount of work not done--is essential" (10), and "The best architectures, requirements, and designs emerge from self-organizing teams" (11) [14]. Principle 9 is a manifesto-level endorsement of the draft's steelman of up-front design.
- Fowler, "Code As Documentation" (2005): agile methods treat code as "a major, if not the primary documentation of a software system", because it is "the only one that is sufficiently detailed and precise to act in that role"; but it is not the only documentation and code "is no more inherently clear than any other form of documentation" [15].
- Ambler, Agile Modeling documentation essay: "The fundamental issue is communication, not documentation."; developers rarely trust detailed documentation "because it's usually out of sync with the code"; "Documentation should be just barely good enough."; "Document stable things, not speculative things."; "Create documentation only when you need it at the appropriate point in the lifecycle."; "Update documentation only when it hurts"; "Executable specifications offer far more value than static documentation"; Single Source Information "suggests that you strive to capture information once, in the best place possible"; "the creation and maintenance of a document is a burden on your development team" [16]. The draft's "liability" framing is mine or secondary, not a verified Ambler phrase: I searched for "documentation is a liability" and found "burden" and "Travel Light: every artifact you decide to keep must be maintained" only through a secondary slideshow [unverified wording].
- XP: the Agile Alliance XP page has no documentation section; it emphasises simplicity, face-to-face communication and information radiators [17]. Beck's own sentences on documentation in XP Explained: not verified in this pass [unverified].
- What they recommended instead: (a) face-to-face conversation and collective ownership; (b) tests and working software as evidence; (c) readable code as primary documentation; (d) executable specifications and a single source; (e) just-in-time, minimal, audience-specific documents. Evidence that these did not fully work: agile teams surveyed in 2011 reported too little documentation [29].
- Related, for the "semantic diffusion" thread and the Parnas angle: Parnas and Clements argue no process allows perfectly rational design, so "we can fake it. We can present our system to others as if we had been rational designers" [18]. That is a defence of documenting after the fact for the reader's benefit, and it is closer to the draft's "field guide" than to specification-first.

## 4. Historical attempts to make documentation stay true

- Literate programming (Knuth, The Computer Journal, 1984): "Let us change our traditional attitude to the construction of programs: Instead of imagining that our main task is to instruct a computer what to do, let us concentrate rather on explaining to human beings what we want a computer to do." The practitioner "can be regarded as an essayist, whose main concern is with exposition and excellence of style" [19]. Mechanism: documentation and code share one source file, so they sit together and cannot be separated by a repository boundary. It did not become mainstream; the modern descendant is the computational notebook. A large GitHub study of Jupyter notebooks (1.4 million notebooks) reported that 24% of executed notebooks ran without errors and 4% reproduced the same results, but I only saw those numbers via a forum summary and the "same" criterion has been criticised [unverified; 42]. Lesson: co-location is not enough; without something that re-runs the document, notebooks drift too.
- Model-driven engineering and UML:
  - Petre (ICSE 2013): interviews with 50 professionals in 50 companies, five usage patterns: no UML 35, selective use 11, automated code generation 3, retrofit 1, wholehearted 0; reasons for not using UML included lack of whole-system context, the overhead of learning the notation, and difficulty keeping models synchronised and consistent with code [9]. Counts are from a secondary review; the ICSE page confirms 50 interviewees [9].
  - Whittle, Hutchinson and Rouncefield (IEEE Software 2014): online survey plus 22 semi-structured interviews; productivity gains from code generation ranged "from a 27 percent loss to an 800 percent gain", with most companies seeing 20 to 30 percent, which "easily offset" by training and organisational change; "code generation is a red herring"; the main advantages were "in the support that MDE provides in documenting a good software architecture"; more than 40 modelling languages and 100 tools were reported as regularly used [10]. Their ICSE 2011 and journal studies conclude that organisational and social factors, not technical ones, mostly decide success [11][12]. So the retrospective is mixed: MDE succeeded where adopted incrementally and bottom-up, and failed where imposed top-down, which is a process-tax story, not a pure drift story.
  - I did not find a source for "MDA failed" as a consensus statement; treat that phrase as an interpretation [unverified].
- Executable specifications: Adzic coined "living documentation" for a validated specification-by-example suite (credit to Adzic via a secondary source, [unverified attribution]); Martraire's book generalises it to documentation that "evolves at the same pace as the source code", with "Reliable" among its principles (publisher and review summaries; I did not read the book) [24]. Cost side: Binamungu et al. (SANER 2018) surveyed the BDD community on the challenges of keeping BDD specifications maintainable; I could not retrieve the findings [unverified] [25]. BDD origin (Dan North's "Introducing BDD") could not be retrieved in this pass [unverified].
- Doctests: Python's doctest module is documented as a way "To check that a module's docstrings are up-to-date by verifying that all interactive examples still work as documented." [20].
- Architecture fitness functions: Ford, Parsons and Kua define one as "an objective integrity assessment of some architectural characteristic(s)", borrowed from evolutionary computing [21]. I cite the definition through secondary summaries because the publisher page was not retrievable; check the book before quoting [unverified].
- ArchUnit: "a free, simple and extensible library for checking the architecture of your Java code", run inside ordinary unit tests [22]. It turns a design rule (for example, persistence must not depend on web) into a test that fails the build when violated.
- Docs-as-code: not researched in this pass; no source gathered [unverified].

### Which kind of document can stay true

Reading the above together [estimate: synthesis]:

1. Documents a machine can check against the code: doctests, executable specs, architecture tests. The truth is enforced; drift becomes a red build, not a surprise. The cost moves to maintaining the checks [20][21][22][23].
2. Documents too small and too close to the code to be worth separating: inline comments, test cases, bug-tracker notes [1]. Even these drift (Wen: roughly 13 to 20% of code changes touch a comment) [8].
3. Abstract documents whose claims change slowly: architecture overviews stayed useful when detail went stale [1]. Rost's survey nevertheless found architecture documentation the most frequently outdated [28]; the two are compatible only if "useful" and "current" are kept apart.
4. Documents that are records, not descriptions: a dated decision ("we chose option A because...") cannot be out of date about what was decided, only superseded. This is the logic behind decision records; I did not source a specific ADR reference in this pass [unverified], but Parnas's "fake it" argument is the nearest verified support for writing the rationale for the reader [18].
5. Large, structured, human-maintained specifications separated from the code: the type Lethbridge found engineers would not update [1], and the type MDE tried to keep synchronised by tooling and mostly did not [9].

For the draft: the design docs the author describes (options, tradeoffs, data) sit in category 4, with an expiry date tied to the facts they cite. That is a stronger defence of the author's practice than "agents can now read it".

## 5. The boilerplate and process-overhead tax

- Microsoft (n=5,971): documentation 2% (9 min), specification 4% (20 min), administrative tasks 2% (12 min), meetings 15% (85 min), email 10% (53 min) of a roughly 9-hour day [5]. So explicit documentation time is small; the heavy human costs are comprehension [3] and coordination [5]. This weakens the draft's "every hour formatting NFR sections" framing as a general claim: in modern Microsoft teams, documentation is not where the time goes. It may still describe plan-driven shops, but no source in this report measures those.
- Time Warp: developers wanted more time on documentation in an ideal week, and less on meetings, task management and compliance [34].
- Stripe's Developer Coefficient (2018): 17.3 hours per week dealing with technical debt and bad code, split as 13.5 + 3.8 hours; Harris Poll survey of developers and executives; sample and methodology not verified, and one outlet quotes "approximately four hours a week on bad code" [37, secondary] [unverified]. Not about documentation, so use only for "maintenance eats time", with the caveat.
- "Developers code only X hours a day": see section 2; no stable X, and the 52-minute figure is vendor telemetry [36][5].
- Not found: a peer-reviewed measurement of time spent on process documentation in plan-driven (waterfall-style) organisations. That would be the evidence the draft's third reason ("the boilerplate slowed down the actual work") most needs.
- DORA: documentation quality "drives every technical practice" studied, and amplifies their impact on organisational performance; DORA's own words: "Documentation needs to be actively created and maintained, which takes work." [35]. Correlational survey data; the large percentage lifts shown on the DORA page are as published and I would not quote them without reading the report [35].

## 6. Does AI change documentation economics?

Evidence exists for pieces of the claim; no study tests the whole claim (cheaper to write, read and follow, so heavier up-front documentation now pays).

- Writing: LLM-generated Javadoc was judged by two experts as equivalent to the original in 58.8% of cases and better in 27.7% (142 classes, 273 methods, newer than the models' training cutoffs), and automatic metrics such as BLEU did not track human judgement [43, secondary reading of a workshop paper]. A Python-docstring study of 84,000 generated docstrings found the conclusions depend heavily on the evaluation method [44, secondary]. These cover comments and docstrings, not design documents. I found no study of LLM-generated architecture or design documents [gap].
- Drift detection: DocChecker reports 72.3% accuracy on code-comment inconsistency detection [45]; just-in-time inconsistency work goes back to 2021 [46]. For documentation drift beyond comments (READMEs, guides), I only found tools and blog posts, with no evaluation [gap; grey literature only].
- Docs for agents:
  - llms.txt was proposed by Jeremy Howard on 3 September 2024: "Context windows, while larger than they were, are still too small for most websites in their entirety"; its purpose is to "provide information to help agents use a website" [47]. Claims that crawlers ignore it come from SEO-vendor reports I could not verify [unverified].
  - AGENTS.md describes itself as "a README for agents", "used by over 60k open-source projects" (self-reported), stewarded by the Agentic AI Foundation under the Linux Foundation [48].
  - Gloaguen et al. (arXiv 2602.11988, 2026): "providing context files does not generally improve task success rates"; cost up "over 20% on average"; developer-written files improve resolution by about 4% on average over none, LLM-generated files lower it (about -0.5% on SWE-bench Lite and -2% on AGENTbench), and "agents generally follow instructions present in the context files"; recommendation: "human-written context files should describe only minimal requirements" [38]. Setup: 300 SWE-bench Lite tasks, 138 AGENTbench tasks from 12 repositories with developer-written context files, four models. Their finding that repository overviews do not help is the closest evidence to the draft's design-doc claim, and it cuts against it [38] [estimate: my reading].
  - Lulla et al. (arXiv 2601.20404): 10 repositories and 124 pull requests, median runtime down 28.64% and output tokens down 16.58% with an AGENTS.md, task completion roughly comparable; correlational wording, small sample [39].
- Reading reliability: Liu et al. (TACL): performance "is often highest when relevant information occurs at the beginning or end of the input context" [40]; Chroma's 2025 report on 18 models says reliability falls as input length grows even on simple tasks [49]. This qualifies "the agent reads every word. It does not skim."
- Spec-driven development: Böckeler (on Martin Fowler's site, 15 October 2025) reports Tessl's pitch ("specs, not code, are the primary artifact") and GitHub's "maintaining software means evolving specifications", and is "skeptical that lots of up-front spec design is a good idea" [50]; the second quotation is from a fetch summary, verify verbatim before use [unverified wording].
- Productivity honesty: METR's randomised trial (16 experienced open-source developers, 246 tasks, early-2025 tools) measured a 19% increase in completion time with AI allowed, although developers expected a 24% reduction and afterwards believed 20% [51]. Early-2025 tools, one setting; METR notes that newer tools may do better [51]. DORA 2024 estimates, per secondary coverage, that a 25% increase in AI adoption goes with a 7.5% rise in documentation quality and a 7.2% fall in delivery stability [52, secondary] [unverified].

## Quotable lines (exact, with sources)

1. "Documentation is always outdated relative to the current state of a software system." (survey statement, 68% agreed) [1]
2. "Software documentation can be useful, even though it might not always be the most up-to-date." (81% agreed) [1]
3. "the closer you get to the real code, the more accurate the documentation must be for SEs to use it." [1]
4. "Specifications and requirements, however, are big, complex, and of varied structure." [1]
5. "most SEs aren't lazy; they have discipline of a different sort." [1]
6. "Working software over comprehensive documentation" and "while there is value in the items on the right, we value the items on the left more." [13]
7. "Part of this is classifying the code as a major, if not the primary documentation of a software system." (Fowler) [15]
8. "The fundamental issue is communication, not documentation." and "Documentation should be just barely good enough." (Ambler) [16]
9. "Executable specifications offer far more value than static documentation" (Ambler) [16]
10. "Let us change our traditional attitude to the construction of programs: Instead of imagining that our main task is to instruct a computer what to do, let us concentrate rather on explaining to human beings what we want a computer to do." (Knuth 1984) [19]
11. "The bad news is that, in our opinion, we will never find the philosopher's stone... The good news is that we can fake it." (Parnas and Clements) [18]
12. "code generation is a red herring when it comes to describing MDE" (Whittle et al.) [10]
13. "Documentation needs to be actively created and maintained, which takes work." (DORA) [35]
14. "Surprisingly, we find that providing context files does not generally improve task success rates." (Gloaguen et al.; quoted from a fetch summary of the abstract, verify wording) [38]
15. "Context windows, while larger than they were, are still too small for most websites in their entirety." (llms.txt) [47]

## Figures worth reproducing (redraw from the numbers; licences)

- Lethbridge et al., Figure 1 (time between code change and documentation update, by document type): bar chart; I could not extract the bar values from the PDF text, so redrawing needs a manual read of the figure. Copyright IEEE; redraw from numbers, do not paste the image. Better: a small chart of Table 1 (accuracy-consultation correlation: testing 0.67, low-level design 0.58, requirements 0.43, architectural 0.41, detailed design 0.39, specifications 0.03) [1].
- Meyer et al., Table 2 as a stacked bar of a 9-hour day (coding 84 min, bug fixing 74, testing 41, specification 20, code review 25, documentation 9, meetings 85, email 53, interruptions 24, helping 26, networking 10, learning 17, admin 12, breaks 44, various 21; minutes are rounded means and may not sum exactly) [5]. Facts are not copyrightable but cite the source; the pre-print is IEEE-copyrighted.
- Petre's counts as a five-bar chart: no UML 35, selective 11, automated code generation 3, retrofit 1, wholehearted 0 [9].
- Time Warp: actual versus ideal share of the week for coding, architecting, communication, debugging, documentation, code review; the paper's Figure 2 holds the values and I only have rounded figures from the text (actual: communication ~12%, coding ~11%, debugging ~9%, architecting ~6%, review ~5%; ideal: coding ~20%, architecting ~15%) [34]. Read the figure itself before redrawing.
- Gloaguen et al., resolution rate and cost for none, LLM-generated and developer-written context files: the arXiv HTML page is labelled CC BY 4.0, so reproducing the figure with attribution is permitted; I did not extract the absolute values from Figure 3, so they must be read from the paper [38].
- Xia et al.: a single headline number (58% comprehension); no figure needed [3].

## What I could not verify (so the author does not print it)

- Frequencies in Aghajani 2019's taxonomy; Fluri 2007 and 2009 numbers (secondary only); Roehm 2012 documentation findings; Stettina and Heijstek's original numbers; Ko 2007 percentages.
- Adzic's authorship of the term "living documentation" (Hilton's blog is the source I found [24]); Martraire and Adzic book contents; Ford, Parsons and Kua definition from the book itself; Dan North's BDD article; Binamungu SANER 2018 findings; the Jupyter 24% and 4% figures.
- Any peer-reviewed measure of how often design or architecture documents are read; any measure of overhead in waterfall-style shops; any study of LLM-generated design documents; any evaluated LLM drift detector for READMEs and guides.
- The llms.txt crawler-usage statistics; the DORA 2024 percentages; the Stripe methodology.
- Docs-as-code, cognitive load theory beyond Miller and Cowan, and ADR sources (Nygard) were not researched in this pass.

## REFERENCES

1. [Lethbridge TC, Singer J, Forward A. How software engineers use documentation: the state of the practice. IEEE Software 2003;20(6):35-39](https://doi.org/10.1109/MS.2003.1241364). Text read from a PDF copy: [Duke course copy](https://courses.cs.duke.edu//cps196.1/fall11/classwork/Lethbridge-Singer-Forward-2003.pdf)
2. [Forward A, Lethbridge TC. The relevance of software documentation, tools and technologies: a survey. DocEng 2002 (submitted version)](https://www.eecs.uottawa.ca/~tcl/gradtheses/aforward/papers/aforwarddoceng2002sub.pdf)
3. [Xia X, Bao L, Lo D, Xing Z, Hassan AE, Li S. Measuring program comprehension: a large-scale field study with professionals. IEEE TSE 2018;44(10):951-976](https://doi.org/10.1109/TSE.2017.2734091); abstract via [SMU repository](https://ink.library.smu.edu.sg/sis_research/3779) (via search summary, original abstract not retrieved)
4. [Minelli R, Mocci A, Lanza M. I know what you did last summer: an investigation of how developers spend their time. ICPC 2015](https://sonar.ch/documents/328164/files/Mine2015b.pdf)
5. [Meyer AN, Barr ET, Bird C, Zimmermann T. Today was a good day: the daily life of software developers. IEEE TSE 2021;47(5):863-880](https://doi.org/10.1109/TSE.2019.2904957); pre-print read: [UCL Discovery](https://discovery.ucl.ac.uk/id/eprint/10090507/)
6. [Aghajani E, Nagy C, Vega-Marquez OL, Linares-Vasquez M, Moreno L, Bavota G, Lanza M. Software documentation issues unveiled. ICSE 2019](https://doi.org/10.1109/icse.2019.00122); [review](https://neverworkintheory.org/2021/10/06/software-documentation-issues-unveiled.html)
7. [Aghajani E, Nagy C, Linares-Vasquez M, Moreno L, Bavota G, Lanza M, Shepherd DC. Software documentation: the practitioners' perspective. ICSE 2020](https://doi.org/10.1145/3377811.3380405); PDF read: [USI](https://www.inf.usi.ch/faculty/lanza/PUBS/P/Agha2020a.pdf)
8. [Wen F, Nagy C, Bavota G, Lanza M. A large-scale empirical study on code-comment inconsistencies. ICPC 2019](https://www.inf.usi.ch/faculty/lanza/PUBS/P/Wen2019a.pdf)
9. [Petre M. UML in practice. ICSE 2013](https://2013.icse-conferences.org/content/uml-practice.html); counts from [Never Work in Theory review](https://neverworkintheory.org/2013/06/13/uml-in-practice-2.html)
10. [Whittle J, Hutchinson J, Rouncefield M. The state of practice in model-driven engineering. IEEE Software 2014;31(3):79-85](https://doi.org/10.1109/MS.2013.65); PDF read: [Sheffield copy](https://staffwww.dcs.shef.ac.uk/people/A.Simons/remodel/papers/WhittleMDE_IEEE.pdf)
11. [Hutchinson J, Whittle J, Rouncefield M, Kristoffersen S. Empirical assessment of MDE in industry. ICSE 2011](https://doi.org/10.1145/1985793.1985858); [ICSE page](https://2011.icse-conferences.org/content/empirical-assessment-mde-industry.html)
12. [Hutchinson J, Whittle J, Rouncefield M. Model-driven engineering practices in industry: social, organizational and managerial factors that lead to success or failure. Sci Comput Program 2014;89:144-161](https://eprints.lancs.ac.uk/id/eprint/72508)
13. [Beck K et al. Manifesto for Agile Software Development, 2001](https://agilemanifesto.org/)
14. [Principles behind the Agile Manifesto](https://agilemanifesto.org/principles.html)
15. [Fowler M. Code as documentation, 2005](https://martinfowler.com/bliki/CodeAsDocumentation.html)
16. [Ambler SW. Agile/lean documentation: strategies for agile software development (Agile Modeling)](https://agilemodeling.com/essays/agiledocumentation.htm)
17. [Agile Alliance. Extreme Programming (XP)](https://www.agilealliance.org/glossary/xp)
18. [Parnas DL, Clements PC. A rational design process: how and why to fake it. IEEE TSE 1986;12(2):251-257 (author copy)](https://userweb.cs.txstate.edu/~rp31/papers/Fake_it.pdf)
19. [Knuth DE. Literate programming. The Computer Journal 1984;27(2):97-111](https://doi.org/10.1093/comjnl/27.2.97); text read: [Tufts copy](https://www.cs.tufts.edu/~nr/cs257/archive/don-knuth/web.pdf); [Knuth's book page](https://www-cs-faculty.stanford.edu/~knuth/lp.html)
20. [Python documentation. doctest: test interactive Python examples](https://docs.python.org/3/library/doctest.html)
21. [Ford N, Parsons R, Kua P. Building Evolutionary Architectures, ch. 2 Fitness functions (O'Reilly)](https://www.oreilly.com/library/view/building-evolutionary-architectures/9781492097532/ch02.html); definition via [Thoughtworks Radar](https://www.thoughtworks.com/radar/techniques/architectural-fitness-function) (secondary)
22. [ArchUnit](https://www.archunit.org/)
23. [Adzic G. Specification by Example (publisher summary, via InfoQ)](https://www.infoq.com/articles/specification-by-example-book)
24. [Martraire C. Living Documentation (Addison-Wesley, 2019), publisher page](https://informit.com/livingdoc); [Hilton P. Living documentation principles](https://hilton.org.uk/blog/living-documentation-principles)
25. [Binamungu LP, Embury SM, Konstantinou N. Maintaining behaviour driven development specifications: challenges and opportunities. SANER 2018](https://doi.org/10.1109/SANER.2018.8330207)
26. [Roehm T, Tiarks R, Koschke R, Maalej W. How do professional developers comprehend software? ICSE 2012; extended as On the comprehension of program comprehension, TOSEM 2014](https://unpaywall.org/10.1145%2F2622669); [review](https://neverworkintheory.org/2015/02/13/on-comprehension-of-program-comprehension.html)
27. [Ko AJ, DeLine R, Venolia G. Information needs in collocated software development teams. ICSE 2007](https://faculty.washington.edu/ajko/papers/Ko2007InformationNeeds.pdf)
28. [Rost D, Naab M, Lima CF, Flach Chavez CV. Software architecture documentation for developers: a survey. Fraunhofer IESE report](https://www.iese.fraunhofer.de/content/dam/iese/dokumente/alte-dateien/study_software_architecture_documentation_for_developers_survey-en-fraunhofer_iese.pdf)
29. [Stettina CJ, Heijstek W. Necessary and neglected? An empirical study of internal documentation in agile software development teams. SIGDOC 2011](https://scholarlypublications.universiteitleiden.nl/access/item%3A2872899/view) (abstract-level findings via search summary)
30. [Miller GA. The magical number seven, plus or minus two. Psychological Review 1956;63(2):81-97](https://doi.org/10.1037/h0043158)
31. [Cowan N. The magical number 4 in short-term memory. Behavioral and Brain Sciences 2001;24(1):87-114](https://doi.org/10.1017/S0140525X01003922)
32. [Dagenais B, Robillard MP. Creating and evolving developer documentation: understanding the decisions of open source contributors. FSE 2010](https://doi.org/10.1145/1882291.1882312) (cited for context; findings are abstract-level)
33. [Dan North. Introducing BDD](https://dannorth.net/introducing-bdd/) (not retrieved; listed for follow-up)
34. [Kumar S, Goel D, Zimmermann T, Houck B, Ashok B, Bansal C. Time Warp: the gap between developers' ideal vs actual workweeks in an AI-driven era. ICSE SEIP 2025](https://arxiv.org/pdf/2502.15287)
35. [DORA. Documentation quality capability](https://dora.dev/capabilities/documentation-quality/)
36. [Software.com. Global Code Time Report](https://www.software.com/reports)
37. [ADT Magazine. Study says enterprise misuse of developers costs billions (on Stripe's Developer Coefficient), 2018](https://adtmag.com/Articles/2018/09/10/developer-survey.aspx)
38. [Gloaguen T, Mündler N, Müller M, Raychev V, Vechev M. Evaluating AGENTS.md: are repository-level context files helpful for coding agents? arXiv 2602.11988, 2026](https://arxiv.org/abs/2602.11988); [HTML, CC BY 4.0](https://arxiv.org/html/2602.11988v1)
39. [Lulla JL, Mohsenimofidi S, Galster M, Zhang JM, Baltes S, Treude C. On the impact of AGENTS.md files on the efficiency of AI coding agents. arXiv 2601.20404](https://arxiv.org/abs/2601.20404v1)
40. [Liu NF et al. Lost in the middle: how language models use long contexts. TACL; arXiv 2307.03172](https://arxiv.org/abs/2307.03172)
41. [Fluri B, Würsch M, Gall HC. Do code and comments co-evolve? WCRE 2007](https://doi.org/10.1109/WCRE.2007.21); figures via [secondary search summary; extended study listed at rero](https://doc.rero.ch/record/322204?ln=de)
42. [Pimentel JF, Murta L, Braganholo V, Freire J. A large-scale study about quality and reproducibility of Jupyter notebooks. MSR 2019](https://2019.msrconf.org/details/msr-2019-papers/36/A-Large-scale-Study-about-Quality-and-Reproducibility-of-Jupyter-Notebooks); figures via [Jupyter forum summary](https://discourse.jupyter.org/t/a-large-scale-study-about-quality-and-reproducibility-of-jupyter-notebooks/1360)
43. [LLM-generated Javadoc evaluation, ICSE 2026 workshop paper (Valente group)](https://homepages.dcc.ufmg.br/~mtov/pub/2026-ai-sqe-icse-workshop.pdf) (figures from a search summary, not read in full)
44. [Evaluating large language models for Python docstring generation: models, metrics, and factuality](https://www.doria.fi/handle/10024/194939) (search summary only)
45. [DocChecker: bootstrapping code large language model for detecting and resolving code-comment inconsistencies. arXiv 2306.06347](https://arxiv.org/pdf/2306.06347)
46. [Panthaplackel S et al. Deep just-in-time inconsistency detection between comments and source code. AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/16119)
47. [Howard J. llms.txt proposal](https://llmstxt.org/)
48. [AGENTS.md](https://agents.md/)
49. [Hong K, Troynikov A, Huber J. Context rot: how increasing input tokens impacts LLM performance. Chroma, 2025](https://trychroma.com/research/context-rot)
50. [Böckeler B. Understanding spec-driven-development: Kiro, spec-kit, and Tessl. martinfowler.com, 2025](https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html)
51. [Becker J, Rush N, Barnes E, Rein D. Measuring the impact of early-2025 AI on experienced open-source developer productivity. METR; arXiv 2507.09089](https://arxiv.org/abs/2507.09089)
52. [Google Cloud. Announcing the 2024 DORA report](https://cloud.google.com/blog/products/devops-sre/announcing-the-2024-dora-report) (percentages via secondary coverage, e.g. [heise](https://heise.de/-9999452))
