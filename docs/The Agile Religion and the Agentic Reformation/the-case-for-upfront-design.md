# The case for upfront design: what the evidence says

## Summary for the blog draft

1. The "100x" cost-of-change figure is the weakest leg of the draft's steelman, and the draft's hedge is right: the widely reproduced chart traces to IBM course notes via Pressman's 1987 textbook and its underlying data is, at best, 1970s-era and unpublished [1][2]. Boehm himself later wrote that the factor is "more like 5:1 than 100:1" for small non-critical systems [3]. What survives: late-found defects cost more *tentatively*, for large and critical systems, and for architectural (not local) decisions [1][3].
2. The strongest empirical case for upfront design is not the cost curve but *context-dependence*: Boehm & Turner's five dimensions (size, criticality, dynamism, personnel, culture) and Waterman et al.'s grounded theory (44 participants) both say "how much up-front" depends on requirements instability and technical risk, and that pure BDUF is undesirable while pure emergence is risky [4][5]. The draft's "moves the optimum" framing fits this literature well; the "pick a point on a line" framing is close to Boehm & Turner's own "continuum".
3. Correction for footnote 1: Royce's paper does not merely warn against the one-pass flow. It also argues for *more* documentation, which I recall from the paper but could not re-verify here (see section 1.4). Check before using. What I did verify is the line "I believe in this concept, but the implementation described above is risky and invites failure" via reprints [6].
4. Nuance for "documents rot, nobody reads them": Parnas & Clements (1986) already argued that we can never follow a rational design process, but should *fake it* by writing the documentation as if we had, because it pays in maintenance [7]. That is a 40-year-old version of the draft's thesis: the docs are worth it if cheap to keep consistent.
5. The most credible evidence that rigor pays where change is costly is AWS's TLA+ experience: 10 large systems, bugs "we are sure we would not have found by other means", one with a 35-step error trace that survived design reviews, code reviews and testing, with engineers productive in 2 to 3 weeks [8]. This is an industry experience report, not a controlled study.
6. Evidence that design-doc and ADR cultures work is almost entirely *proponent testimony* (Nygard, Google, Oxide, Rust). I found no controlled study of ADR or design-doc effectiveness (an absence of evidence I could not fully rule out; see section 4.6). The draft should say "practitioners report", not "research shows".
7. Evidence that misalignment is costly exists but is mostly self-reported or correlational: Besker et al. (23% of development time wasted on technical debt, 43 developers, self-reported) [9]; Stripe's Developer Coefficient (secondary reports only; I could not reach the original) [10]; MacCormack et al. and Nagappan et al. on Conway's law (observational, but quantitative) [11][12].
8. The draft's "quadratically cheaper" claim is a modelling assumption (cost of influence = writing cost times reading cost), not an empirical finding. It should be labelled as an argument, and the agent-reads-every-word premise needs its own evidence (out of scope here; the draft's own list flags it).

## 1. The cost-of-change curve

### 1.1 What Boehm claimed

Boehm's curve appears in *Software Engineering Economics* (Prentice Hall, 1981) [13]. The commonly told version: data from 1970s projects at TRW and IBM showed that fixing a requirements error after delivery costs up to 100 times what it costs during requirements. I could not access the book; secondary sources agree on the 1981 origin and the 100x headline [14][15].

Boehm & Basili, "Software Defect Reduction Top 10 List", *IEEE Computer* 34(1), Jan 2001, pp. 135-137 [3]. Item 1 says finding and fixing a problem after delivery is "often" about 100 times more expensive than finding and fixing it during requirements and design. The word "often" was added relative to their 1987 wording. The same item then says one insight shows the escalation factor for small, non-critical systems to be "more like 5:1 than 100:1". I checked this wording only via a search-engine excerpt of a hosted PDF, not the PDF itself (my direct fetch failed), so confirm the exact phrasing before quoting.

### 1.2 The critique

Laurent Bossavit, *The Leprechauns of Software Engineering* (Leanpub, work in progress) [16]. Per secondary coverage, he argues the evidence behind the curve does not meet a research standard, that studies Boehm cited were misrepresented, and that one found a 2:1 ratio in the opposite direction [14]. I could not read the book chapter itself; the chapter text is behind Leanpub, so these points rest on the Techwell summary and an interview.

Quotes verified via The Register's report of Hillel Wayne's investigation [1]:

- Wayne: "There's one tiny problem with the IBM Systems Sciences Institute study: it doesn't exist."
- Bossavit, on the Institute: "an internal training program for employees."
- Bossavit, on the data: "The original project data, if any exist, are not more recent than 1981, and probably older; and could be as old as 1967."
- Wayne, on the broader literature: "the body of research so far tentatively points in that direction" (late defects cost more).

The chain, per the same report: Roger Pressman's 1987 *Software Engineering: A Practitioner's Approach* presented relative cost figures citing "[IBM81]", described as course notes; later publications cited Pressman as authority, hiding the tentative nature of the figures [1]. Wayne's original post is in his newsletter archive [2]. I did not find a separate longer Wayne article on this at the URL I guessed (404), so I rely on the newsletter post as reported by The Register.

A literature survey of defect-detection economics also exists [17]; I only saw its title in search results and did not read it, so I make no claim about its findings.

### 1.3 Kent Beck and the flattened curve

Fowler's "Is Design Dead?" states the assumption: "it is possible to flatten the change curve enough to make evolutionary design work" and describes the classical curve as "as the project runs, it becomes exponentially more expensive to make changes" [18]. The article does not name Beck in the passage I retrieved; I could not verify Beck's own wording in *Extreme Programming Explained* (1999) [unverified]. Secondary sources (Ambler) report that Beck later dropped the curve discussion because no studies backed it; that is a second-hand claim and I could not read the cut-off source [15]. The agile counter-argument in those sources is that the real variable is feedback-cycle length, not project phase [15].

Honest reading: the flattened curve is also an *assumption without empirical grounding*; Fowler explicitly presents it as the condition on which evolutionary design depends [18]. Both sides of the debate rest on thin data.

### 1.4 Royce (relevant to the draft's footnote 1)

Verified, via reprints and slide decks quoting the original: "I believe in this concept, but the implementation described above is risky and invites failure" [6]. He notes the testing phase at the end is the first time timing, storage and I/O are "experienced rather than analyzed", and argues for doing the cycle at least twice [6]. My recollection that Royce also demanded heavy documentation (a "ruthless enforcement of documentation requirements" line) is from memory and I did not verify it. The original paper (WESCON, 1970) should be read directly before the post claims it either way.

### 1.5 Verdict: what survives

- Survives, tentatively: defects (and especially wrong design decisions) cost more the later they are found, in large and critical systems [1][3].
- Does not survive: a universal, precise "100x" exponential, as a law of software [1][14].
- Open: whether agile practices flatten the curve. Evidence is mostly anecdote and argument [15][18]. Do not cite it as established in either direction.
- Draft impact: keep the draft's ⚠️ hedge, cite Boehm & Basili's own 5:1 qualification [3], and shift the weight of the steelman from "cost curve" to "context dependence" (sections 2 and 3).

## 2. Boehm & Turner: home grounds

Boehm & Turner, *Balancing Agility and Discipline: A Guide for the Perplexed* (Addison-Wesley, 2003) [19]. The book argues agile and plan-driven methods lie on a continuum rather than being exclusive, and uses risk analysis to choose between them [19][20]. Cockburn's foreword credits five critical factors: personnel, criticality, size, culture, dynamism [20].

How each tilts, per a course deck built on the book and a Microsoft practitioner's analysis [20][21] (I did not have the book; the detailed polar-chart ratings below are therefore second-hand):

- Size: plan-driven methods grew out of large products and teams; agile fits small ones.
- Criticality: plan-driven for high cost of failure (the search results gave no detail on this factor; the tilt is my reading of the book's framing, unverified).
- Dynamism: high rate of requirements change favours agile.
- Personnel: agile needs more skilled, higher-level people; plan-driven tolerates fewer.
- Culture: agile favours many degrees of freedom; plan-driven needs clear roles and process.

Boehm, "Get Ready for Agile Methods, with Care", *IEEE Computer* 35(1), Jan 2002: I could not retrieve this paper or find a reliable excerpt [unverified]. Do not quote it without reading it.

Draft impact: the five dimensions give the post a principled place to put AI agents. Agents plausibly shift *personnel* (a reader that never skims) and the cost side of *size/criticality* (documents cheaper to produce and keep consistent). That is an argument, not a finding; no study I found measures it.

## 3. Agile and architecture

### 3.1 Waterman, Noble & Allan (ICSE 2015)

"How Much Up-Front? A Grounded Theory of Agile Architecture", ICSE 2015, vol. 1, pp. 347-357, DOI 10.1109/ICSE.2015.54; Distinguished Paper Award; 44 participants from 36 organisations (per the university and the thesis) [5][22][23]. Abstract framing: too little up-front architecture raises risk, too much delays value and makes responding to change difficult [22].

Six forces (paraphrased from a summary of the paper [24]): requirements instability (favours deferring), technical risk (favours up-front), early value (works against heavy up-front), team culture, customer agility, and experience (pattern recognition short-circuits the work). Five strategies: respond to change; address risk; emergent architecture (minimal up-front decisions); big design up front (deemed undesirable in agile contexts); and using frameworks and template architectures [24]. I did not read the paper directly; the summary is Adrian Colyer's, so verify against the paper before quoting.

### 3.2 Fairbanks, Kruchten/Nord/Ozkaya, Fowler

- Fairbanks, *Just Enough Software Architecture* (Marshall & Brainerd, 2010): risk-driven model, in which you do only as much architecture as your most pressing risks call for and apply only techniques that mitigate those risks; process-independent (usable with waterfall or agile) [25]. I could not verify an exact quote; the Methods & Tools excerpt of chapter 3 is the place to look.
- Kruchten, Nord & Ozkaya, "Technical Debt: From Metaphor to Theory and Practice", *IEEE Software* 29(6), 2012, pp. 18-21 [26]. They call for a better definition and a theoretical foundation for technical debt. Their book *Managing Technical Debt* (2019) and an InfoQ interview say their interest began in balancing agile development with architecture thinking [26]. I did not read the article text.
- Fowler, "Is Design Dead?" [18], verified quotes: "In its common usage, evolutionary design is a disaster."; "Indeed I prefer planned design to 'code and fix'."; "I think there is a role for a broad starting point architecture."; YAGNI: "The point of YAGNI is that you don't add complexity that isn't needed for the current stories."; limit: when the balance between planned and evolutionary design alters, "YAGNI becomes good practice (and only then)"; and "In the end the willingness to refactor is much more important than knowing what the simplest thing is right away." (the last attributed within the article to Robert Martin).
- Fowler, "Who Needs an Architect?" (IEEE Software 2003): the line "architecture is the stuff that's hard to change" is attributed to Ralph Johnson. I could not verify wording or attribution [unverified].
- "Architecturally significant requirements": the term appears as "architecturally significant requirements" in the Waterman et al. summary ("challenging or demanding architecturally significant requirements") [24]; a canonical definition source (e.g. Chen, Ali Babar & Nuseibeh) was not retrieved [unverified].

Draft impact: Fowler, the patron saint of agile design, himself prefers planned design to code-and-fix and endorses a broad starting architecture [18]. The draft's "seams, not features" point is well supported; YAGNI's own stated limit is that it applies to features within a known design balance.

## 4. Design-doc and decision-record cultures

### 4.1 Nygard, ADRs (2011) [27]

Verified: the problem, "One of the hardest things to track during the life of a project is the motivation behind certain decisions." Template sections: Title, Context ("the forces at play, including technological, political, social, and project local"), Decision, Status, Consequences ("All consequences should be listed here, not just the 'positive' ones."). Benefit for newcomers: without rationale, a new team member can only accept or change a decision blindly; "better to avoid either blind acceptance or blind reversal". Evidence offered: after the experience report, "All of them have stated that they appreciate the degree of context they received by reading them." This is testimony from his team, not a study.

### 4.2 Google design docs (Ubl) [28]

Verified quotes on goals: "Early identification of design issues when making changes is still cheap."; "Achieving consensus around a design in the organization."; "Ensuring consideration of cross-cutting concerns."; "Scaling knowledge of senior engineers into the organization."; "Form the basis of an organizational memory around design decisions." Verified limits: if the design is not ambiguous "there is little value in going through the process of writing a doc"; docs that are "implementation manuals" are an anti-pattern; and the overhead "may not be compatible with prototyping and rapid iteration". (Quotes are as returned by the fetch tool's extraction; check the page before publishing.)

### 4.3 RFD/RFC processes

- Rust RFC process: provides "a consistent and controlled path for new features to enter the language and standard libraries"; applies to "substantial" changes, not bug fixes [29].
- Oxide RFDs (RFD 1): "Writing down ideas is important: it allows them to be rigorously formulated (even while nascent)."; the RFDs are "a permanent repository for more established ones"; goal of making it easy for "our future selves" to understand past decisions [30].
- Uber, Squarespace, and similar engineering RFC cultures: not retrieved [unverified]. Amazon six-pagers and PR-FAQ: not retrieved [unverified]. Do not cite specifics for these without sources.

### 4.4 Ousterhout, "Design it twice"

*A Philosophy of Software Design*, chapter 11. Secondary sources paraphrase: your first design idea is unlikely to be the best; sketch a second radically different design even if sure there is only one approach; list pros and cons of each [31]. I could not obtain verbatim text from the book; the quoted-looking phrases in my search results were reading notes, so I do not reproduce them as quotes.

### 4.5 Parnas & Clements, "A Rational Design Process: How and Why to Fake It" (IEEE TSE, 1986) [7]

Verified from a PDF copy (the file's metadata author field is wrong, but the text is the paper): "We will never find a process that allows us to design software in a perfectly rational way. The good news is that we can fake it. We can present our system to others as if we had been rational designers and it pays to pretend do so during development and maintenance." Reasons for the pretence include that following an ideal process as closely as possible gets you closer to a rational design than ad hoc work, and it makes progress measurable. In the requirements-document section they list benefits including insurance against personnel turnover, a basis for test plans, and settling arguments among programmers [7].

This is the closest historical ancestor to the draft's thesis, and it explicitly separates *the process you follow* from *the documentation you leave*.

### 4.6 Empirical studies of ADR or design-doc effectiveness

I found none in this session, but my search was limited to a handful of queries and I did not search the architecture-knowledge-management literature (e.g. Tyree & Akerman, Zimmermann, the "architectural knowledge" research stream). Treat "no controlled study exists" as unverified and ask a dedicated search before asserting it.

## 5. Evidence that misalignment is costly

- Besker, Martini & Bosch, "Technical Debt Cripples Software Developer Productivity", TechDebt 2018, DOI 10.1145/3194164.3194178: developers waste on average 23% of development time on technical debt, and are often forced to introduce new debt because of existing debt; survey of 43 developers plus interviews with 16 practitioners; extended in J. Systems and Software 156 (2019) [9]. Caveat: self-reported, small sample, so it indicates a large effect rather than a precise rate. Technical debt is broader than architectural misalignment.
- Stripe, *The Developer Coefficient* (2018, with Harris Poll; over 1,000 developers and 1,000 C-level executives): secondary reports say developers spend over 17 hours a week on maintenance and roughly 4 hours a week on bad code, about $85 billion a year; one summary says 3.8 hours and 42% on tech debt and maintenance [10]. I could not reach the original PDF, so the exact figures (including "17.3") are unverified. Survey-based, not measured.
- Conway's law, MacCormack, Baldwin & Rusnak, "Exploring the duality between product and organizational architectures", *Research Policy* 41(8), 2012, pp. 1309-1324: in matched pairs, the product from the loosely coupled organization was significantly more modular, with differences "up to a factor of eight" in change propagation potential [11]. This supports the mirroring hypothesis, observational but quantitative.
- Nagappan, Murphy & Basili, "The Influence of Organizational Structure on Software Quality: An Empirical Case Study", ICSE 2008, pp. 521-530 (Windows Vista): organizational metrics were significant predictors of failure-proneness, with better precision and recall than churn, complexity, coverage and dependency metrics (per a secondary summary; check the paper) [12].
- The original Conway statement (1968, "How Do Committees Invent?") was not retrieved [unverified].

Draft impact: this supports "architecture is the set of decisions you want everyone to make the same way" only indirectly. The evidence says structure mirrors organization and debt is costly; it does not show that written design docs reduce either.

## 6. Safety-critical and formal-methods domains

- AWS: Newcombe et al., "How Amazon Web Services Uses Formal Methods", CACM 58(4), April 2015 [8][32]. I verified quotes from the earlier September 2014 technical-report version [8]; the CACM wording may differ. Verified: formal methods "have a reputation of requiring a huge amount of training and effort ... so the return on investment is only justified in safety-critical domains", yet "Our experience with TLA+ has shown that perception to be quite wrong." Used on 10 large complex systems; "In every case TLA+ has added significant value, either finding subtle bugs that we are sure we would not have found by other means, or giving us enough understanding and confidence to make aggressive performance optimizations without sacrificing correctness." Engineers "from entry level to Principal" learned it "in 2 to 3 weeks". DynamoDB: bugs "requiring traces of 35 steps". On a data-loss bug: "The bug had passed unnoticed through extensive design reviews, code reviews, and testing". On why specifications beat prose: "talk and design documents can be ambiguous or incomplete, and the executable code is far too large to absorb quickly ... In contrast, a formal specification is precise, short, and can be explored and experimented upon with tools."
- That last quote is directly useful to the blog: it is the AWS authors' own argument about readers of design documents, from before agents.
- DO-178C: I did not retrieve any source in this session. The claim that upfront requirements and traceability are mandatory in avionics software certification is general knowledge but unverified here; cite RTCA DO-178C directly before using [unverified].

## 7. Quotable lines (exact, with source)

- Parnas & Clements: "We will never find a process that allows us to design software in a perfectly rational way. The good news is that we can fake it." [7]
- Royce: "I believe in this concept, but the implementation described above is risky and invites failure." [6] (as reprinted in secondary copies)
- Fowler: "Indeed I prefer planned design to 'code and fix'." and "In its common usage, evolutionary design is a disaster." [18]
- Nygard: "One of the hardest things to track during the life of a project is the motivation behind certain decisions." [27]
- Ubl (Google): "Early identification of design issues when making changes is still cheap." [28]
- Oxide RFD 1: "Writing down ideas is important: it allows them to be rigorously formulated (even while nascent)." [30]
- Wayne: "There's one tiny problem with the IBM Systems Sciences Institute study: it doesn't exist." [1]
- Newcombe et al.: "talk and design documents can be ambiguous or incomplete ... a formal specification is precise, short, and can be explored and experimented upon with tools." [8]
- Agile Manifesto ("there is value in the items on the right"): not re-verified in this session; fetch agilemanifesto.org before quoting.

## 8. Figures worth reproducing

- Boehm's relative-cost-to-fix curve (the 1981 chart, reproduced widely). Copyright: from a Prentice Hall book, so do not copy the image; redraw. Underlying data: no authoritative table obtained. The honest approach is to redraw the *claim* with explicit labelling as "a claim whose data is disputed" [1][14], or to redraw Pressman's relative-cost table, whose provenance is the doubtful part. I could not retrieve the numbers, so any redrawn values would be illustrative and must be labelled as such.
- Boehm & Turner's home-grounds chart: a five-axis polar chart (personnel, dynamism, culture, size, criticality) contrasting agile and plan-driven home grounds, from the book (Addison-Wesley, 2003) [19]. Copyrighted; redraw as an original figure. The axes are verified [20]; the specific axis scales and values I could not verify, so a redraw should use qualitative "tilts toward agile / plan-driven" positions rather than numbers.
- Waterman et al.'s forces and strategies diagram: not retrieved; redraw from the list in section 3.1 if wanted.
- Licences: I did not determine licence terms for any of the above figures. Assume all-rights-reserved unless confirmed. The Rust RFC text and Oxide RFDs are public documents, and I did not check their licences either.

## 9. Open items not verified in this session

Beck's own flattened-curve wording; Boehm 2002 text; Bossavit's chapter text; Fairbanks and Ousterhout verbatim quotes; Fowler "Who Needs an Architect" quote; Uber/Squarespace/Amazon practices; DO-178C; Conway 1968; ADR/design-doc effectiveness studies; Stripe original figures; Boehm & Turner polar-chart values; the Agile Manifesto and Royce documentation passages.

## REFERENCES

1. [The Register, "Everyone cites that 'bugs are 100x more expensive to fix in production' research, but the study might not even exist" (2021)](https://www.theregister.com/2021/07/22/bugs_expense_bs/)
2. [Hillel Wayne, newsletter post on the IBM Systems Sciences Institute study (Buttondown)](https://buttondown.email/hillelwayne/archive/i-ing-hate-science/)
3. [Boehm B, Basili VR. Software Defect Reduction Top 10 List. IEEE Computer 34(1):135-137, 2001 (UMIACS record)](https://umiacs.umd.edu/node/12156)
4. [Boehm B, Turner R. Balancing Agility and Discipline (publisher page)](https://www.informit.com/store/balancing-agility-and-discipline-a-guide-for-the-perplexed-9780132651806)
5. [Waterman M, Noble J, Allan G. How Much Up-Front? A Grounded Theory of Agile Architecture. ICSE 2015 (IEEE Xplore)](https://ieeexplore.ieee.org/document/7194587)
6. [Royce WW. Managing the Development of Large Software Systems (1970), reprint excerpts](https://www.projectmanagement.com/blog-post/80348/the-waterfall-misconception--what-dr--royce-really-said-in-1970)
7. [Parnas DL, Clements PC. A Rational Design Process: How and Why to Fake It. IEEE Trans. Software Eng., 1986 (PDF copy)](https://www.cs.tufts.edu/~nr/cs257/archive/david-parnas/fake-it.pdf)
8. [Newcombe C et al. Use of Formal Methods at Amazon Web Services (technical report, 29 Sep 2014)](https://6826.csail.mit.edu/2019/papers/formal-methods-amazon.pdf)
9. [Besker T, Martini A, Bosch J. Technical Debt Cripples Software Developer Productivity. TechDebt 2018 (Chalmers)](https://research.chalmers.se/en/publication/511450)
10. [Secondary report on Stripe's Developer Coefficient (2018), I-Programmer](https://www.i-programmer.info/news/99-professional/12118-old-and-bad-code-waste-billions.html)
11. [MacCormack A, Baldwin C, Rusnak J. Exploring the duality between product and organizational architectures. Research Policy 41(8), 2012 (Harvard DASH)](https://dash.harvard.edu/handle/1/34403525)
12. [Nagappan N, Murphy B, Basili VR. The Influence of Organizational Structure on Software Quality. ICSE 2008 (Microsoft Research)](https://www.microsoft.com/en-us/research/?p=153188)
13. Boehm BW. Software Engineering Economics. Prentice Hall, 1981 (not accessed; no URL)
14. [Techwell, "What Does It Really Cost to Fix a Software Defect?" (2013)](https://www.techwell.com/techwell-insights/2013/10/what-does-it-really-cost-fix-software-defect)
15. [Ambler S. Examining the Agile Cost of Change Curve](https://agilemodeling.com/essays/costOfChange.htm)
16. [Bossavit L. The Leprechauns of Software Engineering (Leanpub)](https://leanpub.com/leprechauns)
17. [A literature survey of the quality economics of defect-detection techniques (arXiv 1612.04590), listed not read](https://arxiv.org/pdf/1612.04590)
18. [Fowler M. Is Design Dead?](https://martinfowler.com/articles/designDead.html)
19. Boehm B, Turner R. Balancing Agility and Discipline: A Guide for the Perplexed. Addison-Wesley, 2003 (see [4])
20. [Course deck summarising the five factors](https://people.eecs.ku.edu/~saiedian/811/Papers/Stu-Presentations/Agility-vs-Discipline/wishnie-agililty-discipline.pdf) (listed in search results, not opened; the factor list is also from Cockburn's foreword as reported by the search tool)
21. [Microsoft engineer's blog, "Agile AND Plan Driven"](https://learn.microsoft.com/en-us/archive/blogs/dave_froslie/agile-and-plan-driven) (search result, not opened)
22. [Victoria University of Wellington, "How much architecture up front?"](https://ecs.wgtn.ac.nz/Main/Howmucharchitectureupfront)
23. [Waterman M. Reconciling agility and architecture: a theory of agile architecture (thesis)](https://openaccess.wgtn.ac.nz/articles/thesis/Reconciling_agility_and_architecture_a_theory_of_agile_architecture/17007694)
24. [Colyer A. The Morning Paper summary of Waterman et al. (2015)](https://blog.acolyer.org/2015/06/22/how-much-up-front-a-grounded-theory-of-agile-architecture/)
25. [Fairbanks G. A Risk-Driven Model for Agile Software Architecture (excerpt of Just Enough Software Architecture, ch. 3)](https://www.methodsandtools.com/archive/agilearchi.html)
26. Kruchten P, Nord RL, Ozkaya I. Technical Debt: From Metaphor to Theory and Practice. IEEE Software 29(6):18-21, 2012 (citation confirmed via bibliographies; no direct URL accessed)
27. [Nygard M. Documenting Architecture Decisions (2011)](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
28. [Ubl M. Design Docs at Google](https://www.industrialempathy.com/posts/design-docs-at-google/)
29. [Rust RFC 2: the RFC process](https://rust-lang.github.io/rfcs/0002-rfc-process.html)
30. [Oxide Computer, RFD 1: Requests for Discussion](https://rfd.shared.oxide.computer/rfd/0001)
31. [Ousterhout J. A Philosophy of Software Design, ch. 11 (Chinese translation notes, secondary)](https://github.com/Cactus-proj/A-Philosophy-of-Software-Design-zh/blob/main/docs/ch11.md)
32. [Newcombe C et al. How Amazon Web Services Uses Formal Methods. CACM 58(4), 2015 (Amazon Science)](https://www.amazon.science/publications/how-amazon-web-services-uses-formal-methods)
