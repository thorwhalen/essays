# AI coding agents and specifications: what the evidence says about the draft's thesis

Research notes for [thorwhalen/essays#31](https://github.com/thorwhalen/essays/issues/31), compiled 2026-10-08. Sources from 2025-2026 are prioritised and older ones are flagged. Many facts were read on the original page; where a fact comes only from a secondary summary or could not be verified, the text says so.

## Summary for the blog draft

1. The best-supported part of the thesis is the "reader" half: agents do follow written instructions faithfully. The 2026 ETH Zurich study found that "instructions in the context files are well followed by coding agents" [1]. The "agents read every word" claim holds, with a caveat: attention degrades as context grows ("context rot", the "curse of instructions") [14][15].
2. The "so implementations land in the right ballpark" half is NOT supported by controlled evidence, and the one rigorous study of context files points the other way: LLM-generated context files slightly lowered success and raised cost by over 20%, while human-written ones helped modestly (about +4% by the paper's introduction) at up to 19% more cost; repository overviews were "not helpful" [1][2]. The study measures bug-fixing tasks, not architectural alignment, so it neither confirms nor refutes the draft's anecdote. Say "in my experience" and cite it as such. I found no study that measures how often agents violate ADRs while generating code [3][4].
3. What does exist on architecture: "constraint decay" (arXiv 2605.06445) found that agents lose on average 27.28 points of assertion pass rate as structural requirements (architecture, DB, ORM) accumulate [3]; LLMs can detect ADR violations in code well for code-visible decisions but poorly for implicit or deployment decisions [4]. So the ballpark claim needs "for decisions an agent can see and check in code" and "with tests/validators".
4. The "quadratically cheaper" product-of-two-factors intuition should be softened or dropped. Writing got cheaper, but reading did not become free: review of agent output and of agent-written specs is the new bottleneck. Birgitta Böckeler: "I'd rather review code than all these markdown files" [5]. A hands-on Spec Kit trial produced 2,577 lines of markdown and about 3.5 hours of review for 689 lines of code, versus ~24 minutes for plain iterative prompting [6]. The honest framing is that cost moved from reading to reviewing and curating, and the multiplication is unmeasured.
5. The "spec-driven development is waterfall in a trench coat" debate is real and public (Marmelab, HN, Scott Logic) [6][7][8]; Thoughtworks put SDD in Assess (Nov 2025), Spec Kit in Assess (Apr 2026), and context engineering in Adopt (Apr 2026) [9][10][11]. The draft's own caveat (volume of documents outgrows judgement) matches Thoughtworks' "lengthy spec files that are hard to review" [9]. The draft's position (short iterations steered by design knowledge, not a frozen spec pipeline) is close to what the defenders say: the difference is whether the spec is frozen, not whether one exists [12].
6. Prior art exists, so do not claim novelty: Simon Willison ("Write good documentation first and the model may be able to build the matching implementation from that input alone") [13], Duncan Davidson ("Coding agents love decision records") [16], Addy Osmani on specs for agents [17], Andrew Ng's spec-driven development course [18], DORA's "AI-accessible internal data" capability [19], and Thoughtworks' "curated shared instructions" in Adopt [11].
7. Productivity evidence must be handled with care. METR's 2025 RCT: developers expected a 24% speedup, believed afterwards they were 20% faster, and were measured 19% slower (n=16, 246 tasks, early-2025 tools) [20]. The 2026 follow-up is NOT a clean replication: METR calls its data an "unreliable signal" because developers refused to work without AI, and "only very weak evidence" of improvement [21]. DORA 2024 associated AI adoption with lower throughput and stability; DORA 2025 says adoption is now linked to higher throughput but still higher instability [19][22][23].
8. Risks that nuance the thesis are well documented: comprehension/cognitive/intent debt [24][25], skill formation (a 17% lower quiz score for the AI group in Anthropic's RCT) [26], persistent complexity growth after Cursor adoption [27]. Storey's "intent debt" (missing externalised rationale) is actually an argument FOR the draft's ADR habit [25].
9. Draft corrections: (a) "Waterfall" footnote and "defect cost 100x" claims are outside this report's scope; (b) the claim that writing research-heavy docs is now much cheaper needs the caveat that deep-research agents fabricate or mis-cite a non-trivial share of sources (3-13% of cited URLs never existed in one 2026 audit) [28]; (c) the "AGENTS.md standard" point can be strengthened: AGENTS.md went to the Linux Foundation's Agentic AI Foundation on 2025-12-09, with "more than 60,000" adopting projects [29].

## 1. Spec-driven development (SDD): tools and practice

### The tools

- GitHub Spec Kit: announced 2025-09-02 as an open-source (MIT) toolkit; phases Specify, Plan, Tasks, Implement; GitHub describes specs as "living, executable artifacts that evolve with the project" [30]. Its central device is a "constitution" of principles agents must follow; Thoughtworks notes that teams use it to encode scope, versions, coding standards and architecture (e.g. hexagonal architecture) so agents stay within intended boundaries, and reports "instruction bloat and context rot" [10].
- AWS Kiro: launched July 2025 (The New Stack reported it 2025-07-14); a VS Code-based IDE whose spec workflow generates requirements (user stories with acceptance criteria in EARS notation), a design document and a task list [31]. Details are consistent across the secondary reports I found; I did not fetch AWS's own launch page.
- Tessl: Spec Registry (open beta 2025-09-16) and Tessl Framework (closed beta), the latter treating the spec rather than the code as the maintained artifact [9][32]. Secondary sources report a repositioning towards a skills registry on 2026-01-29; I could not verify that on a primary page.
- Others (not examined in depth): BMAD-METHOD and OpenSpec appear in comparison articles [33]; Spec Kit's own site now lists around 30 agent integrations including Kiro [30].
- Andrew Ng and JetBrains released the short course "Spec-Driven Development with Coding Agents" (announced around mid-April 2026) [18].

### Böckeler's analysis (Thoughtworks, martinfowler.com, 2025-10-15)

Levels she proposes [5]: "spec-first" ("A well thought-out spec is written first, and then used in the AI-assisted development workflow for the task at hand"), "spec-anchored" ("The spec is kept even after the task is complete, to continue using it for evolution and maintenance"), "spec-as-source" ("The spec is the main source file over time, and only the spec is edited by the human").

Parallel with model-driven development: "I think another important parallel to look at for spec-as-source in particular is MDD (model-driven development)"; "MDD never took off for business applications, it sits at an awkward abstraction level"; she wonders whether spec-as-source "might end up with the downsides of both MDD and LLMs: Inflexibility" [5].

Review burden: "To be honest, I'd rather review code than all these markdown files"; the files were "repetitive, both with each other, and with the code that already existed" and "very verbose and tedious to review"; "An effective SDD tool would have to provide a very good spec review experience"; one workflow "was like using a sledgehammer to crack a nut" [5]. She also writes that "the general principle of spec-first is definitely valuable in many situations" and that the term "spec-driven development" "isn't very well defined yet" [5]. The article does not use the word waterfall [5].

### Thoughtworks Technology Radar

- Spec-driven development: Assess, Radar vol. 33 (2025-11-05). "Interesting but the workflows elaborate and opinionated"; "Some generate lengthy spec files that are hard to review"; the field "may be relearning the bitter lesson" that hand-written detailed rules for AI do not scale [9] (my paraphrase of the page, which I read via an automated summary).
- GitHub Spec Kit: Assess, 2026-04-15. Mostly brownfield; spec, plan, tasks, coding, review lifecycle "surfaced issues earlier"; rough edges include unnecessary defensive checks and verbose outputs; experienced engineers get the most value [10].
- AGENTS.md: Trial, vol. 33 (Nov 2025); described as in essence a README for agents. The April 2026 edition (vol. 34) warns about instruction bloat in context files and notes research suggesting that hand-written versions are often more effective than LLM-generated ones (reported by a search summary; I did not open the PDF) [11][34].
- Curated shared instructions for software teams: Adopt (Nov 2025, updated Apr 2026); names AGENTS.md as the simplest implementation [11].
- Context engineering: moved from Assess (Nov 2025) to Adopt (Apr 2026); "a foundational architectural concern"; progressive disclosure to avoid context rot [35].

### "SDD is waterfall in disguise"

- François Zaninotto (Marmelab), "Spec-Driven Development: The Waterfall Strikes Back", 2025-11-12: "SDD is a step in the wrong direction"; compares it to Waterfall, which "required massive documentation before coding"; "SDD produces too much text"; "spending 80% of your time reading instead of thinking"; "review time doubles" [7].
- Hacker News discussion of that post (item 45935763): a skeptic says SDD will fail for the reasons waterfall did because code and spec drift apart as a project grows; a defender separates waterfall's long lead time and lack of cheap iteration from the detailed spec itself; others describe specs as agent context, or keep a one-page spec per small module [8]. I saw only part of the thread through a search summary.
- Scott Logic, "Putting Spec Kit through its paces: radical idea or reinvented waterfall?", 2025-11-26 [6]. Figures as printed in the article: feature 1 took 2,577 lines of markdown for 689 lines of code with about 3.5 hours of review; plain iterative prompting for the same feature set took about 8 minutes of agent time and 24 minutes of review. The article's own per-step table does not sum to its stated totals, so treat these as approximate. Verdict: "I'm not sure it is a practical approach"; "For now, the fastest path is still iterative prompting and review, not industrialised specification pipelines" [6]. This is one author's single-case experiment, not a study.
- Defences: Atomic Object, "Spec-Driven Development Is Not Waterfall" [12]; Allstacks, "Spec-Driven Development Isn't Waterfall" [36]. I summarised them from search excerpts only. Their shared argument: Waterfall freezes the document as a gate, SDD treats it as a working tool.
- Gojko Adzic is cited (second-hand, via a notes page) as warning that SDD structure could reintroduce rigidity agile sought to escape; not verified at source [37].

## 2. Context files

### Standardisation

- AGENTS.md was a founding project of the Linux Foundation's Agentic AI Foundation (AAIF), announced 2025-12-09 alongside MCP and goose. The press release calls AGENTS.md "a simple, universal standard that gives AI coding agents a consistent source of project-specific guidance" and says it "has already been adopted by more than 60,000 open source projects and agent frameworks" [29]. Platinum members include AWS, Anthropic, Block, Bloomberg, Cloudflare, Google, Microsoft and OpenAI [29].
- Claude Code reads CLAUDE.md files "naively dropped into context up front", while other files are retrieved just in time with glob and grep [14]. Cursor rules and skills: not researched in depth here (gap).
- GitHub's analysis of over 2,500 repositories' agent files (blog, 2025): the best files combine a specific role, exact commands early, explicit boundaries and examples of good output [38]. The "Always / Ask first / Never" tier is attributed to it by secondary sources and I could not confirm it in the original text.

### Empirical studies of context files

- Gloaguen, Mündler, Müller, Raychev, Vechev (ETH Zurich), "Evaluating AGENTS.md: Are Repository-Level Context Files Helpful for Coding Agents?", arXiv 2602.11988, 2026-02-12, ICLR 2026 listing [1][2]. Abstract: providing context files "does not generally improve task success rates" while "increasing inference cost by over 20% on average"; "instructions in the context files are well followed by coding agents"; "repository overviews, although popular and recommended by model providers, are not helpful"; "context files are useful for specifying non-standard coding practices"; results hold for both LLM-generated and developer-committed files; and "any attempts to improve performance should be rigorously evaluated before deployment" [1]. Details from the HTML version [2]: AGENTbench has 138 instances from 12 repositories (5,694 PRs); agents: Claude Code with Sonnet-4.5, Codex with GPT-5.2 and GPT-5.1 mini, Qwen Code with Qwen3-30b-coder. LLM-generated files lowered success by 0.5% (SWE-bench Lite) and 2% (AGENTbench) on average and raised cost by 20% and 23%; developer-written files gave a 4% average improvement (stated in the introduction) at up to 19% more cost; with documentation stripped from the repository, LLM-generated files improved performance by 2.7%. Conclusion: context files "have only marginal effect on agent behavior, and are likely only desirable when manually written" [2]. Inconsistency to flag: the paper's introduction reports a 3% average decrease for LLM-generated files while section 4.2 reports 0.5% and 2%; quote the paper's range rather than a single figure. Note also that the abstract says the improvement does not hold generally even for developer files, so "human-written helps modestly" is the intro's claim, not the abstract's.
- Lulla, Mohsenimofidi, Galster, Zhang, Baltes, Treude, "On the Impact of AGENTS.md Files on the Efficiency of AI Coding Agents", arXiv 2601.20404 (2026-01-28, v2 2026-03-30) [39]: across 10 repositories and 124 pull requests, AGENTS.md presence was associated with 28.64% lower median runtime and 16.58% fewer output tokens with comparable task completion. Associational; the sampled PRs were small (a secondary summary says under 100 lines and at most 5 files; not verified in the paper).
- MSR 2026 mining study of context files in open source (466 of 10,000 repositories had adopted one): descriptive, found content heavy on build/test/implementation instructions and light on non-functional concerns [40]. Known only through a search summary and the MSR programme title "Context Engineering for AI Agents in Open Source Software"; verify before citing numbers.
- The two efficiency/success studies are compatible: files can speed agents up on small tasks and still not raise success, and the ETH explanation is that files help when they supply knowledge the agent cannot find in the repository, not when they restate it [1][39].

### Studies on architectural constraints and ADRs

- Dente, Satriani, Papotti, "Constraint Decay: The Fragility of LLM Agents in Backend Code Generation", arXiv 2605.06445 (2026-05-07, v2 2026-09-18) [3]: 80 greenfield and 20 feature tasks over eight web frameworks; on average 27.28 points of assertion pass rate lost from baseline to fully specified tasks; worse in convention-heavy frameworks (FastAPI, Django) than Flask for mid-tier models; data-layer defects are the leading cause. This is the closest evidence to "do agents respect architecture", and it is cautionary.
- Su et al., "Evaluating Large Language Models for Detecting Architectural Decision Violations", arXiv 2602.07609 (ICSA 2026) [4]: 980 ADRs from 109 GitHub repositories; models agree substantially and are accurate for explicit decisions checkable in code; "Accuracy falls short for implicit or deployment-oriented decisions". This tests detecting violations, not preventing them; it supports the draft's idea that "an agent can be asked to notice the drift" for code-visible decisions.
- ContextCov (arXiv 2603.00822) argues instruction files are passive and agents drift from them, and converts them into executable checks; it reports 46,000+ checks over 723 repositories but only syntax validity, not violation-catching effectiveness [41]. Known from a search summary only.
- I found no controlled study of agents following ADRs during generation. ADR-for-agents advocacy is practitioner opinion (see section 4).

## 3. Productivity evidence

### METR

- Original RCT: Becker, Rush, Barnes, Rein, "Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity", arXiv 2507.09089, July 2025 [20]. 16 experienced developers, 246 tasks in mature repositories (22k+ stars, 1M+ lines on average), mainly Cursor Pro with Claude 3.5/3.7 Sonnet. Developers forecast a 24% reduction in time, afterwards estimated 20%, measured: AI increased completion time by 19%. Economics experts predicted a 39% reduction and ML experts 38%. METR's caveats: it does not show AI slows most developers or that it fails in other settings; 16 developers; learning effects beyond ~50 hours of Cursor use cannot be excluded [20][42].
- 2026 follow-up (METR blog, 2026-02-24): second study begun August 2025 with 57 developers (10 returning, 47 new), 143 repositories, 800+ tasks, paid $50/hour instead of $150. Estimated change: -18% time (a speedup; CI -38% to +9%) for returning developers and -4% (CI -15% to +9%) for new ones. METR itself says non-participation by developers who will not work without AI "likely biases downwards our estimate of AI-assisted speedup", that 30-50% of surveyed developers withheld some tasks, and that the data is "only very weak evidence" of an increase since early 2025 and gives "an unreliable signal" [21]. Do not cite the -18% as a result.
- METR survey (2026-05-11): 349 technical workers (87 software engineers), February-April 2026; median self-reported value uplift 1.4x-2x, speed change 3x; ~2% response rate on emailed invitations; "Importantly, survey results are not necessarily grounded in reality" [43]. Useful only as perception data, and perception is exactly what the 2025 RCT showed to be unreliable.

### DORA

- 2024 report (Accelerate State of DevOps): InfoQ reports AI adoption associated with a 1.5% drop in delivery throughput and a 7.2% drop in delivery stability, for a 25% increase in AI adoption; the same summaries report +7.5% documentation quality for a 25% adoption increase (GetDX, cusy) [22][44]. All from secondary summaries; the report's own PDF exceeded my fetch limit. Sample sizes quoted in secondary sources conflict (about 3,000 vs 39,000), so I do not give one.
- 2025 report (State of AI-assisted Software Development): about 5,000 respondents; 90% use AI (up 14 points), median about two hours a day; over 80% report improved productivity; 59% report positive effect on code quality; 24% have "a great deal" or "a lot" of trust and 30% "a little" or none ("trust paradox"); "AI adoption is now linked to higher software delivery throughput" (reversing 2024), but Google's summary states it still increases instability (a secondary summary says so; I did not find the number) [23][19]. Framing: "AI's primary role is as an amplifier, magnifying an organization's existing strengths and weaknesses" [45].
- DORA AI Capabilities Model, seven capabilities (Google Cloud blog; built from 78 interviews and a survey of almost 5,000): clear and communicated AI stance; healthy data ecosystems; AI-accessible internal data; strong version control practices; working in small batches; user-centric focus; quality internal platforms [19]. This is a useful supporting source: "AI-accessible internal data" is summarised by Google as "context engineering" connecting AI tools to internal documentation and codebases [46], and small batches is DORA's guard against AI producing "massive blocks of code" that are hard to review [46] (secondary description).
- DORA content licence: the dora.dev research page states content is licensed by Google LLC under CC BY 4.0 unless otherwise specified [45]. Check the specific figure's page before reusing.

### Stack Overflow Developer Survey 2025

Trust in AI accuracy: 32.7% trust (3.1% highly), 45.7% distrust (19.6% highly); experienced developers most skeptical (2.5% highly trust, 20.7% highly distrust). Overall adoption: 78.5% use AI tools at all, 47.1% daily. Top frustration: "AI solutions that are almost right, but not quite" at 66%; 45.2% say debugging AI-generated code takes more time; 30.9% use agents at work [47].

### GitClear

"AI Copilot Code Quality: 2025 Look Back at 12 Months of Data" analysed 211 million changed lines (Jan 2020-Dec 2024); headline "4x more code cloning, 'copy/paste' exceeds 'moved' code for first time in history" [48]. Secondary reports: 8-fold increase in duplicated blocks of five or more lines during 2024, and moved code falling from about 25% to under 10% of changes. I could only read the page summary (full page returned 403), so the detailed figures are unverified. Correlational, mostly private repositories from a vendor with a commercial interest.

### Other large-scale studies

- He, Miller, Agarwal, Kästner, Vasilescu, "Speed at the Cost of Quality: How Cursor AI Increases Short-Term Velocity and Long-Term Complexity in Open-Source Projects", arXiv 2511.04427, MSR 2026 [27]: 807 Cursor-adopting repositories versus 1,380 matched controls (difference-in-differences); "statistically significant, large, but transient increase in project-level development velocity" and "substantial and persistent increase in static analysis warnings and code complexity". The abstract gives no percentages.
- "Agentic Much? Adoption of Coding Agents on GitHub" (arXiv 2601.18341) covers about 130,000 repositories; I did not verify its methods or findings [27].

## 4. "Documentation's new reader is the machine": prior art

- Anthropic, "Effective context engineering for AI agents", 2025-09-29: context engineering is "the set of strategies for curating and maintaining the optimal set of tokens (information) during LLM inference"; the goal is "the smallest possible set of high-signal tokens that maximize the likelihood of some desired outcome"; recall weakens as context grows ("context rot") [14].
- Karpathy: coined "vibe coding" in a post of 2025-02-02 (primary X post not fetched; quote "forget that the code even exists" confirmed only via a secondary page quoting it) [49]; in June 2025 backed "context engineering" as "the delicate art and science of filling the context window with just the right information for the next step", noting that too much or irrelevant context raises cost and lowers performance (via a news summary) [50]. A secondary source says he later preferred "agentic engineering" (Feb 2026); unverified.
- Simon Willison, "Vibe engineering", 2025-10-07: "LLMs reward existing top tier software engineering practices", and: "Write good documentation first and the model may be able to build the matching implementation from that input alone"; "Being able to feed in relevant documentation lets it use APIs from other areas without reading the code first" [13]. This is the strongest prior art for the draft's thesis. He later (2026-05-06) wrote that vibe coding and agentic engineering "are getting closer than I'd like" [51] (title and gist via search summary).
- Kent Beck, "Augmented Coding: Beyond the Vibes" (2025-06-25): in vibe coding "you don't care about the code", in augmented coding you care about "the code, its complexity, the tests, & their coverage"; it "differs fundamentally from 'vibe coding,' where you accept whatever the AI generates"; he reports having to catch design drift and refuse shortcuts [52]. The page does not discuss design docs or ADRs.
- Addy Osmani, "How to write a good spec for AI agents" (2026-01-13): specs as persistent reference; "Simply throwing a massive spec at an AI agent doesn't work"; "curse of instructions" (many directives and the model follows none well); "Planning in advance matters even more with an agent"; start concise rather than "over-engineering upfront" [17]. Notably it does not mention waterfall.
- Andrew Ng: the JetBrains course (April 2026) argues for writing specs instead of vibe coding; his announcement wording is known only second-hand [18]. He earlier called "vibe coding" a misleading name (secondary) [53].
- Duncan Davidson, "Coding agents love decision records", 2026-09-01: records "help them understand the intent behind the code rather than having to infer it"; "When intent is recorded explicitly, an agent is less likely to mistake an implementation detail for a foundational rule"; but agents "cling to outdated decisions", so records should grant permission to challenge them [16]. Other ADR-for-agents posts (all practitioner opinion, no measurements): Stateofme (2025-07-10), BrainGrid, CodeMySpec, Vercel's adr-skill [54].
- Margaret-Anne Storey's "intent debt" (below) is the theoretical counterpart [25].

## 5. AI for research and writing design docs

I did not find a study of speed-ups in writing design docs specifically; the claim "much cheaper" is the author's experience. On accuracy:

- Rao, Wong, Callison-Burch (arXiv 2604.03173, April 2026): across 10 models and agents on DRBench (53,090 URLs) and ExpertQA (168,021 URLs), 3-13% of cited URLs have no archived record and likely never existed, and 5-18% fail to resolve; deep research agents produce far more citations than search-augmented LLMs but hallucinate URLs at higher rates; a URL-checking tool cut non-resolving URLs by 6-79x [28].
- Link validity is not factual support: a secondary summary of arXiv 2605.06635 reports valid-link rates above 94% but only 39-77% factual accuracy against cited sources; I could not open the paper, so treat as unverified [55].
- GhostCite (arXiv 2602.06718): 13 LLMs hallucinated citations at 14.23%-94.93% depending on model and domain, and 1.07% of papers at top venues contain invalid citations, up 80.9% in 2025 (via search summary) [56].
- Implication for the draft: cheaper writing holds, but the author's "data-backed" claim depends on verifying sources; a design doc with unverified citations is the same liability as a stale ADR. The draft's habit of checking sources is the part that does not get automated away.

## 6. Risks that nuance the thesis

- Drift: the HN critique that code and spec diverge as the project grows [8]; ContextCov's premise that agents drift from instruction files [41]; Davidson's warning about stale decisions [16]; Gloaguen et al. show agents do follow instructions, which is a risk when the instruction is stale [1].
- Spec bloat and review burden: Böckeler, Scott Logic, Marmelab and Thoughtworks all report verbose markdown and tedious review [5][6][7][9]; the ETH paper shows unnecessary requirements make tasks harder and recommends minimal human-written files [1]; Thoughtworks reports instruction bloat and context rot in Spec Kit constitutions [10].
- Comprehension and cognitive debt: Osmani (2026-03-14) defines comprehension debt as "the growing gap between how much code exists in your system" and "how much of it any human being genuinely understands" [24]. Storey, "From Technical Debt to Cognitive and Intent Debt", arXiv 2603.22106: cognitive debt is "the erosion of shared understanding across a team" and intent debt is "the absence of externalized rationale", which both developers and AI agents need to change code safely [25]. The arXiv listing date I saw conflicts (March vs 2026-04-06), so cite the ID only. Intent debt supports the draft: ADRs are externalised rationale.
- Skill formation: Anthropic, "How AI Impacts Skill Formation" (arXiv 2601.20245; blog 2026-01-29): RCT with 52 mostly junior engineers learning a new library; quiz score 50% (AI) vs 67% (hand-coded), Cohen's d = 0.738, p = 0.01; speed difference of about two minutes was not significant; debugging questions showed the largest gap; users who delegated code generation scored under 40% while those asking conceptual questions scored 65% or higher [26]. Scope is learning something new, not general skill.
- "Your Brain on ChatGPT" (Kosmyna et al., arXiv 2506.08872, 2025): 54 participants, essay writing, EEG; LLM users showed the weakest connectivity and lowest self-reported ownership, 18 completed the crossover session [57]. It concerns essay writing, not programming, and is a small, preliminary study; use only as an analogy.
- Productivity paradox: perceived speedups exceed measured ones (METR 2025) [20]; DORA 2024 linked adoption to worse delivery metrics, 2025 to better throughput but more instability [22][23]; Cursor study shows transient velocity and persistent complexity [27]; the 2025 Stack Overflow survey shows 66% frustrated by "almost right" solutions [47]. DORA's remedy (small batches, quality platforms) matches the draft's "short iterations" [19].
- Deskilling beyond the studies above: not separately researched (gap).

## Quotable lines (exact, with sources)

- "To be honest, I'd rather review code than all these markdown files." Böckeler [5]
- "the workflow was like using a sledgehammer to crack a nut." Böckeler [5]
- "MDD never took off for business applications, it sits at an awkward abstraction level" Böckeler [5]
- "instructions in the context files are well followed by coding agents" and "repository overviews, although popular and recommended by model providers, are not helpful" Gloaguen et al. abstract [1]
- "context files have only marginal effect on agent behavior, and are likely only desirable when manually written." Gloaguen et al. [2]
- "Write good documentation first and the model may be able to build the matching implementation from that input alone." Willison [13]
- "LLMs reward existing top tier software engineering practices" Willison [13] (this wording came from a search summary; confirm on the page before printing)
- "Decision records help them understand the intent behind the code rather than having to infer it." Davidson [16]
- "the absence of externalized rationale" Storey [25]
- "the smallest possible set of high-signal tokens that maximize the likelihood of some desired outcome" Anthropic [14]
- "Planning in advance matters even more with an agent" Osmani [17]
- "For now, the fastest path is still iterative prompting and review, not industrialised specification pipelines." Scott Logic [6]
- "AI's primary role is as an amplifier, magnifying an organization's existing strengths and weaknesses." DORA 2025 [45]
- "Importantly, survey results are not necessarily grounded in reality." METR [43]
- "only very weak evidence" / "an unreliable signal" METR 2026 update [21]

## Figures worth reproducing

- METR forecast versus observed (2025 RCT): redraw from numbers rather than copying. Numbers: developer forecast 24% time reduction; post-hoc developer belief 20% reduction; economics experts 39% reduction; ML experts 38% reduction; measured 19% increase (n=16 developers, 246 tasks) [20]. Licence: the arXiv paper is CC BY 4.0, so its figures may be reused with attribution; the METR blog page states "© 2026 METR. All rights reserved" and gives no open licence, so do not copy images from the blog [20][21][42]. I could not retrieve the blog's own chart values beyond those headline numbers; the 2026 update page gives +2% to +39% as the confidence interval for the 19% [21].
- METR 2026 follow-up (point estimates and CIs): early-2025 study +19% time (CI +2% to +39%), returning developers -18% (-38% to +9%), new developers -4% (-15% to +9%); a good chart for honesty because the intervals straddle zero [21]. Same "all rights reserved" caveat; redraw.
- DORA: CC BY 4.0 per dora.dev [45]; a redraw of the seven-capability list or of the 2024 adoption-versus-outcomes numbers (-1.5% throughput, -7.2% stability, +7.5% documentation quality per 25% adoption increase) would be sound once those numbers are checked against the 2024 PDF [22][44].
- Gloaguen et al.: a small bar chart of success-rate and cost changes (LLM-generated: -0.5% / -2% success, +20% / +23% cost; developer-written: about +4% success, up to +19% cost) with the 2.7% stripped-docs exception; the paper's licence on arXiv HTML shows CC BY 4.0 [2]. Flag the paper's internal inconsistency in a footnote.
- Stack Overflow 2025: trust 32.7% versus distrust 45.7%, per-group numbers above [47]. Licence not checked.
- GitClear: avoid reproducing until the full report figures are checked [48].

## Cannot be verified or gaps

- The Hacker News thread: only a partial summary read. The Stack Overflow survey's overall adoption percentage, DORA 2024 primary numbers, DORA 2025 instability figure, and the full AWS Kiro launch text were not read at source.
- No controlled study of agents following ADRs during generation was found; no study measures the draft's "right architectural ballpark" claim.
- The "quadratic cost reduction" claim has no source; it is an estimate by the author and should be labelled as such.
- Karpathy's 2025 and 2026 statements and Ng's announcement were not read in the original posts.
- Cursor rules and Agent Skills as context mechanisms, deskilling beyond one RCT, and Royce 1970 / defect-cost curve (listed in the draft's own to-do) were out of scope here.

## REFERENCES

1. Gloaguen T, Mündler N, Müller MN, Raychev V, Vechev M. Evaluating AGENTS.md: Are Repository-Level Context Files Helpful for Coding Agents? arXiv 2602.11988, 2026. [arXiv abstract](https://arxiv.org/abs/2602.11988)
2. Same paper, HTML version (numbers). [arXiv HTML](https://arxiv.org/html/2602.11988v1)
3. Dente F, Satriani D, Papotti P. Constraint Decay: The Fragility of LLM Agents in Backend Code Generation. arXiv 2605.06445, 2026. [arXiv](https://arxiv.org/abs/2605.06445)
4. Su R, Bakhtin A, Ahmad N, Esposito M, Lenarduzzi V, Taibi D. Evaluating Large Language Models for Detecting Architectural Decision Violations. arXiv 2602.07609 (ICSA 2026). [arXiv](https://arxiv.org/abs/2602.07609)
5. Böckeler B. Understanding Spec-Driven-Development: Kiro, spec-kit, and Tessl. martinfowler.com, 2025-10-15. [martinfowler.com](https://www.martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html)
6. Scott Logic. Putting Spec Kit through its paces: radical idea or reinvented waterfall? 2025-11-26. [Scott Logic blog](https://blog.scottlogic.com/2025/11/26/putting-spec-kit-through-its-paces-radical-idea-or-reinvented-waterfall.html)
7. Zaninotto F. Spec-Driven Development: The Waterfall Strikes Back. Marmelab, 2025-11-12. [Marmelab](https://marmelab.com/blog/2025/11/12/spec-driven-development-waterfall-strikes-back.html)
8. Hacker News discussion, "Spec-Driven Development: The Waterfall Strikes Back", item 45935763. [HN](https://news.ycombinator.com/item?id=45935763)
9. Thoughtworks Technology Radar. Spec-driven development (Assess, 2025-11-05). [Radar](https://www.thoughtworks.com/radar/techniques/spec-driven-development)
10. Thoughtworks Technology Radar. GitHub Spec Kit (Assess, 2026-04-15). [Radar](https://www.thoughtworks.com/radar/languages-and-frameworks/github-spec-kit)
11. Thoughtworks Technology Radar. Curated shared instructions for software teams (Adopt). [Radar](https://www.thoughtworks.com/radar/techniques/curated-shared-instructions-for-software-teams)
12. Atomic Object. Spec-Driven Development Is Not Waterfall. [Atomic Object](https://spin.atomicobject.com/spec-driven-vs-waterfall/)
13. Willison S. Vibe engineering. 2025-10-07. [simonwillison.net](https://simonwillison.net/2025/Oct/07/vibe-engineering/)
14. Anthropic. Effective context engineering for AI agents. 2025-09-29. [Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
15. Osmani A. "Curse of instructions" discussion, in [17].
16. Davidson D. Coding agents love decision records. 2026-09-01. [duncandavidson.com](https://duncandavidson.com/agents-love-decisions)
17. Osmani A. How to write a good spec for AI agents. 2026-01-13. [addyosmani.com](https://addyosmani.com/blog/good-spec/)
18. DeepLearning.AI. Spec-Driven Development with Coding Agents (with JetBrains). [course page](https://www.deeplearning.ai/short-courses/spec-driven-development-with-coding-agents/)
19. Google Cloud. Introducing DORA's inaugural AI Capabilities Model. [Google Cloud blog](https://cloud.google.com/blog/products/ai-machine-learning/introducing-doras-inaugural-ai-capabilities-model)
20. Becker J, Rush N, Barnes E, Rein D. Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity. arXiv 2507.09089, 2025 (CC BY 4.0). [arXiv](https://arxiv.org/abs/2507.09089)
21. METR. Uplift update (developer productivity follow-up), 2026-02-24. [METR](https://evals.alignment.org/blog/2026-02-24-uplift-update/)
22. InfoQ. 2024 DORA report. [InfoQ](https://www.infoq.com/news/2024/11/2024-dora-report/)
23. Google. How are developers using AI? Inside our 2025 DORA report. [Google blog](https://blog.google/technology/developers/dora-report-2025/)
24. Osmani A. Comprehension debt. 2026-03-14. [addyosmani.com](https://addyosmani.com/blog/comprehension-debt/)
25. Storey M-A. From Technical Debt to Cognitive and Intent Debt: Rethinking Software Health in the Age of AI. arXiv 2603.22106. [alphaXiv](https://www.alphaxiv.org/abs/2603.22106)
26. Anthropic. How AI assistance impacts the formation of coding skills (paper: How AI Impacts Skill Formation, arXiv 2601.20245). [Anthropic](https://www.anthropic.com/research/AI-assistance-coding-skills)
27. He H, Miller C, Agarwal S, Kästner C, Vasilescu B. Speed at the Cost of Quality: How Cursor AI Increases Short-Term Velocity and Long-Term Complexity in Open-Source Projects. arXiv 2511.04427 (MSR 2026). [arXiv](https://arxiv.org/abs/2511.04427)
28. Rao, Wong, Callison-Burch. Detecting and Correcting Reference Hallucinations in Commercial LLMs and Deep Research Agents. arXiv 2604.03173. [alphaXiv](https://www.alphaxiv.org/abs/2604.03173)
29. Linux Foundation. Linux Foundation announces the formation of the Agentic AI Foundation, 2025-12-09. [press release](https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation)
30. GitHub Blog. Spec-driven development with AI: Get started with a new open source toolkit, 2025-09-02. [GitHub Blog](https://github.blog/2025-09-02-spec-driven-development-with-ai-get-started-with-a-new-open-source-toolkit/)
31. The New Stack. Kiro is AWS's specs-centric answer to Windsurf and Cursor, 2025-07-14. [The New Stack](https://thenewstack.io/kiro-is-awss-specs-centric-answer-to-windsurf-and-cursor/)
32. CodeMySpec. Tessl Review (2026): The Spec-as-Source Bet (secondary). [CodeMySpec](https://codemyspec.com/blog/tessl-review)
33. SSOJet. 7 Spec-Driven Development Tools (secondary comparison). [SSOJet](https://ssojet.com/blog/best-spec-driven-development-tools)
34. Thoughtworks Technology Radar. AGENTS.md (Trial, Nov 2025). [Radar](https://www.thoughtworks.com/radar/techniques/agents-md)
35. Thoughtworks Technology Radar. Context engineering (Adopt, Apr 2026). [Radar](https://www.thoughtworks.com/radar/techniques/context-engineering)
36. Allstacks. Spec-Driven Development Isn't Waterfall. [Allstacks](https://allstacks.com/blog/spec-driven-development-isnt-waterfall-why-the-ai-coding-bottleneck-changed-everything)
37. prg.sh. Spec Driven Development (notes quoting Gojko Adzic; secondary). [prg.sh](https://prg.sh/notes/Spec-Driven-Development)
38. GitHub Blog. How to write a great agents.md: Lessons from over 2,500 repositories. [GitHub Blog](https://github.blog/ai-and-ml/github-copilot/how-to-write-a-great-agents-md-lessons-from-over-2500-repositories/)
39. Lulla JL, Mohsenimofidi S, Galster M, Zhang JM, Baltes S, Treude C. On the Impact of AGENTS.md Files on the Efficiency of AI Coding Agents. arXiv 2601.20404. [arXiv](https://arxiv.org/abs/2601.20404)
40. MSR 2026. Context Engineering for AI Agents in Open Source Software (programme entry; content known via secondary summary). [MSR 2026](https://2026.msrconf.org/details/msr-2026-technical-papers/16/Context-Engineering-for-AI-Agents-in-Open-Source-Software)
41. ContextCov, arXiv 2603.00822 (known via search summary). [arXiv](https://arxiv.org/abs/2603.00822v1)
42. METR. Early-2025 AI experienced open-source developer study (blog), 2025-07-10. [METR](https://metr.org/blog/2025-07-10-early-2025-ai-experienced-os-dev-study/)
43. METR. AI usage survey, 2026-05-11. [METR](https://metr.org/blog/2026-05-11-ai-usage-survey/)
44. GetDX. 2024 DORA report summary (documentation +7.5% per 25% adoption; secondary). [GetDX](https://getdx.com/blog/2024-dora-report)
45. DORA. 2025 research page (amplifier quote; CC BY 4.0 statement). [dora.dev](https://dora.dev/research/2025/dora-report/)
46. Google Cloud. From adoption to impact: putting the DORA AI Capabilities Model to work (secondary descriptions via search summary). [Google Cloud blog](https://cloud.google.com/blog/products/ai-machine-learning/from-adoption-to-impact-putting-the-dora-ai-capabilities-model-to-work/)
47. Stack Overflow. 2025 Developer Survey, AI section. [Stack Overflow](https://survey.stackoverflow.co/2025/ai)
48. GitClear. AI Copilot Code Quality: 2025 Look Back at 12 Months of Data. [GitClear](https://www.gitclear.com/ai_assistant_code_quality_2025_research)
49. Questera. The History of Vibe Coding (quotes Karpathy's 2025-02-02 post; secondary). [Questera](https://www.questera.ai/blogs/history-of-vibe-coding-karpathy-tweet)
50. PureAI. Karpathy puts context at the core of AI coding, 2025-09-23 (secondary). [PureAI](https://pureai.com/articles/2025/09/23/karpathy-puts-context-at-the-core-of-ai-coding.aspx)
51. Willison S. Vibe coding and agentic engineering are getting closer than I'd like, 2026-05-06. [simonwillison.net](https://simonwillison.net/2026/May/6/vibe-coding-and-agentic-engineering)
52. Beck K. Augmented Coding: Beyond the Vibes, 2025-06-25. [kentbeck.com](https://www.kentbeck.com/summaries/augmented-coding-beyond-the-vibes/)
53. The Note. Andrew Ng says vibe coding is a bad name (secondary). [thenote.app](https://thenote.app/post/en/andrew-ng-says-vibe-coding-is-a-bad-name-for-a-very-real-and-exhausting-job-yoc9sca7a0?amp=true)
54. Stateofme. Using Architecture Decision Records with AI coding assistants, 2025-07-10. [blog](https://blog.thestateofme.com/2025/07/10/using-architecture-decision-records-adrs-with-ai-coding-assistants/)
55. IntuitionLabs. Citation reliability in AI literature reviews (secondary, summarising arXiv 2605.06635). [PDF](https://intuitionlabs.ai/pdfs/citation-reliability-ai-literature-reviews.pdf)
56. GhostCite, arXiv 2602.06718. [arXiv](https://arxiv.org/html/2602.06718)
57. Kosmyna N et al. Your Brain on ChatGPT: Accumulation of Cognitive Debt when Using an AI Assistant for Essay Writing Task. arXiv 2506.08872, 2025 (CC BY-NC-SA 4.0). [arXiv](https://arxiv.org/abs/2506.08872)
