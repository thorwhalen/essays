# Critiques, counter-arguments and prior art for "AI and the Accretion Trap"

Research note for the essay seeded in issue 30 of thorwhalen/essays [1]. Compiled 2026-11-06. Scope: the strongest objections to the thesis that experts underestimate "broken" primitives because they miss *scaffolding accretion*, the standard names and prior art for that idea, and the current positions of the best-informed AI skeptics and optimists. Every claim carries a numbered reference; secondary reporting, estimates and unverified items are labelled as such.

## Summary for the essay draft

**The five objections the essay must answer or concede:**

1. **Survivorship, and no evidence of a systematic underestimation bias.** The case list is selected on the outcome; the forecasting literature finds technology-forecast errors roughly symmetric (as likely to overestimate as underestimate time to an event), and most hyped technologies never reach mainstream adoption [2, 3, 4, 5].
2. **As stated, the thesis is unfalsifiable.** "Scaffolding will fix it" can absorb any failure; the essay needs a criterion (Lakatos: does each new layer predict new successes, or only patch old failures?) and at least one dated prediction that could prove it wrong [6, 7, 8].
3. **The scaffolding *is* the formalist tradition.** Congestion control, TLA+, typed contracts, verifiers, LLM-Modulo planners: the compensating layers were built by formal methods, so the honest story is a division of labour between the traditions, not a defeat of one by the other [9, 10, 11, 12].
4. **Scaffolding has costs the thesis hides, and these costs differ from the web's.** A measured verification tax (experienced developers 19% slower), reliability that decays with repetition and task length, per-task inference bills in the thousands of dollars, and a capex-to-revenue gap estimated at $800B a year by 2030; the web's layers were cheap, open RFCs, the LLM's are partly closed and metered [13, 14, 15, 16, 17].
5. **Some limits look structural, and the builders themselves say so.** Prompt injection has no analogue of the parameterised query; the field's own leaders describe generalisation as "dramatically worse than people" and agents as a "decade" away; being right about the endpoint while wrong about the timeline (Amara) is still being wrong for anyone acting now [18, 19, 20, 21].

Also concede: the title has a second, true reading. Accreted scaffolding becomes load-bearing and freezes the primitive (NAT and middleboxes ossified TCP; the robustness principle accumulated tolerated errors), so "accretion trap" can name the cost as well as the blind spot [22, 23, 24].

**Prior art the essay must credit:** Richard Gabriel's "worse is better" (1989–1991), including his own Bourbaki rebuttal and his later "I still can't decide" [25]; Saltzer, Reed and Clark's end-to-end argument (1984) [26] and Isenberg's "Rise of the Stupid Network" (1997) [27]; Christensen's disruptive technology (1995/1997) [28] with Lepore's (2014) and King and Baatartogtokh's (2015) critiques [29, 30]; Chris Dixon's "the next big thing will start out looking like a toy" (2010), which already says "look at products as processes" [31]; Amara's law [21]; Clay Shirky on the Semantic Web (2003) and "Ontology is overrated" (2005) [32, 33]; Joel Spolsky's "Law of Leaky Abstractions" (2002), whose central example is TCP over IP [34]; Rich Sutton's "Bitter Lesson" (2019) [35]; Leopold Aschenbrenner's "unhobbling" (2024) [36]; Zaharia et al.'s "compound AI systems" (2024) [37]; Kambhampati's "LLM-Modulo" (2024) [11]; and OpenAI's "harness engineering" (2026) [38]. **"Accretion trap" and "scaffolding accretion" are not established terms**: searches found the phrase only in astrophysics and in this essay's own GitHub issue, so the author is coining them [1]. The nearest existing names for the mechanism are "unhobbling", "compound AI system", "scaffold/harness", and in software "accretion" in Rich Hickey's positive sense (growing by adding, never breaking) [39].

## 1. Survivorship bias and the base rate

### 1.1 The counter-list

The seed's five cases (IP, HTTP, the uncurated web, JavaScript, the web as an application platform) are all cases where the skeptics lost [1]. A fair sample must include hyped primitives where the skeptics won. Sorting them shows that the counter-list is not uniform, which is itself useful to the essay.

| Technology (peak hype) | Outcome | Class |
| :--- | :--- | :--- |
| Segway PT (2001) | Projected 100,000 sales in its first 13 months; about 140,000 sold in total; production ended 15 July 2020 [40] | A: skeptics right |
| 3D TV (2010–2012) | Share of US TV sales dollars fell from 23% (2012) to 8% (2016); LG and Sony dropped it for 2017 models [41] | A |
| NFTs (2021) | A 2023 study of 73,257 collections found 69,795 with zero market cap (about 95%); the study's author is a crypto gambling site, so treat as indicative [42] | A |
| Metaverse / Meta Reality Labs (2021–) | About $83.6B of cumulative operating losses 2020–2025 against $11.8B revenue (secondary report of Meta's filings) [43] | A, so far |
| Web3 for general computing (2021–2024) | Its most articulate advocate's 2024 book was criticised for failing to name a single blockchain project providing a non-speculative service at scale [44] | A, so far |
| Push / PointCast (1996–1997) | Wired's March 1997 "Push!" cover; push was widely considered dead by end of 1997; PointCast was sold off by 2000 [45] | C: the idea returned as phone push notifications, a reincarnation not a rescue |
| Google Glass (2013) | Enterprise edition sales ended 15 March 2023 [46]; Meta and EssilorLuxottica sold over 7 million AI glasses in 2025 [47] | C |
| Self-driving cars (2016 timelines) | Musk promised a coast-to-coast autonomous drive by end of 2017 and admitted the miss in February 2018 [48]; Waymo reported about 400,000 paid rides a week in February 2026, targeting 1 million by year end [49] | C: skeptics right on timing by a decade, optimists right on direction |
| Semantic Web / RDF (2001–) | Shirky (2003) argued it overestimated deductive reasoning over partial, context-dependent knowledge [32]; its afterlife is Google's closed Knowledge Graph (2012: 500M objects, 3.5B facts) [50] | B: a formalist project lost to a statistical one |
| WAP (1999–2001) | Failed in Europe; Japan's i-mode, built on IP and HTTP, gained over 10 million subscribers in 18 months [51] | B |
| Java applets (1995–) | Deprecated in Java 9 (JEP 289), deprecated for removal in JEP 398 [52]; Java itself thrived on servers | B |
| VRML (1994–) | Never reached mainstream use; superseded by X3D [53] | A/B |
| Gopher (1991–1994) | Declined after the University of Minnesota announced licence fees for some users in 1993; Berners-Lee called it a misstep that drove industry away [17] | B, and the cause was licensing, not technology |
| Second Life (2006–2007) | Fell off the hype curve after 2007; still running with a small user base [54] | A |
| XML-as-everything | Not researched in this pass | — |

Class A: the skeptics were right and no scaffolding rescued it. Class B: a heavyweight, often formalist design lost to a scrappier one, which *supports* the thesis. Class C: dead on the hyped timeline, revived years later in another form.

Two lessons for the essay. First, class C is where the thesis is most vulnerable to "proving too much": any dead technology can later be claimed as "just early" once something similar succeeds (push notifications, Ray-Ban Meta), which makes the thesis immune to its own counter-examples unless it fixes a time window. Second, the most famous heuristic in this family, Dixon's "the next big thing will start out looking like a toy", was later applied by its author to web3 [31, 44]; the same pattern-match that would have caught JavaScript also caught NFTs.

### 1.2 What systematic studies of technology forecasting say

No study measures the quantity the thesis needs, the probability that a technology succeeds *given* that domain experts dismissed it. The nearest evidence:

- **Albright (2002)** scored the 100 technology forecasts in Kahn and Wiener's *The Year 2000* (1967): well under half were judged accurate and timely, but forecasts about computers and communications were roughly 80% correct versus 50% or less elsewhere (figures from secondary summaries; Open Philanthropy's re-evaluation warns many forecasts were too vague to score) [55, 56].
- **Mullins (2012) for the US Office of the Secretary of Defense** analysed 2,279 forecasts from 300 documents and found forecasts *neither optimistic nor pessimistic* on balance [3]. The peer-reviewed version, **Fye et al. (2013)**, found only method and time horizon affected accuracy, that forecasts were about equally likely to overestimate and underestimate the time to an event, that "experts are best at predicting if an event will occur", and that computer and autonomous-technology forecasts were more accurate than others [2].
- **Kott and Perconti (2018)** assessed 1990s military-technology forecasts for 2020: mean expert-assessed accuracy 0.76, with informational technologies around 87% versus 65% for physical ones (split from Kott's later summary) [57].
- **Tetlock (2005)** tracked 284 experts making tens of thousands of political and economic forecasts (82,361 in most accounts; some summaries give about 28,000) and found them barely better than chance, with "hedgehogs" committed to one big idea doing worst [58, 59]. A single grand pattern such as "broken primitives always win" is a hedgehog's idea.
- **Armstrong (1980), "The Seer-Sucker Theory"**: beyond a low threshold, more expertise did not improve forecasts of change [60].
- **Schnaars and Berenson (1986)** and Schnaars's *Megamistakes* (1989) examined growth-market forecasts published 1960–1979; a contemporary review reports that fewer than a quarter of the predicted growth markets materialised (secondary figure) [5, 61].
- **Hype cycles.** The Economist (August 2024) traced Gartner's placements and found only about a fifth of breakthrough technologies follow the full hype–trough–adoption path, and that six in ten technologies that fall into the trough never rise again (as reported secondhand) [4]. Michael Mullany's 2016 review of 20 years of Gartner hype cycles reportedly found about a fifth of listed technologies vanished without going mainstream (secondary) [62]. Dedehayir and Steinert (2016) found Gartner's own data inconsistent with the hype-cycle model [63].
- **AI predictions specifically.** Armstrong and Sotala (2012) found AI-arrival predictions cluster at 15–25 years out regardless of date; their headline claim that experts did no better than non-experts was later traced to a spreadsheet error [64, 65].

**Estimate (labelled as such).** Combining the hype-cycle and growth-market figures, somewhere between one in five and one in two loudly hyped technologies reach mainstream use on anything like the predicted timeline. The forecasting studies give no support to a *directional* bias in which experts systematically underestimate. What they do support is a domain effect: forecasts about information technologies have been more accurate, and more often right about capability growth, than forecasts about physical technologies [55, 2, 57]. That is the defensible core of the thesis: not "experts underestimate broken technologies" but "in software and networking, bets on cheap, widely deployed primitives improving through layered tooling have a better track record than in hardware".

**A rescue the essay can adopt.** The seed's cases share a condition the counter-list mostly lacks: the primitive was *already deployed and growing* when the critique was made (IP, HTTP and JavaScript all shipped by default to millions). Conditioning on adoption momentum and near-zero marginal cost of trying is what separates IP and JavaScript from the Segway and 3D TV. LLMs meet that condition; the essay should say so explicitly and claim no more.

### 1.3 AI's own history contains a failed accretion

The 1980s expert-systems boom was itself an accretion strategy around a primitive (rule-based inference): knowledge engineering, shells and dedicated Lisp machines. It collapsed around 1987 through brittleness on inputs slightly outside the design, maintenance costs that outran the value delivered, the collapse of the Lisp-machine market, and the failure of Japan's Fifth Generation project [66]. The essay should name this case: the accretion was real and the primitive still lost, to cheaper general-purpose hardware and later to statistical methods, which is Sutton's "bitter lesson" [35].

## 2. The thesis proves too much

### 2.1 The logical problem

"The limit will be engineered around" is compatible with every observation: a failure today is "the naked primitive", a success is "the scaffolding working". Lakatos's account of research programmes names the pattern: a hard core protected by a belt of auxiliary hypotheses; the programme is *progressive* if each modification predicts new facts that are then confirmed, and *degenerating* if modifications only explain away anomalies [6]. Ptolemaic epicycles are the standard example of accretion that kept a wrong core alive, and the comparison is already being made informally about RAG and agent harnesses [67]. A 2025 position paper argues that scaffolding can mask or mitigate LLM pathologies but cannot remove them without losing the generative breadth that makes LLMs useful [68].

### 2.2 The live dispute, as an example of the shape

- **Apple, "The Illusion of Thinking" (June 2025)** tested reasoning models on puzzles of controlled difficulty (Tower of Hanoi, River Crossing, Blocks World) and reported complete accuracy collapse past a complexity threshold, with reasoning effort *falling* as problems got harder [7].
- **The rebuttal**, "The Illusion of the Illusion of Thinking" (Lawsen, co-written with Claude Opus), attributed much of the collapse to output-token limits (a 15-disk Hanoi solution needs over 32,000 moves) and to River Crossing instances that were unsolvable as posed; asked for a program instead of a move list, models succeeded [8].
- **Marcus** read the Apple paper as confirming that "their generalizations tend to break down outside that distribution" and that "LLMs are no substitute for good well-specified conventional algorithms" [69].

Note the shape: the rebuttal's best move (let the model write code) *is* a scaffolding move, and Marcus's reply (the conventional algorithm does the work) is the formalist reading of the same fact. Both sides agree on what happened and disagree on who gets the credit. The essay should present this as the paradigm case rather than as a win for either side.

### 2.3 Serious critics who already accept scaffolding

- **Kambhampati et al. (ICML 2024)** argue that autoregressive LLMs cannot plan or self-verify, but can serve as candidate generators inside "LLM-Modulo" frameworks with external model-based verifiers; they reject both the "right prompt suffices" optimism and the "LLMs are only translators" pessimism [11]. This is the thesis's own architecture, endorsed by a leading skeptic, with the guarantee located in the verifier.
- **Kambhampati et al. (2025)** argue that calling intermediate tokens "reasoning traces" is "wishful", has little supporting evidence and "engenders false confidence" [70].
- **Melanie Mitchell** (Science, March 2025) argues that "reasoning" in these models is being equated with generating plausible-sounding steps, and that robustness on open-ended tasks is largely untested [71, 72].
- **Narayanan and Kapoor**, "AI as Normal Technology" (April 2025), treat AI as a general-purpose technology like electricity: "To view AI as normal is not to understate its impact", but "Diffusion occurs over decades, not years", and they name a "capability-reliability gap" that blocks useful agents [73]. Their earlier *AI Snake Oil* (2024) separates generative AI, about which they are relatively positive, from predictive AI used to make decisions about people, which they call largely snake oil [74]. Their position is the strongest version of "Amara's law applies": the essay's endpoint may be right while every near-term schedule is wrong.

### 2.4 What would count as evidence that a limit is fundamental

The essay is stronger if it states its own falsifiers. Candidates, each with precedent:

1. **An impossibility result, not just a failure.** The CAP theorem (Gilbert and Lynch 2002) was never engineered away; systems routed around it by choosing which guarantee to weaken, and Brewer later described the design space rather than an escape [75, 76]. If a limit of LLMs is a theorem (for example, that a single token stream cannot reliably separate instructions from data), scaffolding can contain it but not remove it [18].
2. **The competence migrates into the scaffold.** If reliability comes entirely from an external verifier, compiler or solver, the primitive has become a proposal generator, not the substrate (Kambhampati's position) [11].
3. **Reliability cost grows faster than capability.** If the cost of reaching k-of-k success rises steeply with k and with task length, accretion hits diminishing returns (see 3.2) [14, 77].
4. **Degenerating patches.** Layers that fix yesterday's benchmark failure but predict no new success [6].
5. **A persistent gap between benchmark and economic impact.** Sutskever: "the economic impact seems to be dramatically behind" [19].

A dated prediction the essay could commit to, in the spirit of the thesis: by a stated year, scaffolded agents reach a stated pass^k reliability on a stated class of multi-hour tasks at a cost below a human's. If that fails, the thesis loses; if it holds, the skeptics owe it a concession.

## 3. Scaffolding has costs the thesis hides

### 3.1 The verification tax

- **METR's randomised trial (July 2025)**: 16 experienced open-source developers, 246 tasks in repositories they knew well (about five years' experience each). They predicted AI would cut completion time by 24%; with AI allowed they took 19% *longer*; afterwards they still believed AI had sped them up by about 20% [13]. The authors caution against generalising beyond early-2025 tools and this setting.
- **METR's 2026 follow-up** reportedly judged its new data an unreliable signal, mainly because developers increasingly refused to work without AI, which biases the estimate downward; METR said it still believes tools speed developers up more than in early 2025 (secondary account of METR's February–March 2026 post) [78].
- The perception gap (feeling 20% faster while measuring 19% slower) is the part the essay cannot wave away: scaffolding that *feels* productive is not evidence that it is.
- The pre-LLM precedent is Sculley et al. (2015): in real machine-learning systems only a small fraction of the code is the model, the rest is glue, pipelines and configuration that accrue "hidden technical debt" [79]. Spolsky's 2002 "Law of Leaky Abstractions" makes the same point with the thesis's own flagship: TCP's reliability abstraction over IP leaks, so every abstraction layer adds something new to understand when it fails [34]. Brooks's distinction between essential and accidental complexity is the general frame [80].

### 3.2 Reliability decays with repetition and length

- **τ-bench (Sierra, 2024)** introduced pass^k, success on all k attempts at the same task: GPT-4o scored about 61% pass^1 on the retail domain and about 25% pass^8 [14, 81].
- **METR's time-horizon study (March 2025)** found the length of tasks agents complete with 50% reliability doubling about every seven months since 2019 [82]. This is the best single datum *for* the thesis, but it is a 50% threshold; the essay should not quote it as if it were a reliability guarantee.
- **LeCun's compounding-error argument** (2023): if each token has an independent chance e of leaving the set of correct answers, an n-token answer is correct with probability (1−e)^n, so autoregressive generation "diverges exponentially" (reported from his slides and posts; critics reject the independence assumption) [83]. Toby Ord's analysis that agent failure behaves like a constant hazard per unit of human task time is a more empirical version (summarised by Epoch AI, which disputes that the cost burden will persist) [77].

### 3.3 Security: scaffolding adds attack surface

Simon Willison's "lethal trifecta" (June 2025): an agent that combines access to private data, exposure to untrusted content and the ability to communicate externally can be made to exfiltrate data by prompt injection; he argues vendors will not solve this and users must avoid the combination [18]. The web had SQL injection and fixed it structurally with parameterised queries; no equivalent separation of instruction and data exists for a model that reads both as tokens. This is the strongest candidate for a structural, not transitional, limit.

### 3.4 Compute, energy and money

- **Test-time compute.** On ARC-AGI-1, OpenAI's o3 preview scored 75.7% at about $26 per task and 87.5% with 172 times the compute, at about $4,560 per task in ARC Prize's table [15]. In ARC Prize 2025, the top commercial model scored 37.6% on ARC-AGI-2 at $2.20 per task, while a refinement harness around Gemini 3 Pro reached 54% at about $30 per task [84]. That last figure is evidence *for* scaffolding and *for* its cost at once.
- **Energy.** The IEA projects data-centre electricity demand to more than double to about 945 TWh by 2030, with AI-optimised data centres more than quadrupling [85].
- **Capex.** The four largest hyperscalers spent roughly $410B in 2025; guidance after Q1 2026 implied about $725B for 2026, and a later estimate including SpaceX put 2026 at $803–828B, with bank forecasts above $1 trillion for 2027 (secondary reporting of company guidance) [86].
- **Revenue gap.** Sequoia's David Cahn (2024) asked "AI's $600B question": the annual revenue needed to justify the hardware [87]. Bain (September 2025) estimated $2 trillion of annual revenue would be needed by 2030 to fund the projected compute, and an $800B annual shortfall even after AI-driven savings [16].
- **Enterprise returns.** MIT NANDA's "GenAI Divide" (July–August 2025) reported that 95% of organisations got zero return on $30–40B of enterprise GenAI spend [88]. Critiques: the figure rests largely on 52 interviews and 153 conference survey responses, measures P&L impact within six months, and was labelled preliminary by its authors; a Wharton survey found 74% reporting positive returns, on different definitions [89, 90]. The essay should cite it, if at all, with these caveats.
- **The bubble argument.** Ed Zitron's "Hater's Guide to the AI Bubble" argues the leading labs lose billions on thin revenue and that hyperscaler AI revenue is concentrated in two customers [91]; The New Stack assessed which parts of his case hold up [92]. In October 2026 The Conversation described the circular financing in which chip makers invest in their own customers' data centres [93]. Sam Altman himself, in August 2025: "When bubbles happen, smart people get overexcited about a kernel of truth", and yes, investors are overexcited [94].

The essay's own frame can absorb this: the dot-com crash destroyed investors and not the web. But the essay must say so explicitly, because "the primitive becomes the substrate" and "this year's valuations are justified" are different claims, and only the first follows from the thesis.

### 3.5 Was the web's scaffolding special because it was cheap and open?

The web's compensating layers were RFCs, free reference implementations and open-source servers; the clearest counter-case is Gopher, which lost momentum after its owner floated licence fees in 1993 [17]. LLM scaffolding is mixed: frontier models are metered APIs, but the protocols are being opened, with Anthropic's MCP, OpenAI's AGENTS.md and Block's goose placed under the Linux Foundation's Agentic AI Foundation in December 2025, and AGENTS.md reportedly adopted by over 60,000 projects [95]. The honest claim: the *glue* is following the web's open path; the *primitive* is not, and a primitive whose every call is billed is a different substrate from one whose marginal use is free.

### 3.6 The other accretion trap: ossification

Accreted layers become load-bearing. NAT and other middleboxes, added to compensate for IPv4 exhaustion and security gaps, broke the end-to-end principle and froze what transport protocols could be deployed [22]; QUIC encrypts nearly all of its transport headers partly to escape that ossification [24]. The IAB's RFC 9413 (2023) argues that Postel's robustness principle, the founding pragmatist rule, let tolerated errors accumulate until they became de facto requirements [23]. A practitioner guide to agent harnesses makes the same warning: compensatory mechanisms that become load-bearing require a rewrite to remove [96]. The title "AI and the Accretion Trap" can carry this meaning deliberately: the trap is not only that experts miss the accretion, but that what accretes around today's LLMs will constrain tomorrow's.

## 4. The formalists were often right too

- **TCP congestion collapse.** In October 1986, throughput between LBL and UC Berkeley, 400 yards apart, fell from 32 kbps to 40 bps; Jacobson and Karels's 1988 fix (slow start, congestion avoidance, round-trip estimation) was careful control engineering, derived from conservation-of-packets reasoning [9]. The pragmatist primitive was rescued by formalist method.
- **Formal methods at scale.** Newcombe et al. (CACM, April 2015) report that AWS engineers used TLA+ on ten large systems and found subtle design bugs in S3, DynamoDB, EBS and an internal lock manager that testing had missed [10].
- **Transactions came back.** Google's Spanner paper (OSDI 2012): "We believe it is better to have application programmers deal with performance problems due to overuse of transactions as bottlenecks arise, rather than always coding around the lack of transactions" [97, 98]. The NoSQL era's retreat from ACID was partly reversed by the company that had led it.
- **Virtual circuits came back inside IP.** MPLS (RFC 3031, 2001) introduced label-switched paths in carrier cores, a circuit-like mechanism under the best-effort network [99].
- **REST's uniform interface was supplemented by typed contracts.** gRPC (schema-defined RPC) and GraphQL (typed queries) both came out of the largest web companies in 2015 [100, 101].
- **Types won the browser.** TypeScript became the most-used language on GitHub by monthly contributors in August 2025, and GitHub attributes part of the rise to typed code working better with AI coding agents [12]. The thesis lists TypeScript as scaffolding; it is the formalist tradition's return, and it is now scaffolding *for LLMs*.
- **Gabriel himself.** The author of "worse is better" attacked it under the pseudonym Nickieben Bourbaki ("Worse Is Better Is Worse", winter 1991–92), was rebutted again in "Is Worse Really Better?" (1992), wrote both a for and an against position paper for an OOPSLA 2000 panel, and summarises: "I still can't decide" [25, 102].

**The honest version of the thesis.** Both traditions won their layer. The pragmatist primitive won distribution (it was everywhere, cheaply, first); the formalist tradition won the guarantees layered on top (congestion control, transactions, types, verified designs). In the LLM case the same split is visible in Kambhampati's LLM-Modulo architecture and in the industry's own "harness engineering", where linters and checks constrain and correct the agent [11, 38]. The essay's diagram ("probabilistic core, deterministic harness") already says this; the prose should stop calling it a defeat of the formalists.

## 5. Terminology and prior art

| Name | Who, when | What it says | Relation to the thesis, and its critics |
| :--- | :--- | :--- | :--- |
| End-to-end argument | Saltzer, Reed and Clark, 1984 | Put function at the endpoints; the network stays simple [26] | The mechanism behind cases 1–2 |
| Stupid network | David Isenberg, AT&T, 1997 | Intelligence at the edges beats an "intelligent network" [27] | An insider making the thesis's case 1 at the time |
| Worse is better | Richard Gabriel, 1989 talk, 1991 *AI Expert* | Simple, incomplete designs spread faster and get improved later [102] | The closest single precedent; its author's ambivalence is the critique [25] |
| Disruptive technology / innovation | Bower and Christensen 1995; Christensen 1997 | Cheaper, worse products improve from below and displace incumbents [28] | Explains why incumbents dismiss; Lepore (2014) attacked its case evidence [29]; King and Baatartogtokh (2015) found only about 9% (7 of 77) of Christensen's own cases had all four elements of the theory [30, 103] |
| Amara's law | Roy Amara, usually dated 1978 (attribution uncertain) | Overestimate short-run effects, underestimate long-run [21] | The thesis's timing escape hatch; untested empirically |
| Gartner hype cycle | Jackie Fenn, Gartner, 1995 | Trigger, peak, trough, slope, plateau | Empirically weak: few technologies follow it [4, 63] |
| Leaky abstractions | Joel Spolsky, 2002 | All non-trivial abstractions leak; TCP over IP is the example [34] | The cost of accretion, stated with the thesis's own case |
| Blub paradox | Paul Graham, 2001 | Programmers can see down the power continuum, not up [104] | A precedent for "expertise as blind spot", about languages |
| Ontology is overrated | Clay Shirky, 2005 (and Semantic Web essay, 2003) | Pre-defined categories lose to links and tags [33, 32] | The direct precedent for case 3 (Yahoo vs PageRank) |
| The next big thing looks like a toy | Chris Dixon, 2010 | Judge products as processes, by trajectory [31] | Same idea; same author later applied it to web3 [44] |
| Bitter lesson | Rich Sutton, 2019 | General methods that scale with compute beat hand-built knowledge [35] | The AI-internal version; cuts against hand-built scaffolding too |
| Unhobbling | Leopold Aschenbrenner, 2024 | Gains from RLHF, chain of thought, tools and agents on top of the base model [36] | Nearly the thesis's "scaffolding accretion" under another name |
| Compound AI systems | Zaharia, Khattab et al., BAIR, Feb 2024 | State-of-the-art results increasingly come from systems of models, retrievers and tools [37] | The primitive-versus-system distinction, from the systems community |
| LLM-Modulo | Kambhampati et al., 2024 | LLM generators plus sound external verifiers [11] | Scaffolding, framed by a skeptic |
| Harness engineering | OpenAI (Lopopolo), Feb 2026 | Constrain, inform, verify and correct agents with linters and checks [38] | The industry's current word for the scaffold |
| Accretion | Rich Hickey, "Spec-ulation", 2016 | Grow software by adding, never by breaking [39] | The only established software use of "accretion" found |
| Wardley evolution | Simon Wardley, 2004–2005 | Components move from genesis to custom to product to commodity [105] | Possibly the "Simon Wariyar" in the brief; models how primitives commoditise |
| Platform shift | Benedict Evans, recurring; "AI Eats the World", Nov 2025 | Every 10–15 years a platform shift; with LLMs "we don't know how much better this could get" [106] | A careful optimist's uncertainty, useful as a foil |

Not found or not verified in this pass: a specific Venkatesh Rao text naming this pattern; "innovator's blind spot" as an established term (it appears only as a generic phrase). **"Accretion trap" and "scaffolding accretion": no prior use in technology writing was found** (searches return astrophysics and this essay's issue) [1].

## 6. The best-informed skeptics and optimists, 2025–2026

**Skeptics**

- **Gary Marcus.** After the Apple paper (7 June 2025): "their generalizations tend to break down outside that distribution"; "LLMs are no substitute for good well-specified conventional algorithms" [69]. On GPT-5 (9 August 2025): "GPT-5: Overdue, overhyped and underwhelming" (post title) [107]. Position: neurosymbolic systems are needed; he concedes LLM usefulness for coding, brainstorming and writing.
- **Yann LeCun.** Long argued autoregressive LLMs are a dead end for human-level AI (his 2022 "A Path Towards Autonomous Machine Intelligence" proposes JEPA world models) [108]. Left Meta at the end of 2025; his Paris startup AMI Labs raised a $1.03B seed at a $3.5B pre-money valuation in March 2026 to build world models [109]. Note for the essay: the seed's table files LeCun among "classical ML" skeptics; he is a deep-learning pioneer whose objection is to one architecture, not to connectionism [1].
- **Emily Bender and Alex Hanna**, *The AI Con* (2025): chatbots are "synthetic text extruding machines" producing text "without any intent or thinking mind behind it"; "AI" is a marketing label that launders harms to workers and the environment (quotations as reported in reviews) [110].
- **Melanie Mitchell.** Science column (March 2025) and Substack: reasoning claims rest on plausible-looking steps; evaluate abstraction beyond accuracy (paraphrase) [71, 72].
- **Subbarao Kambhampati.** LLMs cannot plan or self-verify; they help inside verifier loops; intermediate tokens are not "thinking" [11, 70].
- **Arvind Narayanan and Sayash Kapoor.** "Diffusion occurs over decades, not years"; a capability–reliability gap blocks agents [73].
- **François Chollet / ARC Prize.** ARC-AGI-2 (2025): the 2025 Kaggle winner scored 24% at $0.20 per task, the top commercial model 37.6%, a refinement harness 54% at about $30 per task; ARC Prize's verdict, "So, do we have AGI? Not yet", and current reasoning performance is "knowledge-dependent" [84]. ARC-AGI-3, an interactive benchmark of learning in novel game environments, launched in late March 2026; secondary reports put every frontier model under 1% at launch against humans solving all games [111, 112]. A direct 2026 Chollet quote was not verified in this pass. Note: the seed's table files Chollet among the "visionaries"; he is better described as the field's most prominent measurer of what LLMs lack [1].

**Optimists' admissions**

- **Sam Altman** (August 2025), asked whether investors are overexcited about AI: yes; "When bubbles happen, smart people get overexcited about a kernel of truth" [94].
- **Dario Amodei** (22 May 2025): "I suspect that AI models probably hallucinate less than humans, but they hallucinate in more surprising ways" [113].
- **OpenAI researchers, "Why Language Models Hallucinate"** (Kalai et al., September 2025): hallucinations are a predictable result of training and evaluation that reward guessing over abstaining; the fix proposed is to change benchmark scoring [114]. This concedes the problem is real and argues it is incentive-shaped, not architectural.
- **The GPT-5 "plateau" discourse** (August 2025): the launch was widely read as incremental, and Marcus framed it as evidence against pure scaling [107].
- **Andrej Karpathy** (Dwarkesh Patel, 17 October 2025): "this is more accurately described as the decade of agents"; "They're cognitively lacking and it's just not working"; "It will take about a decade to work through all of those issues"; "there's some over-prediction going on in the industry"; of some current output, "It's slop" [20, 115]. Note: the seed attributes to Karpathy a line comparing LLM critics to people "dismissing modern web applications because Ethernet drops packets"; no source for it was found (see section 7).
- **Ilya Sutskever** (Dwarkesh Patel, 25 November 2025): "from 2020 to 2025, it was the age of scaling"; "So it's back to the age of research again, just with big computers"; "These models somehow just generalize dramatically worse than people. It's a very fundamental thing"; "the economic impact seems to be dramatically behind" [19].

Karpathy and Sutskever are the two quotations the essay most needs: they come from the thesis's own "visionary" column and concede both the timeline and a possibly fundamental generalisation gap.

## 7. Seed quotations that could not be verified

The essay will be read by people who know the field; an invented quote discredits the whole argument. Searches found no source for these quotations in the seed, so they should not be used unless traced to a primary text:

- Orfali, Harkey and Edwards (1996): "HTTP is a brain-dead protocol..." (no match found) [1].
- Jakob Nielsen (1996): "The Web is an information space, not a software platform..." (no match found).
- Andrej Karpathy: "...like dismissing modern web applications because Ethernet drops packets" (no match found).

Not checked in this pass, and of the same suspicious form (long, perfectly on-message, no URL): the Fielding (2000) "architectural anomaly ... appalled" passage, the SunWorld (1996) JavaScript quote, the Spolsky "toy language" quote, the DHH "mediocre consultant" quote, the Chollet "software engineering has never been about typing code correctly" quote, and the Abbate quotation about AT&T engineers. Stoll's 1995 Newsweek essay is real and quoted widely [116].

## REFERENCES

1. [thorwhalen/essays issue 30, "AI and the Accretion Trap" (seed)](https://github.com/thorwhalen/essays/issues/30)
2. [Fye SR, Charbonneau SM, Hay JW, Mullins CA. An examination of factors affecting accuracy in technology forecasts. Technol Forecast Soc Change. 2013;80(6):1222–1231](https://doi.org/10.1016/j.techfore.2012.10.026)
3. [Mullins C (The Tauri Group). Retrospective Analysis of Technology Forecasting: In-Scope Extension. DTIC ADA568107; 2012](https://apps.dtic.mil/sti/html/tr/ADA568107/index.html)
4. [The Economist, "Artificial intelligence is losing hype", 19 Aug 2024, as summarised in Wikipedia, "Gartner hype cycle" (secondary)](https://en.wikipedia.org/wiki/Gartner_hype_cycle)
5. [Schnaars SP, Berenson C. Growth market forecasting revisited: a look back at a look forward. California Management Review. 1986;28(4)](https://cmr.berkeley.edu/1986/08/28-4-growth-market-forecasting-revisited-a-look-back-at-a-look-forward)
6. [Musgrave A, Pigden C. Imre Lakatos. Stanford Encyclopedia of Philosophy](https://plato.stanford.edu/entries/lakatos/)
7. [Shojaee P, Mirzadeh I, Alizadeh K, Horton M, Bengio S, Farajtabar M. The Illusion of Thinking. arXiv:2506.06941; 2025](https://arxiv.org/abs/2506.06941)
8. [Lawsen A, Opus C. The Illusion of the Illusion of Thinking. arXiv:2506.09250; 2025](https://arxiv.org/abs/2506.09250)
9. [Jacobson V, Karels MJ. Congestion Avoidance and Control. SIGCOMM 1988 (revised Nov 1988)](https://www.cs.princeton.edu/courses/archive/fall17/cos561/papers/TCP88.pdf)
10. [Newcombe C, Rath T, Zhang F, Munteanu B, Brooker M, Deardeuff M. How Amazon Web Services Uses Formal Methods. Commun ACM. 2015;58(4)](https://cacm.acm.org/magazines/2015/4/184701-how-amazon-web-services-uses-formal-methods/fulltext)
11. [Kambhampati S, Valmeekam K, Guan L, et al. Position: LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks. ICML 2024, PMLR 235](https://proceedings.mlr.press/v235/kambhampati24a.html)
12. [Visual Studio Magazine. TypeScript tops GitHub Octoverse as AI era reshapes language choices. 31 Oct 2025](https://visualstudiomagazine.com/articles/2025/10/31/typescript-tops-github-octoverse-as-ai-era-reshapes-language-choices.aspx)
13. [Becker J, Rush N, Barnes E, Rein D (METR). Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity. arXiv:2507.09089; 2025](https://arxiv.org/abs/2507.09089)
14. [Yao S, Shinn N, Razavi P, Narasimhan K. τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. arXiv:2406.12045; 2024](https://arxiv.org/abs/2406.12045)
15. [ARC Prize. OpenAI o3 Breakthrough High Score on ARC-AGI-Pub. Dec 2024](https://arcprize.org/blog/oai-o3-pub-breakthrough)
16. [Bloomberg. An $800 Billion Revenue Shortfall Threatens AI Future, Bain Says. 23 Sep 2025](https://www.bloomberg.com/news/articles/2025-09-23/an-800-billion-revenue-shortfall-threatens-ai-future-bain-says)
17. [Fortune. Remembering Gopher, the Browsable Internet Before the Rise of the Web. 14 Aug 2016](https://fortune.com/2016/08/14/gopher-internet-history/)
18. [Willison S. The lethal trifecta for AI agents. 16 Jun 2025](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/)
19. [Dwarkesh Patel. Ilya Sutskever: We're moving from the age of scaling to the age of research. 25 Nov 2025](https://www.dwarkesh.com/p/ilya-sutskever-2)
20. [Dwarkesh Patel. Andrej Karpathy: AGI is still a decade away. 17 Oct 2025](https://www.dwarkesh.com/p/andrej-karpathy)
21. [Wikipedia. Roy Amara (Amara's law)](https://en.wikipedia.org/wiki/Roy_Amara)
22. [Carpenter B, Brim S. RFC 3234, Middleboxes: Taxonomy and Issues. 2002](https://www.rfc-editor.org/rfc/rfc3234)
23. [Thomson M, Schinazi D. RFC 9413, Maintaining Robust Protocols. IAB; 2023](https://www.rfc-editor.org/rfc/rfc9413)
24. [Iyengar J, Thomson M. RFC 9000, QUIC: A UDP-Based Multiplexed and Secure Transport. 2021](https://www.rfc-editor.org/rfc/rfc9000)
25. [Gabriel RP. Worse Is Better (the author's chronology of the essays). dreamsongs.com](https://dreamsongs.com/WorseIsBetter.html)
26. [Saltzer JH, Reed DP, Clark DD. End-to-End Arguments in System Design. ACM TOCS. 1984;2(4):277–288](https://web.mit.edu/Saltzer/www/publications/endtoend/endtoend.pdf)
27. [P2P Foundation wiki. David Isenberg (on "The Rise of the Stupid Network", 1997)](https://wiki.p2pfoundation.net/David_Isenberg)
28. [Bower JL, Christensen CM. Disruptive Technologies: Catching the Wave. Harvard Business Review. Jan–Feb 1995](https://hbr.org/1995/01/disruptive-technologies-catching-the-wave)
29. [Lepore J. The Disruption Machine. The New Yorker. 23 Jun 2014](https://www.newyorker.com/magazine/2014/06/23/the-disruption-machine)
30. [King AA, Baatartogtokh B. How Useful Is the Theory of Disruptive Innovation? MIT Sloan Management Review. 2015;57(1):77–90](https://sloanreview.mit.edu/article/how-useful-is-the-theory-of-disruptive-innovation/)
31. [Dixon C. The next big thing will start out looking like a toy. 3 Jan 2010](https://cdixon.org/2010/01/03/the-next-big-thing-will-start-out-looking-like-a-toy/)
32. [Shirky C. The Semantic Web, Syllogism, and Worldview. Nov 2003 (archived)](https://gwern.net/doc/ai/2003-11-07-clayshirky-thesemanticwebsyllogismandworldview.html)
33. [Shirky C. Ontology is Overrated: Categories, Links, and Tags. 2005 (archived)](https://gwern.net/doc/philosophy/ontology/2005-04-shirky-ontologyisoverratedcategorieslinksandtags.html)
34. [Spolsky J. The Law of Leaky Abstractions. 11 Nov 2002](https://www.joelonsoftware.com/2002/11/11/the-law-of-leaky-abstractions/)
35. [Sutton R. The Bitter Lesson. 13 Mar 2019](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)
36. [Aschenbrenner L. Situational Awareness: From GPT-4 to AGI (unhobbling). Jun 2024](https://situational-awareness.ai/from-gpt-4-to-agi/)
37. [Zaharia M, Khattab O, et al. The Shift from Models to Compound AI Systems. BAIR blog. 18 Feb 2024](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/)
38. [InfoQ. OpenAI's harness engineering with Codex (on Lopopolo R, "Harness Engineering: Leveraging Codex in an Agent-First World", 11 Feb 2026)](https://www.infoq.com/news/2026/02/openai-harness-engineering-codex)
39. [Yang E. Thoughts about Spec-ulation (Rich Hickey, Clojure/conj 2016 keynote). Dec 2016](https://blog.ezyang.com/2016/12/thoughts-about-spec-ulation-rich-hickey/)
40. [Boston Globe. Segway to end production of its original personal transporter. 24 Jun 2020](https://www.bostonglobe.com/business/2020/06/24/segway-end-production-its-original-personal-transporter/bFtitEmqVpn8s5IvcZLjHN/story.html)
41. [Fierce/StreamTV Insider. 3D TV officially dead as Sony and LG stop making sets. Jan 2017](https://www.streamtvinsider.com/cable/3d-tv-officially-dead-as-sony-and-lg-stop-making-sets)
42. [The Register. 95% of NFTs now totally worthless, say researchers. 21 Sep 2023](https://www.theregister.com/2023/09/21/95_percent_nfts_worthless/)
43. [Outlook Respawn. Meta retains optimism in VR after Reality Labs loses $83.6 billion. 2026](https://respawn.outlookindia.com/pop-culture/pop-culture-news/meta-retains-optimism-in-vr-after-reality-labs-loses-836-billion)
44. [Gilbert ML. Review: Read Write Own by Chris Dixon. Citation Needed. 2024](https://www.citationneeded.news/review-read-write-own-by-chris-dixon/)
45. [Lefkowitz J. An Apologia for Push. Aug 2003](https://jasonlefkowitz.net/2003/08/an_apologia_for/)
46. [9to5Google. Google discontinues Glass Enterprise Edition. 15 Mar 2023](https://9to5google.com/2023/03/15/google-glass-enterprise-edition-discontinued/)
47. [UploadVR. Meta and EssilorLuxottica sold 7 million smart glasses in 2025. Feb 2026](https://uploadvr.com/meta-essilorluxottica-sold-7-million-smart-glasses-in-2025/)
48. [The Drive. Tesla still planning coast-to-coast autonomous drive. Feb 2018](https://www.thedrive.com/tech/18337/tesla-still-planning-coast-to-coast-autonomous-drive)
49. [Claims Journal. Waymo co-CEO outlines path to 1 million weekly trips in 2026. 12 Feb 2026](https://amp.claimsjournal.com/news/national/2026/02/12/335668.htm)
50. [Singhal A. Introducing the Knowledge Graph: things, not strings. Google blog. 16 May 2012](https://blog.google/products/search/introducing-knowledge-graph-things-not/)
51. [Internetnews.com. I-mode whips WAP; analysts say they can co-exist. 2000](https://internetnews.com/it-management/i-mode-whips-wap-analysts-say-they-can-co-exist)
52. [OpenJDK. JEP 289: Deprecate the Applet API; JEP 398: Deprecate the Applet API for Removal](https://openjdk.org/jeps/398)
53. [Wikipedia. VRML](https://en.wikipedia.org/wiki/VRML)
54. [Wikipedia. Second Life](https://en.wikipedia.org/wiki/Second_Life)
55. [Albright RE. What can past technology forecasts tell us about the future? Technol Forecast Soc Change. 2002;69(5):443–464](https://www.nokia.com/bell-labs/publications-and-media/publications/what-can-past-technology-forecasts-tell-us-about-the-future)
56. [Open Philanthropy. Evaluation of some technology forecasts from The Year 2000](https://www.openphilanthropy.org/evaluation-some-technology-forecasts-year-2000)
57. [Kott A, Perconti P. Long-term forecasts of military technologies for a 20–30 year horizon: an empirical assessment of accuracy. Technol Forecast Soc Change. 2018;137:272–279](https://arxiv.org/abs/1807.08339)
58. [Tetlock PE. Expert Political Judgment: How Good Is It? How Can We Know? Princeton University Press; 2005](https://press.princeton.edu/books/paperback/9780691178288/expert-political-judgment)
59. [Cato Unbound. Issue on Tetlock's expert political judgment (discussion of the chimp comparison)](https://www.cato-unbound.org/print-issue/323)
60. [Armstrong JS. The Seer-Sucker Theory: The Value of Experts in Forecasting. Technology Review. 1980 Jun/Jul:16–24](https://repository.upenn.edu/entities/publication/b1866aff-767b-432b-804d-cfb7d140e197/full)
61. [Christian Science Monitor. Flawed Forecasts (review of Schnaars, Megamistakes). 10 Mar 1989](https://www.csmonitor.com/1989/0310/dbmega.html)
62. [Mullany M. 8 Lessons from 20 Years of Hype Cycles. LinkedIn, 7 Dec 2016 (figures via secondary coverage)](https://www.linkedin.com/pulse/8-lessons-from-20-years-hype-cycles-michael-mullany)
63. [Dedehayir O, Steinert M. The hype cycle model: A review and future directions. Technol Forecast Soc Change. 2016;108:28–41](https://ideas.repec.org/a/eee/tefoso/v108y2016icp28-41.html)
64. [Armstrong S, Sotala K. How We're Predicting AI, or Failing To. 2012](https://www.openphilanthropy.org/wp-content/uploads/ArmstrongandSotala2012.pdf)
65. [Grace K. Error in Armstrong and Sotala 2012. AI Impacts](https://aiimpacts.org/error-in-armstrong-and-sotala-2012/)
66. [Wikipedia. AI winter](https://en.wikipedia.org/wiki/AI_winter)
67. [The Copernican Shift: Why the Future of RAG is Agentic LLMs. AI in Plain English](https://ai.plainenglish.io/the-copernican-shift-why-the-future-of-rag-is-agentic-llms-95669c12b13e)
68. [Position: Stop Acting Like Language Model Agents Are Normal Agents. arXiv:2502.10420; 2025](https://arxiv.org/abs/2502.10420)
69. [Marcus G. A knockout blow for LLMs? Marcus on AI. 7 Jun 2025](https://garymarcus.substack.com/p/a-knockout-blow-for-llms)
70. [Kambhampati S, et al. Stop Anthropomorphizing Intermediate Tokens as Reasoning/Thinking Traces! arXiv:2504.09762; 2025](https://arxiv.org/abs/2504.09762)
71. [Mitchell M. Artificial intelligence learns to reason. Science. 2025;387(6740)](https://www.science.org/doi/10.1126/science.adw5211)
72. [Mitchell M. AI: A Guide for Thinking Humans (Substack archive, incl. "Do AI Reasoning Models Abstract and Reason Like Humans?", Oct 2025)](https://aiguide.substack.com/archive)
73. [Narayanan A, Kapoor S. AI as Normal Technology. Knight First Amendment Institute. 15 Apr 2025](https://knightcolumbia.org/content/ai-as-normal-technology)
74. [Princeton SPIA. Excerpt: AI Snake Oil (Narayanan A, Kapoor S, Princeton University Press, 2024)](https://spia.princeton.edu/news/excerpt-princeton-spia-ai-experts-separate-hype-substance-new-book)
75. [Gilbert S, Lynch N. Brewer's conjecture and the feasibility of consistent, available, partition-tolerant web services. ACM SIGACT News. 2002;33(2)](https://dl.acm.org/doi/10.1145/564585.564601)
76. [Brewer E. CAP Twelve Years Later: How the "Rules" Have Changed. InfoQ/IEEE Computer. 2012](https://www.infoq.com/articles/cap-twelve-years-later-how-the-rules-have-changed/)
77. [Epoch AI. How persistent is the inference cost burden? (discussion of Toby Ord's scaling series)](https://epoch.ai/gradient-updates/how-persistent-is-the-inference-cost-burden)
78. [Bowley R. METR's developer productivity research: 2026 update (secondary account of METR's Feb–Mar 2026 post). 4 Apr 2026](https://blog.robbowley.net/2026/04/04/metrs-developer-productivity-research-2026-update/)
79. [Sculley D, Holt G, Golovin D, et al. Hidden Technical Debt in Machine Learning Systems. NeurIPS 2015](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems)
80. [Wikipedia. No Silver Bullet (Brooks FP, 1986)](https://en.wikipedia.org/wiki/No_Silver_Bullet)
81. [Sierra. Benchmarking AI agents (τ-bench launch post)](https://sierra.ai/blog/benchmarking-ai-agents)
82. [Kwa T, West B, Becker J, et al. Measuring AI Ability to Complete Long Tasks. arXiv:2503.14499; 2025](https://arxiv.org/abs/2503.14499)
83. [Bettencourt N. Are Auto-Regressive Large Language Models Here to Stay? (on LeCun's March 2023 slides; secondary)](https://medium.com/@nbettencourt2020/are-auto-regressive-large-language-models-here-to-stay-603baee46a3c)
84. [Knoop M. ARC Prize 2025 Results and Analysis. 5 Dec 2025](https://arcprize.org/blog/arc-prize-2025-results-analysis)
85. [International Energy Agency. Energy and AI. Apr 2025](https://www.iea.org/reports/energy-and-ai)
86. [Hargreaves Lansdown (republishing Business Insider). AI hyperscalers are expected to finish the year spending $828 billion on capex. 2026](https://www.hl.co.uk/news/ai-hyperscalers-are-expected-to-finish-the-year-spending-a-whopping-828-billion-on-capex)
87. [Cahn D. AI's $600B Question. Sequoia Capital. 2024](https://www.sequoiacap.com/article/ais-600b-question/)
88. [Challapally A, Pease C, Raskar R, Chari P. The GenAI Divide: State of AI in Business 2025. MIT NANDA (hosted copy)](https://webberwentzel.com/News/Documents/2025/MIT1757411281972.pdf)
89. [CustomerThink. The Great GenAI Divide: Debunking the Myth of 95% Failure. 2025](https://customerthink.com/the-great-genai-divide-debunking-the-myth-of-95-failure/)
90. [Sukhareva M. GenAI has little to no return on investment? AI Realist. 2025](https://msukhareva.substack.com/p/genai-has-little-to-no-return-on)
91. [Zitron E. The Hater's Guide to the AI Bubble, Vol. 2. Where's Your Ed At](https://www.wheresyoured.at/premium-the-haters-guide-to-the-ai-bubble-vol-2/)
92. [The New Stack. How solid is Ed Zitron's case against generative AI?](https://thenewstack.io/how-solid-is-ed-zitrons-case-against-generative-ai/)
93. [The Conversation. Who's paying whom? The circular money machine behind the AI boom (or bubble). Oct 2026](https://theconversation.com/whos-paying-whom-the-circular-money-machine-behind-the-ai-boom-or-bubble-293938)
94. [Entrepreneur. OpenAI CEO Sam Altman thinks we're in an AI bubble. 19 Aug 2025](https://www.entrepreneur.com/business-news/openai-ceo-warns-that-investors-are-overexcited-about-ai/496081)
95. [Linux Foundation. Linux Foundation announces the formation of the Agentic AI Foundation. 9 Dec 2025](https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation)
96. [Agent Patterns. Temporary compensatory mechanisms](https://agentpatterns.ai/patterns/agent-design/temporary-compensatory-mechanisms/)
97. [Corbett JC, Dean J, et al. Spanner: Google's Globally-Distributed Database. OSDI 2012](https://www.usenix.org/conference/osdi12/technical-sessions/presentation/corbett)
98. [High Scalability. Google Spanner's most surprising revelation: NoSQL is out and NewSQL is in](https://highscalability.com/google-spanners-most-surprising-revelation-nosql-is-out-and/)
99. [Rosen E, Viswanathan A, Callon R. RFC 3031, Multiprotocol Label Switching Architecture. 2001](https://www.rfc-editor.org/rfc/rfc3031)
100. [Wikipedia. gRPC](https://en.wikipedia.org/wiki/GRPC)
101. [Wikipedia. GraphQL](https://en.wikipedia.org/wiki/GraphQL)
102. [Gabriel RP. The Rise of "Worse is Better"](https://dreamsongs.com/RiseOfWorseIsBetter.html)
103. [Digitopoly. Disrupting the theory of disruption. 21 Sep 2015](https://digitopoly.org/2015/09/21/disrupting-the-theory-of-disruption/)
104. [Graham P. Beating the Averages. 2001](https://paulgraham.com/avg.html)
105. [Wikipedia. Wardley map](https://en.wikipedia.org/wiki/Wardley_map)
106. [Evans B. AI Eats the World (Autumn 2025 presentation, hosted copy)](https://businessofsoftware.org/wp-content/uploads/sites/11/2025/11/AI-Eats-the-World-Benedict-Evans-Autumn-2025.pdf)
107. [Marcus G. GPT-5: Overdue, overhyped and underwhelming. 9 Aug 2025](https://garymarcus.substack.com/p/gpt-5-overdue-overhyped-and-underwhelming)
108. [LeCun Y. A Path Towards Autonomous Machine Intelligence. OpenReview; 2022](https://openreview.net/pdf?id=BZ5a1r-kVsf)
109. [SiliconANGLE. Yann LeCun's new startup AMI Labs raises $1.03B to train world models. 10 Mar 2026](https://siliconangle.com/2026/03/10/yann-lecuns-new-startup-ami-labs-raises-1-03b-train-world-models/)
110. [Bender EM, Hanna A. The AI Con (book site)](https://thecon.ai/)
111. [ARC Prize. ARC-AGI-3](https://arcprize.org/arc-agi/3)
112. [AI News. ARC-AGI-3 shows AI models fail at general reasoning. 2026 (secondary)](https://www.ainews.com/p/arc-agi-3-shows-ai-models-fail-at-general-reasoning)
113. [Zeff M. Anthropic CEO claims AI models hallucinate less than humans. TechCrunch. 22 May 2025](https://techcrunch.com/2025/05/22/anthropic-ceo-claims-ai-models-hallucinate-less-than-humans/)
114. [Kalai AT, Nachum O, Vempala SS, Zhang E. Why Language Models Hallucinate. arXiv:2509.04664; 2025](https://arxiv.org/abs/2509.04664)
115. [Willison S. Andrej Karpathy: AGI is still a decade away (transcript excerpts). 18 Oct 2025](https://simonwillison.net/2025/Oct/18/agi-is-still-a-decade-away/)
116. [Stoll C. Why the Web Won't Be Nirvana. Newsweek. 26 Feb 1995](https://www.newsweek.com/clifford-stoll-why-web-wont-be-nirvana-185306)
