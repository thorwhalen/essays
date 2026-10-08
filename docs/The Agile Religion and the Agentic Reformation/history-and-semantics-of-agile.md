# History and semantics of "agile": research notes for The Agile Religion and the Agentic Reformation

## Summary for the blog draft (read this first)

1. Royce's 1970 paper is more nuanced than the draft's footnote. He calls the Figure 2 sequence "risky and invites failure" [1, p. 329], and he does recommend a pilot ("Do it twice"), and customer involvement, but he also demands "quite a lot" of documentation and "ruthless enforcement of documentation requirements" [1, p. 332], and puts program design first. The paper is not an agile ancestor; Larman and Basili say it is "clearly not classic IID" [2]. Safer wording: Royce shows the single-pass diagram and says it invites failure, then prescribes five heavier-weight fixes. His own son says he "was always a proponent of iterative, incremental, evolutionary development" [2].
2. "Waterfall" was not christened by critics. Royce never uses the word (the OCR text of his paper contains zero occurrences). The earliest use I could verify is Bell and Thayer (TRW, ICSE 1976): "he [Royce] introduced the concept of the 'waterfall' of development activities" [3], said neutrally, as praise of "an excellent paper by Royce". It became a pejorative later. Fix the draft's "christened by its critics".
3. The straw man became a standard through the US DoD: DoD-STD-2167 (1985) required a document-driven, single-pass waterfall; its principal creator later regretted it; 2167A (Feb 1988) tried to be life-cycle neutral but was read as waterfall; MIL-STD-498 (Dec 1994) replaced it and removed the "waterfall bias" [2][4]. This is a better story than "nobody read past the first diagram": the standard-writers had not heard of iterative development.
4. Brooks recanted "plan to throw one away" in 1995: "The biggest mistake in the 'Build one to throw away' concept is that it implicitly assumes the classical sequential or waterfall model" [5][6] (second-hand quote; see section 3). Royce and Brooks both advocated a throwaway first version, and both are about to be recast by agents making a second version cheap.
5. The Manifesto quotes in the draft are correct [7], but the draft's "we kept the right-hand items in the fine print" fits the record: Highsmith wrote "We embrace documentation, but not hundreds of pages of never-maintained and rarely-used tomes" [8], and principle 9 says "Continuous attention to technical excellence and good design enhances agility" [9]. Fowler's own Yagni says it "does not apply to effort to make the software easier to modify" [10], which backs the draft's "seams" argument.
6. "Every developer agrees with agile" needs nuance. The Microsoft longitudinal study (1,969 respondents, 2006-2012) found developers using agile like it, yet agile adoption grew slower than expected and people moved to non-agile teams willingly [11]. The 17th State of Agile report (788 respondents, 2024) reports 71% using agile but only 11% "very satisfied" (press summary) [12]. Jeffries titled a 2018 article "Developers Should Abandon Agile" [13]. Frame the claim as personal experience, or as agreement with the word and disagreement with the practice, which the sources support.
7. The Engprax "268%" study is a commercial 600-respondent survey with a contested definition of agile; do not use it as evidence [14][15].
8. The religion metaphor is not novel, and the Reformation metaphor for agile is already in print: Acceptance Criteria, "The Agile Reformation" (6 March 2024) [16]; a DZone piece "Agile Manifesto: The Reformation That Became the Church" (undated in my fetch) [17]; a Scrum Expert piece on certifications as indulgences (12 May 2026) [18]; Atwood, "Software Development: It's a Religion" (2006) [19]. The printing-press metaphor for AI coding is also in print (Fuller, Dec 2025 [20]; Yavuz, Feb 2026 [21]). I found no piece that combines agile, Reformation, and AI agents under the title "Agentic Reformation", but my search was limited (a few queries), so say "I have not seen" rather than "nobody has".

## 1. Royce 1970

Citation: W. W. Royce, "Managing the Development of Large Software Systems", Proc. IEEE WESCON, August 1970, pp. 1-9; reprinted in Proc. 9th International Conference on Software Engineering (ICSE), 1987, pp. 328-338 [1]. The page numbers below are those of the 1987 reprint, read off the page footers of an OCR scan; I could not check them against the IEEE original. The copy I read is the scan reprinted with the notice "Copyright 1970 by The Institute of Electrical and Electronics Engineers, Inc. Originally published by TRW" [1].

Stable URLs: there is no open official copy I could verify. The ACM Digital Library entry is [https://dl.acm.org/doi/10.5555/41765.41801](https://dl.acm.org/doi/10.5555/41765.41801) (cited by Wikipedia; paywalled/blocked to my fetcher). The University of Maryland course copy ([https://www.cs.umd.edu/class/spring2003/cmsc838p/Process/waterfall.pdf](https://www.cs.umd.edu/class/spring2003/cmsc838p/Process/waterfall.pdf)) and the Praxis copy return HTTP 403 to scripted fetches but are the usual links; I read the Internet Archive snapshot of the UMD copy [1]. Link both the ACM entry and the Wayback snapshot in the published post.

What it says (all quotes from [1]; OCR errors silently corrected where the meaning is certain):

- Opening (p. 328): the two-step analysis-then-coding approach (Figure 1) suffices only "if the effort is sufficiently small and if the final product is to be operated by those who built it"; for larger systems it "is doomed to failure". Customers "would rather not pay" for the additional steps, and "the prime function of management is to sell these concepts to both groups and then enforce compliance".
- Figure 2 (p. 328) is the sequential diagram: system requirements, software requirements, analysis, program design, coding, testing, operations. Figure 3 (p. 330) is captioned "Hopefully, the iterative interaction between the various phases is confined to successive steps" and Figure 4 "Unfortunately, for the process illustrated, the design iterations are never confined to the successive steps."
- The "straw man" sentence (p. 329): "I believe in this concept, but the implementation described above is risky and invites failure." He explains that testing is "the first event for which timing, storage, input/output transfers, etc., are experienced as distinguished from analyzed", and then "one can expect up to a 100-percent overrun in schedule and/or costs". Next paragraph: "However, I believe the illustrated approach to be fundamentally sound. The remainder of this discussion presents five additional features that must be added to this basic approach to eliminate most of the development risks."
- Iteration (p. 329): "as each step progresses and the design is further detailed, there is an iteration with the preceding and succeeding steps but rarely with the more remote steps in the sequence." Royce expects iteration between adjacent phases, which is a limited kind of feedback.
- Step 1, "Program design comes first" (p. 331): the designer must design "even at the risk of being wrong". This is upfront design.
- Step 2, "Document the design" (p. 332): "how much documentation? My own view is 'quite a lot;' ... The first rule of managing software development is ruthless enforcement of documentation requirements." Also: "During the early phase of software development the documentation is the specification and is the design." He estimates a 30-page specification for a $5M hardware procurement against a much longer one for $5M of software (the number is illegible in the OCR; secondary sources say 1,500 pages, which I could not verify). Direct support for the blog's thesis that documents were once treated as the design.
- Step 3, "Do it twice" (p. 334): "arrange matters so that the version finally delivered to the customer for operational deployment is actually the second version insofar as critical design/operations areas are concerned." Figure 7 caption: "Attempt to do the job twice - the first result provides an early simulation of the final product." A 30-month effort gets a pilot of about 10 months.
- Step 4, "Plan, control and monitor testing" (p. 335): testing is "the phase of greatest risk in terms of dollars and schedule".
- Step 5, "Involve the customer" (p. 335): "what a software design is going to do is subject to wide interpretation even after previous agreement. It is important to involve the customer in a formal way so that he has committed himself at earlier points before final delivery." Figure 9 marks three customer review points; this is formal sign-off, not Manifesto-style collaboration.
- Summary (p. 335): "I would emphasize that each item costs some additional sum of money ... the simpler method has never worked on large software development efforts". Figure 10 (last figure, p. 338 in the OCR) summarises the five steps.

Answers to the three questions: iteration, yes but limited (adjacent phases plus one pilot); "do it twice", yes, explicitly; customer involvement, yes, but as formal commitment at review gates. Documentation: strongly pro, which the draft's footnote omits.

Secondary confirmation: Larman and Basili: "Many-incorrectly-view Royce's paper as the paragon of single-pass waterfall", and "Royce's recommendation was to do it twice" [2, p. 3]. Walker Royce (his son) to Larman and Basili: "He was always a proponent of iterative, incremental, evolutionary development. His paper described the waterfall as the simplest description, but that it would not work for all but the most straightforward projects." [2]. Larman and Basili add that this was "clearly not classic IID" [2]. Hogarth's close reading agrees and also stresses the documentation passages [22].

Figures worth reproducing: Figure 2 (sequential diagram), Figure 3 vs Figure 4 (hoped-for vs actual iteration, a good visual for "the straw man the paper warns about"), Figure 7 (do it twice), Figure 10 (summary of five steps). Licence: the reprint carries a 1970 IEEE copyright notice (TRW as originator) [1], so do not embed the scans. Redraw the boxes yourself (simple geometry, no creative text), credit Royce, and link to the paper. I could not verify any open licence.

## 2. Who coined "waterfall", and how it became the standard

- The word does not appear in Royce's text [1]. First verified use: T. E. Bell and T. A. Thayer, "Software requirements: Are they really a problem?", TRW, Proc. 2nd International Conference on Software Engineering, 1976 [3]: "Royce [5]; he introduced the concept of the 'waterfall' of development activities. In this approach software is developed in the disciplined sequence of activities shown in Figure 1." The Internet Archive catalog lists the year as 1974 and the author as "Thaye", which looks like cataloguing error; the paper cites TRW reports from April 1976, so 1976 is plausible [3]. Wikipedia likewise gives 1976 and notes the term "may" first appear there [23]. I did not search earlier literature exhaustively, so say "earliest use I know of".
- Earlier phased models: Benington's 1956 SAGE paper (republished 1983) is cited by Wikipedia as the first presentation of the staged approach [23].
- DoD-STD-2167: dated 4 June 1985, mandated a document-driven phase structure (Wikipedia, secondary) [23]. Larman and Basili: "By the late 1980s, the DoD was experiencing significant failure in acquiring software based on the strict, document-driven, single-pass waterfall model that DoD-Std-2167 required" and cite a 1999 review: "Of a total $37 billion for the sample set, 75% of the projects failed or were never used" [2]. The 1987 Defense Science Board Task Force on Military Software, chaired by Brooks, said of 2167: "it continues to reinforce exactly the document-driven, specify-then-build approach that lies at the heart of so many DoD software problems" [2].
- DoD-STD-2167A (February 1988) was intended to be life-cycle neutral ("This standard is not intended to specify or discourage the use of any particular software development method") but "many (justifiably) interpreted the new standard as containing an implied preference for the waterfall model" [2]. The principal creator of 2167 later expressed regret: "he had not heard of iterative development" [2] (Larman and Basili's paraphrase of a personal conversation).
- MIL-STD-498 replaced 2167A in December 1994, with a section titled "Removing the Waterfall Bias" in Newberry's summary [2]; it describes development "in one or more incremental builds" [2].
- Larman and Basili on iterative development before agile [2]: Project Mercury (late 1950s) used half-day iterations; X-15; Randell and Zurcher's 1968 IBM report is the earliest reference "specifically focused on describing and recommending iterative development"; Lehman's 1969 report; Harlan Mills's 1970s work at IBM FSD; Tom Gilb's Evo (1976, 1988); Trident submarine (1977). One IBM veteran quoted: "All of us ... thought waterfalling of a huge project was rather stupid" [2]. Scrum is traced partly to Takeuchi and Nonaka (1986) [2].

This supports a stronger framing than the draft's: the sequential model was a government-contracting artefact, spread by a procurement standard, while practitioners were already iterating.

## 3. Brooks: "plan to throw one away"

- Original (1975, ch. 11): "Hence, plan to throw one away: You will, anyhow." and "The management question here, therefore, is not whether to build a pilot system and throw it away" (quoted from a chapter summary blog, second-hand) [24]. I could not read the book itself.
- Revision (1995 Anniversary Edition, ch. 19 "The Mythical Man-Month after 20 Years"): the section is headed "Don't Build One to Throw Away - The Waterfall Model Is Wrong!", followed by "An Incremental-Build Model Is Better - Progressive Refinement" (table of contents, publisher sample) [6]. The sentence "The biggest mistake in the 'Build one to throw away' concept is that it implicitly assumes the classical sequential or waterfall model of software construction" is quoted by a blog and a search result [5]; I could not check it against the book, so quote it as "as quoted in [5]" or verify against the print edition (approximately p. 265, per a search summary, unverified).
- Relevance: Royce (1970) and Brooks (1975) both proposed building a first version to be discarded; both hedge against the cost of the throwaway; and Brooks later saw that incremental building made the throwaway unnecessary. If agents make a second version cheap again, the blog can argue that "build one to throw away" returns as a cheap option. That is my inference, not a sourced claim.

## 4. The Agile Manifesto (2001)

Exact text from [7] (page notice: "this declaration may be freely copied in any form, but only in its entirety through this notice", so reproduce it whole or only quote the lines):

> We are uncovering better ways of developing software by doing it and helping others do it. Through this work we have come to value:
> Individuals and interactions over processes and tools
> Working software over comprehensive documentation
> Customer collaboration over contract negotiation
> Responding to change over following a plan
> That is, while there is value in the items on the right, we value the items on the left more.

(I confirmed the four values and the final line by fetch; the lead-in "Through this work we have come to value:" is from memory of the page and not shown in my fetch output, so check before quoting.)

Principles on design and documentation [9]: 9, "Continuous attention to technical excellence and good design enhances agility."; 10, "Simplicity--the art of maximizing the amount of work not done--is essential."; 11, "The best architectures, requirements, and designs emerge from self-organizing teams."; 6, "The most efficient and effective method of conveying information to and within a development team is face-to-face conversation." No principle mentions documentation by name; the documentation stance is only in value 2 and in the history page.

Highsmith's history page [8]: seventeen people met 11-13 February 2001 at The Lodge at Snowbird, Utah; the stated shared aim was an alternative to "documentation driven, heavyweight software development processes"; "We embrace documentation, but not hundreds of pages of never-maintained and rarely-used tomes"; "We plan, but recognize the limits of planning in a turbulent environment." The page records that "Light" was disliked but stuck for the time being, and that Fowler's only recorded concern about "agile" was pronunciation.

Fowler, "Is Design Dead?" (posted July 2000, revised May 2004) [25]: contrasts planned and evolutionary design; "it seems that XP calls for the death of software design", which he disputes; design is not dead but changes nature; evolutionary design depends on "enabling practices" (testing, continuous integration, refactoring); on diagrams: "The primary value is communication", "Only use diagrams that you can keep up to date without noticeable pain", "The best UML diagrams are not artifacts". His "The New Methodology" [26] (July 2000, revised December 2005) says "The most frequent criticism of these methodologies is that they are bureaucratic" and that "The term 'agile' got hijacked for this activity in early 2001". Yagni (2015) [10]: it "does not apply to effort to make the software easier to modify".

## 5. Why "agile" means different things to everyone

- Fowler, "Semantic Diffusion" (14 December 2006) [27]: defined as a word "coined by a person or group, often with a pretty good definition" whose meaning then spreads and blurs; he notes "I've run into people who think agile methods mean you shouldn't do any planning" and "'Agile' sounds like something you'd certainly want to be".
- Dave Thomas, "Agile Is Dead (Long Live Agility)" (4 March 2014) [28]: "The word 'agile' has been subverted to the point where it is effectively meaningless, and what passes for an agile community seems to be largely an arena for consultants and vendors"; "Agile is not a noun, it's an adjective"; "forming an industry group around the four values always struck me as creating a trade union for people who breathe."
- Ron Jeffries, "Developers Should Abandon Agile" (10 May 2018) [13]: the thesis in the title; he uses "Faux Agile" and defines "Dark Agile" as "so-called 'Agile' approaches that have really gone bad". "Dark Scrum" (8 September 2016) [29]: "Dark Scrum begins when people who know their old job, but not their new Scrum job, begin to do the Scrum activities", "Dark Scrum oppresses the team every day."
- Fowler, keynote at Agile Australia, 25 August 2018, "The State of Agile Software in 2018" [30]: "The first one of these is what I would call the Agile Industrial Complex"; "The Agile Industrial Complex imposing methods on people is an absolute travesty"; faux-agile is "agile that's just the name, but none of the practices and values in place"; the second problem is "the lack of recognition of the importance of technical excellence". Related: Fowler's "Agile Imposition" bliki entry argues an agile process should not be imposed from outside [31].
- Forrester, "Water-Scrum-Fall Is The Reality Of Agile For Most Organizations Today" (Dave West, 26 July 2011; paywalled) [32]; secondary summaries: "water" is upfront planning, "Scrum" the team's iterations, "fall" a controlled, infrequent release [33]. I could not read the report itself.
- "No True Scotsman": Gregorio (2008) applied it to Scrum Alliance statements that blamed failures on people rather than process [34]; a Hackernoon essay also invokes it for "agile fundamentalists" [35]; both are opinion pieces. Jeffries's "Dark Scrum" and Fowler's "faux-agile" are arguably the same move made by insiders; that characterisation is mine.

Survey and sentiment data (treat as snapshots, not as a consensus measure):

- Digital.ai, 17th State of Agile Report (January 2024; 788 respondents): 71% use agile in the software lifecycle; 11% of agile users "very satisfied" and 33% "somewhat satisfied"; small organisations 52% "works very or somewhat well" against 43% for medium and large; figures from press coverage, not from the report PDF [12][36]. Respondents are self-selected practitioners of agile, so this is not a developer-wide sample.
- Murphy, Bird, Zimmermann, Williams, Nagappan, Begel, "Have Agile Techniques been the Silver Bullet for Software Development at Microsoft?", ESEM 2013: 1,969 agile and non-agile practitioners in five surveys over 2006-2012; adoption "slower than would be expected", "no clear trends in practice adoption"; those who use agile report liking it, and some respondents who had used agile moved to non-agile teams and did not try to convert their new team [11]. Both camps agreed on relative benefits and problems; non-agile practitioners were "less enamored of the benefits" [11]. The paper shows managers somewhat more positive than developers on average (project managers 78.3% average positive response to agile practices against 75.2% for developers) [11].
- Stack Overflow 2017 survey (secondary summaries only): methodology use of Agile 76.9%, Scrum 65.2%, Kanban 34.8%, Waterfall 26.9% [37]. I did not find methodology questions in later Stack Overflow surveys; unverified.
- Engprax / Dr Junade Ali, *Impact Engineering* survey (May 2024): 600 software engineers (250 UK, 350 US), fieldwork 3-7 May 2024; headline "268% higher failure rate for agile projects" [14]. Criticism: Bloor Research calls the number "ludicrously high" and says the study relies on a strawman of agile [15]; Jon Kern, a Manifesto co-author, called it trash (as reported by The Register) [38]; the study counted projects as "Agile requirements engineering" when development started before clear requirements; it is a vendor-commissioned self-report survey, not peer-reviewed, and Engprax's rebuttal is self-defence, not independent verification [14][15]. A related arXiv paper by the same author exists (arXiv 2410.20696) [39], which I did not read.
- Practitioner voices against: Hacker News collections of negative comments on Scrum [40] are anecdotal and cannot be turned into a percentage.

Conclusion for the draft: there is no good survey that shows "all developers agree with agile". What the evidence supports is that the word is nearly universally endorsed (71% claim to use it) while satisfaction is lukewarm and the practice varies widely, which is the draft's real point.

## 6. Terms for the anti-agile pole, with origins

Verified origins are marked; the rest are "not verified".

- Waterfall: Royce's diagram (1970) [1]; word from Bell and Thayer (1976) [3]; standardised by DoD-STD-2167 (1985) [2][23].
- Big Design Up Front (BDUF): critics' label; defined on the c2 wiki without naming originator or date [41]; I found no verified coiner. Microsoft archive blog posts date the practice's naming to the mid-1990s (recollection, unverified) [42]. Fowler says much design activity "is ridiculed as 'Big Up Front Design'" [25]. Joel Spolsky used it self-descriptively in 2005 (second-hand) [43].
- Big Requirements Up Front (BRUF): synonym for BDUF; popularised as a negative term by Scott Ambler's Agile Modeling essay "Examining the Big Requirements Up Front Approach" [44]; coiner not verified.
- Plan-driven: Boehm's term in the agile debate: Boehm, "Get Ready for Agile Methods, with Care", IEEE Computer 35(1), 64-69, January 2002; Boehm and Turner, *Balancing Agility and Discipline* (Addison-Wesley, 2003/2004), with "home grounds" where each approach dominates [45]. I read only secondary descriptions.
- Heavyweight / lightweight: used by Fowler in "The New Methodology" (2000) ("heavyweight method"); the "lightweight" label was disliked by participants and "agile" chosen at Snowbird in 2001 [8][26].
- Document-driven: used by the Manifesto history page ("documentation driven, heavyweight software development processes") [8] and by the Defense Science Board report ("document-driven, specify-then-build") [2].
- Phase-gate / stage-gate: Robert G. Cooper; "Stage-Gate" first in print 1988, with the 1990 Business Horizons article "Stage-Gate Systems: A New Tool for Managing New Products"; originally called phase-gate; the claimed NASA origin rests on one thesis abstract and is unverified [46]. This is a product-development tool that spread to software, not a software term.
- V-Model: emerged independently in Germany and the US in the late 1980s; NASA bent the waterfall into a "V" in 1988; Forsberg and Mooz presented the Vee model at INCOSE in October 1991 [47]. The German V-Modell 97 was the federal standard [47]. Dates for Rook (1986) and a 1992 German release: not verified.
- Analysis paralysis: attested from 1970 (Silver and Hecker, in a non-software context); popularised in software by *AntiPatterns* (Brown et al., 1998), where it is listed as a waterfall-related antipattern [48]. I did not verify the book's text.
- Taylorism / scientific management: Fowler's "The New Methodology" traces "people as resources" to "Frederick Taylor's Scientific Management approach" and argues it suits factory work, not software [26].
- RUP: Rational Unified Process 5.0 in June 1998, led by Kruchten, descended from Objectory (bought 1995); Fowler calls it "a process framework rather than a process" [26][49]. Note RUP is itself iterative; it is "heavyweight" by reputation, not a waterfall.
- CMM / CMMI: SEI's CMM v1.0 in August 1991, v1.1 in 1993; CMMI v1.1 documents date from December 2001 to August 2002 [50]. The search did not support a 2000 CMMI release.
- Cost-of-change curve: Boehm's "100x" curve is criticised by Bossavit in *The Leprechauns of Software Engineering*, who says the evidence "just isn't up to any reasonable standard of 'research'" and that one cited study found a 2:1 ratio in the opposite direction [51]. Soften the draft's ⚠️ line accordingly: the "100x" figure is poorly supported, as the draft guessed.

## 7. Religion metaphors already in print (do not claim novelty)

- Atwood, "Software Development: It's a Religion", Coding Horror, 9 October 2006: "software development is, and has always been, a religion"; "Isn't Agile the Church of Lightweight?"; "the only truly dangerous people are the religious nuts who don't realize they are religious nuts" [19].
- InfoQ news (c. 2006) on "Agile religion" and Jim Webber's "Agile atheist" (from a search summary, not read directly) [52].
- Cargo cult: Holub's "The Death of Agile" talk (transcribed) [53]; Medium and Hackernoon posts on "Scrum cargo cult" [35][54]; Fowler's cargo-cult remark is cited second-hand and I did not find the original.
- Reformation metaphor for agile: K. Thomas Ulland, "The Agile Reformation: a critical look at the manifesto", Acceptance Criteria, 6 March 2024: "I'm sure some will think using comparisons to The Protestant Reformation is problematic ... But, as a metaphor, I think it's quite apt."; compares the Manifesto authors to Luther's 95 Theses and himself to someone who agrees with "(most of) the underlying principles" [16]. DZone, "Agile Manifesto: The Reformation That Became the Church" argues a Luther-like trajectory from simplicity to scaling frameworks, certification levels and consultancies (fetch blocked; summary from a search snippet) [17]. Scrum Expert (12 May 2026) compares certifications to indulgences: "Some began to sell the 'certifications' of being good Agilists" and "this also led to a schism and the creation of different certifications" [18].
- Printing-press metaphor for AI coding: Boden Fuller, "The Gutenberg Moment for Code" (3 December 2025, updated 23 June 2026): the press "shipped in 1440", the Reformation came "77 years later" [20]; Mahir Yavuz, "Is This the Printing Press Moment of Software?" (16 February 2026): "The key change was not speed. It was access." and "The scribes' guilds fought it." [21]. Brookings (Tom Wheeler) argues the secondary effects of printing included the Reformation [55].
- Not found in my searches: "Scrum priesthood" as an established phrase (no source located), so do not attribute it. Whether anyone has joined agile, Reformation and AI coding explicitly is unverified beyond my limited search.

## Quotable lines (all verified in the sources named)

- "I believe in this concept, but the implementation described above is risky and invites failure." Royce, p. 329 [1].
- "The first rule of managing software development is ruthless enforcement of documentation requirements." Royce, p. 332 [1].
- "During the early phase of software development the documentation is the specification and is the design." Royce, p. 332 [1].
- "Arrange matters so that the version finally delivered to the customer ... is actually the second version." Royce, p. 334 [1].
- "Many-incorrectly-view Royce's paper as the paragon of single-pass waterfall." Larman and Basili [2].
- "[Royce] introduced the concept of the 'waterfall' of development activities." Bell and Thayer [3].
- "We embrace documentation, but not hundreds of pages of never-maintained and rarely-used tomes." Highsmith [8].
- "while there is value in the items on the right, we value the items on the left more." Manifesto [7].
- "Simplicity--the art of maximizing the amount of work not done--is essential." Principle 10 [9].
- "The word 'agile' has been subverted to the point where it is effectively meaningless". Thomas [28].
- "forming an industry group around the four values always struck me as creating a trade union for people who breathe." Thomas [28].
- "The Agile Industrial Complex imposing methods on people is an absolute travesty." Fowler [30].
- "Dark Scrum begins when people who know their old job, but not their new Scrum job, begin to do the Scrum activities." Jeffries [29].
- "Yagni ... does not apply to effort to make the software easier to modify." Fowler [10].
- "Software development is, and has always been, a religion." Atwood [19].

## Caveats and gaps

- I could not read the Royce original on IEEE Xplore, Brooks's 1995 text, Boehm's and Turner's works, the Forrester report, the State of Agile PDF or Bell and Thayer's full publication details; each is marked where second-hand.
- OCR of the Royce scan is imperfect (the documentation page-count and Figure 8-10 page numbers are uncertain).
- Search-engine summaries (e.g. about Stack Overflow 2017, CMMI dates, DZone) were not independently opened and should be re-checked before they go into the published post.

## REFERENCES

1. Royce WW. Managing the Development of Large Software Systems. Proc. IEEE WESCON, Aug 1970, pp. 1-9; reprinted Proc. 9th ICSE, 1987, pp. 328-338. [Internet Archive snapshot of the UMD course copy](https://web.archive.org/web/2016id_/http://www.cs.umd.edu/class/spring2003/cmsc838p/Process/waterfall.pdf); [ACM DL entry](https://dl.acm.org/doi/10.5555/41765.41801).
2. Larman C, Basili VR. Iterative and Incremental Development: A Brief History. IEEE Computer 36(6):47-56, 2003. [PDF](https://www.craiglarman.com/wiki/downloads/misc/history-of-iterative-larman-and-basili-ieee-computer.pdf).
3. Bell TE, Thayer TA. Software requirements: Are they really a problem? TRW; Proc. 2nd ICSE, 1976. [Internet Archive scan](https://archive.org/details/software_requirements_are_they_really_a_problem).
4. Newberry GA. Changes from DOD-STD-2167A to MIL-STD-498. Crosstalk, April 1995 (cited in [2]; not opened).
5. Brooks FP. The Mythical Man-Month after 20 Years (ch. 19), quoted in [Safnet blog](https://blog.safnet.com/2011/12/11/mythical_man-month_planning_for_change) (second-hand).
6. Brooks FP. The Mythical Man-Month, Anniversary Edition (Addison-Wesley, 1995), table of contents: [publisher sample pages](https://www.ciscopress.com/content/images/9780201835953/samplepages/0201835959.pdf).
7. [Manifesto for Agile Software Development](https://agilemanifesto.org/), 2001.
8. Highsmith J. [History: The Agile Manifesto](https://agilemanifesto.org/history.html).
9. [Principles behind the Agile Manifesto](https://agilemanifesto.org/principles.html).
10. Fowler M. [Yagni](https://martinfowler.com/bliki/Yagni.html), 26 May 2015.
11. Murphy B, Bird C, Zimmermann T, Williams L, Nagappan N, Begel A. Have Agile Techniques been the Silver Bullet for Software Development at Microsoft? ESEM 2013. [PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Agile20Trends20ESEM20Master.pdf).
12. Digital.ai. [17th State of Agile Report](https://www.businesswire.com/news/home/20240116199385/en/17th-State-of-Agile-Report), January 2024 (press release; figures as summarised by secondary coverage).
13. Jeffries R. [Developers Should Abandon "Agile"](https://ronjeffries.com/articles/018-01ff/abandon-1/), 10 May 2018.
14. Engprax. [268% higher failure rates for Agile software projects](https://engprax.com/post/268-higher-failure-rates-for-agile-software-projects-study-finds); [response to Jon Kern](https://engprax.com/post/response-to-misleading-claims-by-jon-kern-agile-manifesto-co-author-blasts-failure-rates-report-talks-up-reimagining-project).
15. Bloor Research. [To no-one's surprise, bad Agile is still bad](https://www.bloorresearch.com/to-no-ones-surprise-bad-agile-is-still-bad/).
16. Ulland KT. [The Agile Reformation: a critical look at the manifesto](https://acceptancepod.com/the-agile-reformation-a-critical-look-at-the-manifesto/), Acceptance Criteria, 6 March 2024.
17. [Agile Manifesto: The Reformation That Became the Church](https://dzone.com/articles/agile-manifesto-reformation-to-church), DZone (fetch blocked; snippet only).
18. [Sacred Agile: Religious Approaches to Software Development](https://www.scrumexpert.com/knowledge/sacred-agile-religious-approaches-to-software-development/), Scrum Expert, 12 May 2026.
19. Atwood J. [Software Development: It's a Religion](https://blog.codinghorror.com/software-development-its-a-religion/), Coding Horror, 9 October 2006.
20. Fuller B. [The Gutenberg Moment for Code](https://www.bodenfuller.com/writing/gutenberg-moment-for-code), 3 December 2025.
21. Yavuz M. [Is This the Printing Press Moment of Software?](https://mahir.substack.com/p/is-this-the-printing-press-moment), 16 February 2026.
22. Hogarth S. [Re-reading Royce](https://samhogy.co.uk/2023/07/re-reading-royce/), July 2023.
23. [Waterfall model](https://en.wikipedia.org/wiki/Waterfall_model), Wikipedia (secondary; used for Benington 1956, DoD-STD-2167 date).
24. Ograca H. [11. Plan to Throw One Away](https://herbertograca.com/2018/11/19/11-plan-to-throw-one-away/) (chapter notes quoting Brooks).
25. Fowler M. [Is Design Dead?](https://martinfowler.com/articles/designDead.html), 2000, rev. 2004.
26. Fowler M. [The New Methodology](https://martinfowler.com/articles/newMethodology.html), 2000, rev. 2005.
27. Fowler M. [Semantic Diffusion](https://martinfowler.com/bliki/SemanticDiffusion.html), 14 December 2006.
28. Thomas D. [Agile Is Dead (Long Live Agility)](https://pragdave.me/thoughts/active/2014-03-04-time-to-kill-agile.html), 4 March 2014.
29. Jeffries R. [Dark Scrum](https://ronjeffries.com/articles/016-09ff/defense/), 8 September 2016.
30. Fowler M. [The State of Agile Software in 2018](https://martinfowler.com/articles/agile-aus-2018.html), Agile Australia, 25 August 2018.
31. Fowler M. [Agile Imposition](https://martinfowler.com/bliki/AgileImposition.html).
32. West D. [Water-Scrum-Fall Is The Reality Of Agile For Most Organizations Today](https://www.forrester.com/report/water-scrum-fall-is-the-reality-of-agile-for-most-organizations-today/RES60109), Forrester, 26 July 2011 (paywalled).
33. [Water-Scrum-Fall is the norm](https://www.infoq.com/news/2011/12/water-scrum-fall-is-the-norm), InfoQ; [Analyst Watch](https://sdtimes.com/agile/analyst-watch-water-scrum-fall-is-the-reality-of-agile), SD Times.
34. Gregorio J. [No true X for X in [Scotsman, Scrum, Free Market, Communist State]](https://bitworking.org/news/2008/07/no-true-scotsman/), 2008.
35. [The Strawmen of Agile](https://hackernoon.com/the-strawmen-of-agile), Hackernoon.
36. [Enterprises struggle with agile methodology](https://devclass.com/2024/01/17/enterprises-struggle-with-agile-methodology-reports-long-standing-survey-of-practitioners/), DevClass, 17 January 2024.
37. [Key takeaways from the Stack Overflow 2017 survey](https://www.sitepoint.com/key-takeaways-stack-overflow-2017-developer-survey/), SitePoint (secondary).
38. [Study finds 268% higher failure rates for Agile](https://www.theregister.com/2024/06/05/agile_failure_rates/), The Register, 5 June 2024; [Jon Kern interview](https://www.theregister.com/2024/07/16/jon_kern/).
39. [Mitigating Software Project Failure With Loss-Aversion-Aware Development Methodologies](https://arxiv.org/pdf/2410.20696), arXiv 2410.20696 (not read).
40. [Negative developer comments about Agile and Scrum](https://news.ycombinator.com/item?id=37127218), Hacker News.
41. [Big Design Up Front](https://c2.com/xp/BigDesignUpFront.html), c2 wiki.
42. [Coding blind](https://learn.microsoft.com/en-us/archive/blogs/agileer/coding-blind) and [Agile architecture](https://learn.microsoft.com/cs-cz/archive/blogs/randymiller/agile-architecture), Microsoft archive blogs (recollections).
43. Holgate L. [Joel is a bit confused about agility and design](https://lenholgate.com/blog/2005/08/joel-is-a-bit-confused-about-agility-and-design.html), 2005 (quotes Spolsky).
44. Ambler S. [Examining the Big Requirements Up Front Approach](https://agilemodeling.com/essays/examiningbruf.htm).
45. Boehm B, Turner R. Balancing Agility and Discipline. Addison-Wesley, 2003/2004; Boehm B. Get Ready for Agile Methods, with Care. IEEE Computer 35(1), 2002 (bibliographic details via [Eyrolles](https://www.eyrolles.com/Informatique/Livre/balancing-agility-and-discipline-9780321186126/); not read).
46. [Stage-Gate: origin, status quo and future](https://www.five-is.com/download/stage-gate-origin-status-quo-and-future); [Stage-gate model: history and evolution](https://aaltodoc.aalto.fi/items/8cbdc1f4-37e3-4405-b3b3-00710a5d15b7).
47. FHWA, [Systems engineering guidebook, Vee model](https://www.fhwa.dot.gov/cadiv/segb/views/document/Sections/section4/4_2.cfm); [Vee (V) Model glossary](https://sandbox.sebokwiki.org/Vee_(V)_Model_(glossary)), SEBoK.
48. [Analysis Paralysis](https://sourcemaking.com/antipatterns/analysis-paralysis), SourceMaking; Brown WJ et al., AntiPatterns, 1998.
49. Ambler S. [The History of the Unified Process](https://scottambler.com/unified-process-history/).
50. [Capability Maturity Model](https://en.wikipedia.org/wiki/Capability_Maturity_Model), Wikipedia (secondary).
51. Bossavit L. [The Leprechauns of Software Engineering](https://leanpub.com/leprechauns); summary at [TechWell](https://www.techwell.com/techwell-insights/2013/10/what-does-it-really-cost-fix-software-defect).
52. [Agile religion / Agile atheist](https://www.infoq.com/agile/news/1528), InfoQ (snippet only).
53. [The Death of Agile (Allen Holub)](https://www.vexplode.com/en/agile/the-death-of-agile-allen-holub/), transcript.
54. [The Scrum Cargo Cult](https://medium.com/@jason.godesky/the-scrum-cargo-cult-98def5b4af2f), Medium.
55. Wheeler T. [Gutenberg's message to the AI era](https://www.brookings.edu/articles/gutenbergs-message-to-the-ai-era/), Brookings.
