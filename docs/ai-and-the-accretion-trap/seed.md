# AI and the Accretion Trap: Why Technical Experts Consistently Underestimate Broken Technologies

Seed: https://github.com/thorwhalen/essays/issues/30 (by thorwhalen, 2026-09-18). Fetched 2026-11-06 by penult. This file is the point of departure, kept as it was.

## Body

# **Architectural Skepticism Across Computing Revolutions: A Comparative Analysis of Early Internet Critiques and Modern Artificial Intelligence Discourse**

## **1\. Executive Summary & Epistemological Framework**

Every major transition in computing triggers a rift within the technical community. This division is rarely between those who understand technology and those who do not. Instead, it is an **epistemological conflict between two rigorous, competing engineering traditions**:

* **The Formalist Tradition:** Grounded in deterministic correctness, provable state, clear semantic hierarchies, bounded error rates, and explicit symbolic contracts.  
* **The Pragmatic/Connectionist Tradition:** Grounded in statistical heuristics, rough consensus, emergent behaviors, decentralized topologies, and runtime self-correction.

When a new technological primitive emerges from the latter tradition—such as packet-switched routing in the 1970s, the stateless World Wide Web in the 1990s, or autoregressive Large Language Models (LLMs) today—the formalist tradition evaluates the naked primitive against mature system guarantees.

Their technical critiques are often **factually undeniable at time $t\_0$**. However, their long-term predictions frequently fail because of a consistent blind spot: **confusing an implementation bottleneck of a primitive for an immutable law of computational physics.**

&nbsp;

&nbsp;

&nbsp;

```
┌────────────────────────────────────────────────────────────────────────┐  
│                        THE ACCRETION LIFECYCLE                         │  
└────────────────────────────────────────────────────────────────────────┘

&nbsp;&nbsp;Phase 1: Bare Primitive Exposed  
&nbsp;&nbsp;┌──────────────────────────────────────────────┐  
&nbsp;&nbsp;│ Naked Primitive (HTTP 0.9, Lossy IP, LLM)    │ ──► Low reliability,  
&nbsp;&nbsp;└──────────────────────────────────────────────┘     high latency, no state  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼  
&nbsp;&nbsp;Phase 2: The Formalist Critique  
&nbsp;&nbsp;┌──────────────────────────────────────────────┐  
&nbsp;&nbsp;│ "Statelessness cannot run software."          │ ──► Factually true at t₀;  
&nbsp;&nbsp;│ "Packet loss prevents real-time media."      │     mistakes current limit  
&nbsp;&nbsp;│ "Stochastic tokens cannot reason."           │     for theoretical ceiling  
&nbsp;&nbsp;└──────────────────────────────────────────────┘  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼  
&nbsp;&nbsp;Phase 3: The Synthesist Mitigation (Tooling Accretion)  
&nbsp;&nbsp;┌──────────────────────────────────────────────┐  
&nbsp;&nbsp;│ State Abstractions (Cookies, Sessions, REST) │ ──► Solves limits through  
&nbsp;&nbsp;│ Error Correction (Jitter Buffers, RTP/RTCP)  │     external scaffolding,  
&nbsp;&nbsp;│ Dynamic Runtimes (JIT, V8, WASM)             │     not core rewrites  
&nbsp;&nbsp;│ Verification Harnesses (RAG, MCTS, PRMs)     │  
&nbsp;&nbsp;└──────────────────────────────────────────────┘  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼  
&nbsp;&nbsp;Phase 4: Paradigm Inversion  
&nbsp;&nbsp;┌──────────────────────────────────────────────┐  
&nbsp;&nbsp;│ The once-"broken" primitive becomes the      │ ──► Global standard;  
&nbsp;&nbsp;│ universal substrate for global computation   │     formal systems become  
&nbsp;&nbsp;└──────────────────────────────────────────────┘     specialized subsystems
```

Technologists often fail to anticipate **scaffolding accretion**: the rapid layer-by-layer construction of deterministic, compensating harnesses around a messy, probabilistic, or degraded primitive. This report examines five architectural debates of the early internet era alongside their modern counterparts in generative AI, analyzing the primary source critiques, the counter-arguments, and the socio-technical mechanisms driving these patterns.

## **2\. Historical Case Studies: The Early Internet & Web Era**

### **Case 1: Best-Effort Packet Switching vs. Deterministic Telecom Routing**

#### **The Structural Conflict**

In the 1970s and 1980s, the telecommunications sector operated on circuit switching via the Public Switched Telephone Network (PSTN). Circuit switching was deterministic: when a connection was established, physical or time-division multiplexed circuits were reserved end-to-end. This guaranteed dedicated bandwidth, bounded microsecond jitter, and high reliability ("five nines").

The Internet Protocol (IP), codified in RFC 791 (1981), rejected this approach. It embraced **connectionless, best-effort packet switching**:

* Packets could be dropped arbitrarily when router buffers overflowed.  
* Packets could take divergent routes and arrive entirely out of order.  
* Latency was variable and nondeterministic.

#### **Primary Source Critiques**

Telecom engineers, defense contractors, and classical network architects argued that an inherently lossy, unbounded network could never handle mission-critical, interactive, or real-time voice communications.

In his analysis of this conflict, computing historian Abbate (1999) documents how traditional telecommunication carriers (e.g., AT\&T, European PTTs) viewed ARPANET and IP as fundamentally flawed:

*"AT\&T engineers insisted that packet switching could not support voice traffic, which required guaranteed circuit reservations. They viewed the datagram model not as an innovation, but as an abdication of the network's duty to manage quality of service."*

Classical network designers favored the ITU-T **X.25** standard and later **ATM (Asynchronous Transfer Mode)**, which enforced virtual circuits, explicit traffic policing, and cell-switched determinism. To them, IP was an academic toy that ignored queueing theory.

#### **The Blind Spot & Eventual Resolution**

The critics failed to foresee how dumb networks combined with intelligent endpoints could resolve quality issues:

* **The End-to-End Principle (Saltzer, Reed, & Clark, 1984):** Pushing reliability to host boundaries dramatically lowered intermediate routing costs, accelerating network deployment.  
* **Algorithmic Compensation:** Transport layer enhancements (Jacobson’s congestion control algorithms in TCP, RFC 2581), dynamic jitter buffers, UDP streaming protocols (RTP/RTCP, RFC 3550), and acoustic packet loss concealment (PLC) made best-effort delivery adequate for human conversation.  
* **Bandwidth Over-Provisioning:** Dense Wavelength Division Multiplexing (DWDM) fiber lowered the cost of bandwidth, making over-provisioning far cheaper than maintaining stateful ATM virtual circuits across every hop. By the 2000s, PSTN backbones were replaced by IP networks, and Voice over IP (VoIP) became ubiquitous.

### **Case 2: The Statelessness of HTTP vs. Distributed Transactional Systems**

#### **The Structural Conflict**

In the late 1980s and early 1990s, enterprise computing focused on distributed object architectures: **CORBA (Common Object Request Broker Architecture)**, **DCOM**, and transaction monitors like **Tuxedo**. These systems maintained persistent client-server connections, tracked session state across multiple invocations, and enforced the ACID paradigm (Atomicity, Consistency, Isolation, Durability) across distributed nodes via two-phase commit protocols.

Tim Berners-Lee’s Hypertext Transfer Protocol (HTTP 0.9/1.0, RFC 1945\) took the opposite path:

* Connections closed after single resource transfers.  
* Every request was atomic and isolated.  
* The server retained zero memory of past interactions.

&nbsp;

&nbsp;

&nbsp;

Distributed Object Model (CORBA/RPC, \~1992):  
Client ──\[Open Stateful TCP Connection\]──► Server  
Client ──\[Begin Distributed Transaction\]─► Remote Object (State Locked)  
Client ──\[Invoke Method 1\]───────────────► Remote Object (State Mutated)  
Client ──\[Invoke Method 2\]───────────────► Remote Object (State Mutated)  
Client ──\[Two-Phase Commit / Close\]─────► Persist & Release Locks

Stateless Document Model (HTTP 1.0, \~1995):  
Client ──\[Connect & Send GET /index.html\]─► Server (Zero Session Memory)  
Server ──\[Return Document & Sever TCP\]──► Client drops connection  
(Server instantly forgets client identity, IP state, and navigational history)

#### **Primary Source Critiques**

Architects trained in distributed objects regarded HTTP as a step backward that could never support enterprise applications or commerce.

In a retrospective on the design of the World Wide Web, architectural theorist Roy Fielding (2000) noted that the enterprise computing establishment dismissed HTTP because it could not manage state:

*"The Web was an architectural anomaly... Computer scientists looked at HTTP/1.0 and were appalled by its inefficiency: opening a new TCP connection for every single item retrieved, sending redundant headers, and throwing away connection state... They argued that real distributed enterprise systems required persistent object handles and fine-grained distributed RPC mechanisms."*

Robert Orfali, Dan Harkey, and Jeri Edwards, in their 1996 systems text *The Essential Client/Server Survival Guide*, voiced the prevailing industry sentiment:

*"HTTP is a brain-dead protocol... It treats every interaction as an isolated, amnesic event. You cannot build serious commercial client/server applications when your middleware forgets who the client is between mouse clicks."*

#### **The Blind Spot & Eventual Resolution**

Statelessness turned out to be the Web's primary scaling advantage. By discarding server-side connection affinity, any web server could answer any request, enabling cheap, horizontal load balancing across stateless server farms.

Engineers bridged the functional gap without compromising core statelessness:

* **Lightweight Session Cookies:** Lou Montulli invented HTTP cookies at Netscape in 1994, shifting the storage of ephemeral session identifiers to the client while keeping the network protocol stateless.  
* **RESTful Architecture:** Formalized by Fielding in 2000, REST turned HTTP's apparent flaws (idempotent verbs, uniform resource identifiers, explicit representation caching) into an architectural style for internet-scale services.  
* **Asynchronous Client Runtimes:** The arrival of AJAX (2005) and WebSockets decoupled UI mutations from full-page reloads, transforming the browser into an application runtime without returning to the lock-in of CORBA.

### **Case 3: Unfiltered Information Chaos vs. Curated Taxonomies**

#### **The Structural Conflict**

Information retrieval and library science long prioritized curated classification systems: Dewey Decimal, Library of Congress, and manually indexed corporate databases. When the consumer web emerged, the early organization paradigm followed this philosophy. Yahoo\! hired human catalogers ("surfers") to categorize the web into an ontology:

&nbsp;

$$\\text{Computers and Internet} \\longrightarrow \\text{Software} \\longrightarrow \\text{Operating Systems} \\longrightarrow \\text{Unix}$$

The open web was uncurated, spam-ridden, devoid of consistent metadata (Dublin Core was rarely adopted), and filled with broken links.

#### **Primary Source Critiques**

Astronomer and Unix systems administrator **Clifford Stoll** published an influential op-ed in *Newsweek* (February 26, 1995\) titled *"The Internet? Bah\! Why Cyberspace Isn't and Never Will Be Nirvana"*:

*"What the Internet hucksters won't tell you is that the Internet is one big ocean of unedited data, without any pretense of completeness. Lacking editors, reviewers or critics, the Internet has become a wasteland of unfiltered data. You don't know what to ignore and what's worth reading... We're promised instant catalog shopping—just point and click for great deals. We'll order airline tickets over the network, make restaurant reservations and negotiate sales contracts. Stores will become obsolete. So how come my local mall does more business in an afternoon than the entire Internet handles in a month?"*

&nbsp;

Stoll expanded this argument in his 1995 book, *Silicon Snake Oil: Second Thoughts on the Information Highway*:

*"A cacophony of voices, all shouting at once... It is an anarchic medium where the noise completely drowns out the signal, precisely because the barriers to entry are zero."*

#### **The Blind Spot & Eventual Resolution**

Critics like Stoll evaluated the problem through the lens of **pre-publication editorial filtering**. They assumed that if quality was not enforced at ingestion, retrieval would degrade into noise.

They missed the shift to **post-publication algorithmic filtering**:

* **Graph Topology as Latent Consensus:** In 1998, Larry Page and Sergey Brin formulated **PageRank**:

&nbsp;

$$R(u) \= \\frac{1-d}{N} \+ d \\sum\_{v \\in B\_u} \\frac{R(v)}{L(v)}$$

PageRank treated hyperlinked structures not as an unedited mess, but as a dynamic, crowdsourced citation graph. The open web's noise became the very training corpus used to identify relevance, outperforming human ontologies like Yahoo\!'s directory within years.

### **Case 4: Client-Side Scripting as an Insecure Toy vs. Compiled Desktop Runtimes**

#### **The Structural Conflict**

In 1995, serious client application engineering was built on compiled, type-checked, optimized languages (C, C++, Object Pascal) or managed bytecode runtimes with rigorous sandbox architectures, such as Sun Microsystems’ Java Applets.

In May 1995, Brendan Eich created **Mocha** (later renamed LiveScript, then JavaScript) in 10 days for Netscape Communications. It was an interpreted, dynamically typed language with prototypal inheritance, permissive implicit type coercion, and direct access to the Document Object Model (DOM).

&nbsp;

&nbsp;

&nbsp;

&nbsp;&nbsp;&nbsp;1995 Computing Orthodoxy                 Brendan Eich / Netscape Paradigm  
┌────────────────────────────────┐         ┌────────────────────────────────┐  
│ • Strongly & statically typed  │         │ • Dynamically & weakly typed   │  
│ • Ahead-of-time compiled (C++) │   vs.   │ • Interpreted on-the-fly       │  
│ • Explicit memory management   │         │ • Garbage collected            │  
│ • Heavy enterprise runtimes    │         │ • Delivered via text inside DOM│  
└────────────────────────────────┘         └────────────────────────────────┘

#### **Primary Source Critiques**

Systems engineers and enterprise architects criticized the language's security risks, performance flaws, and bizarre type coercion rules:

&nbsp;

$$\[\] \+ \\{\\} \\implies \\text{"\[object Object\]"} \\quad \\text{versus} \\quad \\{\\} \+ \[\] \\implies 0$$

Writing in *Dr. Dobb's Journal* and Usenet newsgroups in the late 1990s, software commentators routinely warned against using JavaScript for anything beyond simple form validation. As software engineer and author Michael O'Connell wrote in *SunWorld* (1996):

*"JavaScript lacks the robustness, typing discipline, and security boundary guarantees required for serious computing... It is an insecure scripting hack thrown together to animate images and manipulate forms, distracting developers from robust, compiled network languages like Java."*

Software luminary **Joel Spolsky** later noted the traditional enterprise developer's perspective in an essay reflecting on the era:

*"For years, real programmers looked down on JavaScript as a toy language for script kiddies to make annoying status bar animations... Real work was done in C++ or Java."*

#### **The Blind Spot & Eventual Resolution**

Traditional developers prioritized language purity and compile-time guarantees over **zero-installation distribution friction**. JavaScript could be delivered instantly over the wire and run immediately inside an OS-agnostic viewport without requiring users to download binaries or run JVM plugins.

Instead of replacing the flawed language, the ecosystem engineered the runtime environment:

* **Just-In-Time (JIT) Compilation:** In 2008, Lars Bak and Google built the **V8 Engine**, employing inline caches and dynamic machine code generation to accelerate JavaScript execution to speeds comparable to native code.  
* **Static Type Layers:** TypeScript (2012) added compile-time typing over the runtime without sacrificing dynamic web delivery.  
* **Virtual Machines & Sandboxing:** Browsers transformed into secure operating systems, rendering compiled plugins like Flash, Silverlight, and Java Applets obsolete.

### **Case 5: The Static Document Assumption vs. The Application Platform**

#### **The Structural Conflict**

Tim Berners-Lee envisioned the World Wide Web as an academic hypertext documentation repository. The architecture reflected this: HTML was an SGML dialect meant to define semantic document structure (\<p\>, \<h1\>, \<cite\>), not absolute spatial layouts or interactive components.

#### **Primary Source Critiques**

When designers began transforming HTML into an interactive layout engine using nested tables, transparent spacer GIFs, and CSS hacks, systems purists and usability pioneers pushed back.

Usability pioneer **Jakob Nielsen**, writing in his famous *Alertbox* column (*"Top Ten Mistakes in Web Design"*, 1996), maintained that the Web should not mimic desktop application software:

*"The Web is an information space, not a software platform. Using the Web as an interactive application delivery mechanism breaks the fundamental mental model of users, violating the browser's navigation conventions—like the 'Back' button—and results in slow, confusing, non-standard user interfaces."*

#### **The Blind Spot & Eventual Resolution**

Nielsen accurately observed that the early Web made for a poor interactive application platform. However, he underestimated **user demand for ubiquitous, cross-platform access**. Users preferred an accessible, slightly clunky web application to an isolated, platform-locked, installation-heavy desktop client. Over two decades, browser standards evolved from static document layouts into full application platforms featuring Canvas, WebGL, WebSockets, LocalStorage, ServiceWorkers, and WebAssembly.

## **3\. Structural Analysis: Why Technical Experts Stumble**

Technological skepticism among engineers often stems not from ignorance, but from domain-specific expertise. When engineers develop deep mastery of an established paradigm, their mental models can create structural blind spots during paradigm shifts:

| Structural Failure Mode | Internet Era Exemplar (1990–1999) | Modern AI Era Exemplar (2020–Present) |
| :---- | :---- | :---- |
| **1\. The Metric Substitution Fallacy** | Measuring IP's fitness for voice by circuit-switched latency guarantees rather than operational deployment costs. | Evaluating LLM intelligence using deterministic symbolic benchmarks rather than cross-domain synthesis and code completion speed. |
| **2\. The Primitive Conflation Trap** | Conflating the statelessness of bare HTTP 1.0 with the architectural possibilities of the entire web application ecosystem. | Conflating the stochastic token prediction of raw foundation models with complete agentic pipelines, tool-use loops, and verifiers. |
| **3\. The Curated Ingestion Bias** | Believing information retrieval requires pre-publication human curation (Yahoo\!, Gopher) rather than post-publication algorithmic ranking (PageRank). | Assuming AI reasoning requires hand-engineered, top-down knowledge bases (Cyc, symbolic graphs) rather than post-training self-play and RL. |
| **4\. Zero-Friction Distribution Underestimation** | Dismissing JavaScript because of weird type coercion, while missing the utility of zero-install, browser-level code execution. | Dismissing probabilistic code generation because of non-zero bug rates, while missing the impact of zero-latency draft synthesis. |
| **5\. The Static Boundary Mistake** | Assuming network bandwidth would remain capped at 28.8k dial-up speeds, making video-on-demand a structural impossibility (Stoll). | Assuming frontier models will remain constrained by static pre-training datasets, ignoring synthetic data, inference compute, and process reward models. |

## **4\. The Contemporary Battlefield: Generative AI and Foundation Models**

A similar philosophical divide is playing out today around generative artificial intelligence, transformer architectures, and large language models (LLMs). The technical objections raised by skeptics are largely valid reflections of bare foundation models at time $t\_0$, but they often miss the trajectory of the surrounding scaffolding.

&nbsp;

&nbsp;

&nbsp;

┌────────────────────────────────────────────────────────────────────────┐  
│             THE ARCHITECTURAL SHIFT: FROM BARE TO SCAFFOLDED           │  
└────────────────────────────────────────────────────────────────────────┘

&nbsp;Traditional ML / Skeptical Target:  
&nbsp;Prompt ──────────────────► \[ Raw LLM Transformer \] ──────────────────► Output  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Probabilistic Token Guess)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Subject to Hallucinations & Drift)

&nbsp;Modern Agentic AI System Architecture:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;┌────────────────────────┐  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│ External Deterministic │  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│ Environment / Context  │  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└───────────┬────────────┘  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▲  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│ Retrieval (RAG)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼  
&nbsp;Prompt ──► \[ Search & Reason \] ──► \[ Core Model \] ──► Draft Token Stream  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▲                      │                      │  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│                      ▼                      ▼  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│             \[ Tool / Code Exec \]   \[ Unit Tests & PRM \]  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│                      │                      │  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ Verification Loop ◄─┴──────────────────────┘  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼ (Verified Execution)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Final Deterministic Output

### **Case 1: "LLMs are Stochastic Parrots Without Real World Models"**

#### **The Contemporary Critique**

Critics argue that autoregressive transformers are merely predicting the next token based on statistical correlations in text:

&nbsp;

$$P(w\_t \\mid w\_1, w\_2, \\dots, w\_{t-1})$$

Because they lack sensory grounding and true world models, skeptics assert they are fundamentally incapable of robust reasoning, abstraction, or planning.

Linguist **Emily M. Bender**, computational linguist **Timnit Gebru**, and colleagues published a foundational critique in their 2021 paper, *"On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?"*:

*"An LM is a system for haphazardly stitching together sequences of linguistic forms it has observed in its vast training data, according to probabilistic information about how they combine, but without any reference to meaning: a stochastic parrot."*

Cognitive scientist **Gary Marcus** has repeatedly argued that deep learning has hit a wall regarding abstract reasoning:

*"Deep learning is good at perception, but terrible at reasoning... What we have actually made is fun and amazing, but rarely reliable and often makes mistakes... they still have no understanding of what they are doing. Manipulating symbols without understanding their semantic connection to the real world inevitably hits a wall."*

Meta's Chief AI Scientist **Yann LeCun** has similarly characterized autoregressive models as an architectural dead end for Artificial General Intelligence (AGI):

*"Auto-regressive generation is a dead end... It's not fixable. The error accumulation is exponential. You cannot achieve true planning, hierarchical reasoning, or common-sense world models simply by scaling next-token prediction on text."*

#### **The Counter-Perspective & The Blind Spot**

The counter-camp does not deny that bare autoregression accumulates error across long horizons. However, they argue that **learning accurate next-token distributions over internet-scale code, math, and prose requires the model to develop implicit, compressed world models**. Furthermore, the raw model is only the generator inside a larger computational pipeline.

Computer scientist and OpenAI co-founder **Ilya Sutskever** articulated the world-model thesis:

*"Predicting the next token well means that you understand the underlying reality that led to the creation of that token... If you read a detective novel, and there is a mystery, and at the end of the book the detective gathers everyone and says 'The murderer is...', in order to predict that next word correctly, you have to have understood the entire narrative, the human psychology, and the physical constraints of the world."*

**Andrej Karpathy** frames this shift around the concept of **Software 2.0 and 3.0**:

*"Software 1.0 is code written by humans in languages like C++ or Python. Software 2.0 is code written by neural network optimization based on an evaluation metric. Software 3.0 is programs steered by prompts and agentic orchestration... Dismissing the entire field because the bare underlying engine predicts tokens is like dismissing modern web applications because Ethernet drops packets."*

#### **The Engineering Mitigation**

The industry is addressing the limitations of next-token prediction by building scaffolding around the core models:

* **Inference-Time Search & Test-Time Compute:** Models generate multiple candidate solutions through tree-search algorithms (such as Monte Carlo Tree Search) and evaluate them using **Process Reward Models (PRMs)** rather than emitting an unverified token stream.  
* **External Verification Loops:** Connecting neural outputs to deterministic interpreters (Python, Lean, Z3 solvers). The model acts as an intuitive hypothesis generator, while the symbolic runtime checks correctness and catches hallucinations before execution.

### **Case 2: "Hallucinations Make Probabilistic Code Synthesis Infeasible for Production"**

#### **The Contemporary Critique**

Software engineers frequently point out that LLMs hallucinate nonexistent API methods, introduce security vulnerabilities, and generate code that appears plausible but fails silently on edge cases. The prevailing formalist critique contends that the verification overhead outweighs the productivity gains: *If an engineer must review every line of generated code to avoid subtle bugs, writing it manually from first principles is safer and more efficient.*

Prominent software engineers and open-source advocates share this view. Longtime developer and author **David Heinemeier Hansson (DHH)** observed:

*"LLMs generate plausible-looking bullshit with the supreme confidence of a mediocre consultant... In programming, code that is 95% right is often 100% wrong, because the last 5% is where the actual engineering happens—the edge cases, the security boundaries, and the architectural invariants."*

#### **The Counter-Perspective & The Blind Spot**

Critics often assume that developers interact with models by pasting generated text directly into production codebases without automated checks.

This argument mirrors early criticisms of high-level languages like C, garbage collectors, and dynamic runtimes:

&nbsp;

&nbsp;

&nbsp;

┌────────────────────────────────────────────────────────────┐  
│                  THE ABSTRACTION RESISTANCE                │  
└────────────────────────────────────────────────────────────┘

&nbsp;&nbsp;1970s Assembly Programmers:  
&nbsp;&nbsp;"Compilers generate bloated, suboptimal machine instructions\!  
&nbsp;&nbsp;&nbsp;A human hand-tuning assembly will always beat automated C output."  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼ (Result: Compilers improved; developer  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;throughput multiplied 10x)

&nbsp;&nbsp;2020s Deterministic Developers:  
&nbsp;&nbsp;"LLMs write buggy, hallucinated, insecure functions\!  
&nbsp;&nbsp;&nbsp;A human typing manual syntax will always beat probabilistic code."  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼ (Result: Test-driven agentic loops generate,  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;lint, test, and repair code automatically)

Computer scientist **François Chollet** (creator of Keras) notes that the nature of development work changes as abstractions advance:

*"The problem isn't whether LLMs hallucinate or write imperfect code on the first pass; human software engineers write imperfect code on the first pass constantly. Software engineering has never been about typing code correctly; it is about verifying, testing, and debugging against operational constraints."*

#### **The Engineering Mitigation**

Instead of treating code generation as a single-shot completion task, builders wrap LLMs in **test-driven agentic loops**:

> 1. The model produces both a code candidate and a corresponding unit test harness.  
> 2. An isolated sandbox compiles the code and executes the tests.  
> 3. If an exception or assertion fails, the stack trace feeds back into the prompt context for self-correction.  
> 4. Static analysis tools (linters, AST parsers, type checkers) enforce security boundaries deterministically before the PR is surfaced for human review.

### **Case 3: "The Impending Data Wall and Economic Collapse of Scaling Laws"**

#### **The Contemporary Critique**

Skeptics argue that foundation model progress is hitting an insurmountable wall due to two physical constraints:

> 1. **Data Exhaustion:** The pool of human-generated tokens on the public web (books, Wikipedia, Reddit, GitHub) has been largely consumed. Feeding models their own synthetic outputs can trigger model collapse (loss of variance and distributional drift).  
> 2. **Thermodynamic and Economic Costs:** Training cluster energy demands are scaling exponentially, resulting in diminishing returns per dollar invested.

Venture capitalist and tech commentator **Paul Kedrosky**, writing with Gary Marcus in *The Atlantic* and on Substack, pointed out the industry's steepening cost curves:

*"The AI industry is running out of other people's data... We are seeing diminishing returns to scale. The marginal cost of training these mega-models is exploding into billions of dollars and gigawatts of power, while the marginal capability improvements are visibly slowing down. The scaling hypothesis is hitting the wall of physics and economics."*

#### **The Counter-Perspective & The Blind Spot**

The critique assumes that models learn solely from uncurated, human-written web text. This parallels early telecom projections that internet traffic would saturate global copper wiring, which failed to anticipate fiber optics, algorithmic caching, and multiplexing.

**Ilya Sutskever** addressed this pivot away from raw pre-training volume:

*"The 2010s was the age of scaling pre-training... finding more data, building bigger clusters. That era has shifted. We are now entering the era of test-time compute, reasoning models, and synthetic data verification. Compute during inference changes the scaling calculus completely."*

Technologist and investor **Marc Andreessen** highlighted how high-level architectures adapt to physical constraints:

*"Every single technological revolution hits a 'cost wall' when evaluated through the lens of brute-force consumption of existing inputs... The critics assume the engineering stops at the boundary of current inputs, ignoring how synthetic data generation, architectural optimization, and domain-specific post-training reshape the underlying economics."*

#### **The Engineering Mitigation**

* **Inference-Time Scaling Laws:** Discoveries around test-time compute show that spending compute dynamically during inference (deliberation, self-correction, tree search) improves benchmark performance more cost-effectively than scaling pre-training parameters alone.  
* **Synthetic Data Pipelines with Ground-Truth Verifiers:** Models are trained not on raw web scrapings, but on high-entropy synthetic data filtered by deterministic validators (compilers, formal math solvers, game engines). This filters out invalid data and prevents recursive model collapse.

### **Case 4: "Copyright Infringement and the Inevitable Legal Collapse"**

#### **The Contemporary Critique**

Legal scholars and intellectual property rights holders argue that generative models are derivative works built on widespread copyright infringement. By training on copyrighted books, journalism, code, and artwork without explicit consent or compensation, AI companies face potential legal, regulatory, or liability crises.

In lawsuits filed by authors, artists, and organizations like the *New York Times*, plaintiffs argue that neural networks serve as unauthorized distribution systems for copyrighted content:

*"Defendants seek to free-ride on The Times's massive investment in its journalism by using it to build substitutive products without permission or payment... A business model predicated on the mass misappropriation of intellectual property cannot stand."*

#### **The Counter-Perspective & The Blind Spot**

Critics view LLMs as glorified file-storage and compression engines, expecting courts to treat them like Napster (unauthorized file distribution).

In contrast, AI legal scholars and technologists point to precedents like **Sony Corp. v. Universal City Studios (1984)** (fair use for recording devices), **Kelly v. Arriba Soft (2003)** (search engine thumbnail caching), and **Authors Guild v. Google (2015)** (the Google Books digitization project). In the Google Books case, the Second Circuit ruled that scanning books to build a searchable statistical index was transformative fair use.

Venture capitalist **Marc Andreessen** commented on the legal trajectory:

*"We have seen this script play out with the phonograph, the radio, the VCR, MP3s, and search engines. Every time a new technology decouples the transmission of information from historical distribution monopolies, incumbents declare it an existential copyright violation... What follows is not the death of the technology, but a transition toward fair use frameworks and statutory blanket licensing schemes."*

#### **The Engineering & Institutional Mitigation**

* **Machine Unlearning & Content Attribution:** Developing algorithmic unlearning techniques to excise specific training sets from model weights on demand.  
* **Institutional Licensing:** Moving toward structured data-licensing agreements with major publishers (e.g., Reddit, Axel Springer, News Corp) to formalize training data access.  
* **Synthetic Replacement:** Swapping litigious training corpuses with procedurally generated synthetic domains for foundational pre-training.

## **5\. Comparative Structural Taxonomy**

The following matrix maps early internet debates to their generative AI counterparts, showing the consistency of technical criticism and evolutionary response across both eras:

| Dimension | Early Internet & Web Era (1990–2000) | Generative AI & Foundation Models (2020–Present) |
| :---- | :---- | :---- |
| **The Naked Primitive** | Stateless, lossy, best-effort communication protocols (IP, HTTP 1.0). | Stochastic, autoregressive, token-prediction neural networks (Transformers). |
| **The Formalist Critique** | *"Statelessness makes real software impossible; lossy routing cannot support voice or commerce."* | *"Stochastic token predictors cannot reason, plan, or understand; error accumulation is exponential."* |
| **The Blind Spot** | Viewing the naked protocol as the whole system, ignoring endpoints and compensating software scaffolding. | Viewing the bare model weights as the whole application, ignoring test-time compute, agentic loops, and external tooling. |
| **The Primary Skeptics** | Systems architects, telecom engineers, library scientists (Stoll, PTTs, CORBA designers). | Cognitive scientists, formal linguists, classical ML researchers (LeCun, Marcus, Bender). |
| **The Primary Visionaries** | Protocol builders, open-source engineers, web pioneers (Berners-Lee, Andreessen, Cerf, Page). | Frontier lab architects, connectionist researchers (Karpathy, Sutskever, Hassabis, Chollet). |
| **Engineering Mitigations** | Cookies, sessions, AJAX, TCP congestion control, VoIP jitter buffers, PageRank. | Retrieval-Augmented Generation (RAG), Process Reward Models (PRMs), Tree Search, Tool/Code Execution, Linters. |
| **Ultimate Outcome** | The degraded primitive became the universal substrate; stateful systems were rebuilt on top. | The probabilistic model becomes the universal cognitive engine; formal systems act as deterministic validators. |

## **6\. Synthesis: The Pragmatist Path Forward**

The historical record suggests that the ultimate utility of a foundational computing primitive is rarely determined by its initial limitations. Rather, it depends on whether the primitive's advantages—such as low distribution friction, horizontal scalability, or cross-domain pattern transfer—outweigh the cost of building compensating software around it.

&nbsp;

&nbsp;

&nbsp;

┌────────────────────────────────────────────────────────────────────────┐  
│               THE CONVERGENT ARCHITECTURE OF MODERN AI                 │  
└────────────────────────────────────────────────────────────────────────┘

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;┌────────────────────────────────────────┐  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│      Probabilistic Core Engine         │  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│  (Intuition, Synthesis, Broad Search)  │  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──────────────────┬─────────────────────┘  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Interactive Exchange Loop  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;┌────────────────────────────────────────┐  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│     Deterministic Tooling Harness      │  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│ (Compilers, Solvers, State Machines)   │  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└────────────────────────────────────────┘

The early internet succeeded not because best-effort packet switching and HTTP were conceptually flawless, but because engineers accepted their trade-offs and built practical abstractions on top: cookies for state, jitter buffers for voice, and search indices for uncurated data.

Generative AI appears to follow this same architectural arc. Foundation models are noisy, non-deterministic, and prone to hallucinations when running unassisted. However, engineers are not deploying them in a vacuum. Instead, they are integrating these probabilistic engines into hybrid architectures, surrounding them with compilers, symbolic solvers, retrieval systems, and verification loops.

Engineers who focus solely on the failure modes of the bare primitive risk misinterpreting temporary implementation hurdles as fundamental limits. Meanwhile, those who build the compensating scaffolding continue to define the next paradigm of computing.

## **7\. References**

* Abbate, J. (1999). *Inventing the Internet*. MIT Press.  
* Bender, E. M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). "On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?" *Proceedings of the 2021 ACM FAccT Conference*, 610–623.  
* Berners-Lee, T., Fielding, R., & Frystyk, H. (1996). *Hypertext Transfer Protocol \-- HTTP/1.0*. RFC 1945\.  
* Brin, S., & Page, L. (1998). "The anatomy of a large-scale hypertextual Web search engine." *Computer Networks and ISDN Systems*, 30(1-7), 107–117.  
* Chollet, F. (2019). "On the Measure of Intelligence." *arXiv preprint arXiv:1911.01547*.  
* Fielding, R. T. (2000). *Architectural Styles and the Design of Network-based Software Architectures*. Doctoral dissertation, University of California, Irvine.  
* Jacobson, V. (1988). "Congestion avoidance and control." *ACM SIGCOMM Computer Communication Review*, 18(4), 314–329.  
* Karpathy, A. (2017). "Software 2.0." *Medium*.  
* LeCun, Y. (2022). "A Path Towards Autonomous Machine Intelligence." *Open Review*.  
* Marcus, G. (2020). "The Next Decades in AI: Four Steps Towards Robust Artificial Intelligence." *arXiv preprint arXiv:2002.06177*.  
* Nielsen, J. (1996). "Top Ten Mistakes in Web Design." *NN/g Alertbox*.  
* Orfali, R., Harkey, D., & Edwards, J. (1996). *The Essential Client/Server Survival Guide*. John Wiley & Sons.  
* Postel, J. (1981). *Internet Protocol: DARPA Internet Program Protocol Specification*. RFC 791\.  
* Saltzer, J. H., Reed, D. P., & Clark, D. D. (1984). "End-to-end arguments in system design." *ACM Transactions on Computer Systems (TOCS)*, 2(4), 277–288.  
* Schulzrinne, H., Casner, S., Frederick, R., & Jacobson, V. (2003). *RTP: A Transport Protocol for Real-Time Applications*. RFC 3550\.  
* Stoll, C. (1995a). "The Internet? Bah\! Why Cyberspace Isn't and Never Will Be Nirvana." *Newsweek*, 125(9), February 26, 1995\.  
* Stoll, C. (1995b). *Silicon Snake Oil: Second Thoughts on the Information Highway*. Doubleday.  
* Sutskever, I. (2023). "An Observation on Generalization and Next-Token Prediction." *Invited Talk at Stanford MLSys Seminar*.

**Sources**

1\. [https://1995blog.com/2015/02/26/the-internet-bah-classic-off-target-essay-appeared-20-years-ago/](https://1995blog.com/2015/02/26/the-internet-bah-classic-off-target-essay-appeared-20-years-ago/)

2\. [https://1995blog.com/2015/02/26/the-internet-bah-classic-off-target-essay-appeared-20-years-ago/](https://1995blog.com/2015/02/26/the-internet-bah-classic-off-target-essay-appeared-20-years-ago/)

3\. [https://youtube.com/v/aa2HXQ1G-MM](https://youtube.com/v/aa2HXQ1G-MM)
