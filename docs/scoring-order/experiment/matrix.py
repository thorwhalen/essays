"""The scoring task: eight candidate use cases for a fictional graph-analytics startup,
five criteria, and the scores each description was written to support (thorwhalen/essays#28).

Each description gives one piece of evidence per criterion, in no fixed order, written to a
designed 1-5 level. The designed matrix has low (slightly negative) correlation between
criteria, so a rater with no halo should not see criteria rise and fall together; a rater
with halo will. Totals are close (12 to 18), so rankings can flip.
"""

COMPANY = (
    "Nodewise is a small startup with a toolkit for exploring large graphs in the browser: "
    "fast layouts of millions of nodes, clustering, filtering, and timelines of how a network changes."
)

CRITERIA = {
    "pain": "Market pain: how acute and costly the customer's problem is today (5 = losing serious money or time every week and actively looking for a fix; 1 = a nice-to-have nobody asked for).",
    "fit": "Technical fit: how directly Nodewise's existing toolkit solves it (5 = exactly what the toolkit already does; 1 = needs capabilities Nodewise does not have).",
    "reuse": "Reusability: how much of what is built here serves other use cases (5 = almost all of it is reusable; 1 = a bespoke one-off).",
    "moat": "Defensibility: how hard it would be for a competitor to copy the offer (5 = proprietary data or network effects; 1 = trivially copied, free alternatives exist).",
    "speed": "Time to signal: how soon Nodewise would learn whether it works (5 = a clear signal within weeks; 1 = a year or more before any signal).",
}

ALTERNATIVES = {
    "A": ("Fraud rings for regional banks",
          "Regional banks lose millions a year to organised fraud rings and their fraud teams say finding them is their top priority. "
          "Spotting rings means laying out transaction networks and clustering them, which the toolkit does well, though it would need a connector to banking data. "
          "The work is tied to each bank's core-banking formats, so little of it carries over to other projects. "
          "Several fraud-analytics vendors exist, but Nodewise's speed on very large networks is a modest advantage. "
          "Banks run procurement and security reviews that typically take six to nine months before a pilot."),
    "B": ("Dependency maps for software teams",
          "Engineering teams find tangled code dependencies painful and often slow, and many have asked for better maps. "
          "Turning a codebase into a graph needs parsers for many languages that Nodewise has not built, and its layouts suit only part of the problem. "
          "Almost everything built here (importers, layouts for directed graphs, change timelines) would serve nearly every other use case. "
          "Free open-source tools already draw dependency graphs and are easy to replicate. "
          "A plugin could be in front of a few hundred developers within two months."),
    "C": ("Citation networks for academic librarians",
          "Librarians find citation maps interesting, but most say their current search tools are good enough. "
          "Citation networks are exactly the large, clustered graphs the toolkit was built for, with no new capability needed. "
          "About half of the work (bibliographic importers, a timeline of growth) would be reusable elsewhere. "
          "A curated citation dataset assembled with partner libraries would be hard for others to reproduce. "
          "University purchasing runs on yearly budget cycles, so the first real answer would come after more than a year."),
    "D": ("Live map of a partner marketplace",
          "Sellers on a mid-size online marketplace mention the problem only occasionally and have not asked for a solution. "
          "The toolkit covers the visual side, but the analytics the marketplace wants would need some new work. "
          "Most of the pipeline (streaming ingestion, live layouts) would be reused by later projects. "
          "Every seller who joins adds data that makes the map more useful to the others, a network effect a competitor could not copy. "
          "The marketplace offered to put a prototype in front of its sellers next month and share usage numbers weekly."),
    "E": ("Compliance graphs for pharma supply chains",
          "Pharma compliance teams describe tracing suppliers as a recurring headache, though they cope with spreadsheets. "
          "The job is mostly document extraction and regulatory reasoning, with graph views as a small part; Nodewise lacks most of it. "
          "Some of the supplier-mapping work could be reused, but much is specific to pharmaceutical regulation. "
          "Exclusive data-sharing agreements with two industry consortia would give Nodewise data no competitor could get. "
          "One consortium agreed to a six-week trial with three member companies starting soon."),
    "F": ("Threat-intel graphs for a large telecom",
          "A large telecom's security team loses significant money weekly to network intrusions and wants a solution urgently. "
          "Exploring attack graphs over billions of events is precisely what the toolkit does best. "
          "The connectors and investigation views would serve almost every other security or network use case. "
          "Large security vendors already sell similar graph views and could match the features quickly. "
          "The telecom requires a year-long certification process before any production data can be used."),
    "G": ("Family-tree visualiser for hobbyists",
          "Genealogy hobbyists enjoy the idea, but none has said they need it and existing apps satisfy most of them. "
          "Family trees need specialised layouts and record-matching that the toolkit does not have. "
          "Around half of the work (importers, layout options) would be usable elsewhere. "
          "Hobbyist apps are common and new entrants appear often, though a large shared tree could become somewhat sticky. "
          "A free web version could be posted to hobbyist forums this week and draw feedback within days."),
    "H": ("Influence maps for a political-risk consultancy",
          "The consultancy says mapping influence networks takes its analysts days per report, a real but manageable cost. "
          "The toolkit handles the network views, though the consultancy's text sources would need moderate new work. "
          "The work is tailored to the consultancy's proprietary report formats and would barely carry over. "
          "The consultancy's methods are known in the industry, and its competitors use similar off-the-shelf tools. "
          "The consultancy can trial a prototype on its next quarterly report, about a month away."),
}

DESIGNED = {
    "A": {"pain": 5, "fit": 4, "reuse": 1, "moat": 3, "speed": 2},
    "B": {"pain": 4, "fit": 2, "reuse": 5, "moat": 1, "speed": 3},
    "C": {"pain": 2, "fit": 5, "reuse": 3, "moat": 4, "speed": 1},
    "D": {"pain": 1, "fit": 3, "reuse": 4, "moat": 5, "speed": 5},
    "E": {"pain": 3, "fit": 1, "reuse": 2, "moat": 5, "speed": 4},
    "F": {"pain": 5, "fit": 5, "reuse": 5, "moat": 1, "speed": 1},
    "G": {"pain": 1, "fit": 1, "reuse": 3, "moat": 2, "speed": 5},
    "H": {"pain": 3, "fit": 3, "reuse": 1, "moat": 2, "speed": 4},
}
