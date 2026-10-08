# Where to publish the essay, and how to semi-automate it: Substack vs Medium (2026)

Research date: 2026-10-08. Scope: a long-form technical-opinion essay by Thor Whalen, drafted as a GitHub issue in thorwhalen/essays and moved to Discussions when done.

## Summary and recommendation

1. Publish the original on Substack (thorwhalen.substack.com), where the author already publishes recent essays; the essays Discussions show Medium for 2024 to early 2025 and Substack from August 2025 on, with one December 2025 essay on a Medium publication [1][2].
2. Reason: you own the subscriber list, discovery runs through Recommendations and Notes, and the editor supports the constructs this essay uses (footnotes, code blocks, captioned images, LaTeX). Tables are the one hole (see section 1.3). Claims about Notes and Recommendations performance come only from third-party growth blogs, not Substack data [3].
3. Optionally syndicate to Medium a few days later, with the canonical link pointing at the Substack URL. Medium's "Import a story" sets the canonical automatically [4]. Substack does not let you set a canonical on its own posts (community-reported, not found in official docs) [5], so the Substack copy must be the original.
4. Automation: the Medium API is closed to new integrations, so do not build on it [6]. The `article` package (thorwhalen/article) is fully stubbed: its Substack and Medium adapters make no network calls [7]. The working route today is the unofficial `python-substack` library, which turns Markdown (footnotes, captions, LaTeX, local image upload) into an unpublished Substack draft [8]. It does not convert tables.
5. Recommended minimal path: markdown file with images, then `substack drafts create`, then a manual review and publish in the Substack editor, then (optionally) Medium "Import a story" by URL. Details in section 4.
6. Unverified items are flagged inline. The main ones: whether Substack's rich-text paste preserves footnotes from rendered HTML, whether Medium's importer keeps footnotes and tables, and the current `python-substack` version number.

## 0. What the essays repo shows

Querying the Discussions of thorwhalen/essays (first 20) shows the platform pattern. Entries dated 2024-07 to 2025-03 link to `medium.com/@thorwhalen1/...` (many as friend links). Entries from 2025-08 (AI on a Bad Trip, AI Privacy) and 2025-12 (The Internet After Browsing) link to `thorwhalen.substack.com`. The 2025-12-23 entry, The AI Engagement Gap, links to a Medium publication (`medium.com/industria-tech/...`) rather than the personal Medium profile [1][2]. So the "Medium then Substack" pattern holds, with one recent Medium-publication exception, which I read as a one-off (inference, not confirmed by the author).

## 1. Substack vs Medium in 2026

### 1.1 Audience and discovery

- Substack: discovery runs through Recommendations (other publications recommending yours) and Notes. A growth guide claims Recommendations account for roughly 30 to 40 percent of new subscribers for established writers and that creators posting three or more Notes a week outperform occasional posters. Both numbers come from vendor or creator blogs with undisclosed methods; treat them as anecdotal [3].
- Medium: a large reader base that browses by tag, and distribution through publications and Boost. Comparison pages describe Medium as good for early reach and Substack as better for owned audience [9][3].
- Better Programming, the large Medium programming publication, was put on hiatus by Medium on 2023-11-10 so that Medium could favour more focused programming publications. I could not confirm whether it has resumed [10]. So "submit to Better Programming" is not a dependable 2026 route; check the Medium publications still accepting programming essays instead.
- Cross-platform claim "Medium has higher domain authority" versus "Substack with a custom domain ranks well" is contested in the sources, with no measurement [3]. I could not verify either.

### 1.2 Owning the list and monetisation

- Substack: you hold the email list and can export it; paid subscriptions carry a 10 percent platform fee plus Stripe fees (sources give 13 to 17 percent total on small plans). These figures come from third-party blogs, several of which sell competing products; confirm on Substack's own pricing pages [11].
- Medium: the audience is Medium's. Partner Program payouts depend mainly on paying members' reading time. Medium's help centre says a paywalled story earns a one-time bonus when a non-member becomes a paying member to read it (replacing the earlier referral scheme), and that a member read-ratio multiplier (readers of 30 seconds or more over all members who opened the story) is applied at the end [12]. Income claims in vendor blogs range from near zero to 2,000 USD a month and are not reliable [3].
- For an opinion essay whose goal is reach and an owned audience rather than per-read income, the list matters more than the Partner Program. This is my judgement, not a sourced fact.

### 1.3 SEO and canonical URLs

- Medium: "Import a story" (Stories page, paste URL) applies a canonical link to the original automatically and backdates the story; you can also set it by hand under More settings, Advanced settings, "originally published elsewhere", and republish [4][13]. Imports can fail on some pages, in which case the help article's fallback is to paste the content and set the canonical manually [4].
- Substack: writers report that Substack does not let you set a canonical link on a post and sets its own [5]. I did not find an official statement. Practical consequence: Substack is the original, Medium is the copy.
- Suggested ordering from practitioners: publish on Substack first, wait a day or two, then import into Medium with the canonical [5]. Verify the tag with "view page source" and, if indexing looks wrong, in Search Console, since one writer reports Google picking the Medium copy over the original [5].
- If a Substack custom domain is planned, settle it before importing so the canonical points at the final URL [5].

### 1.4 Editor capabilities relevant to this essay

| Feature | Substack | Medium |
|---|---|---|
| Footnotes | Native footnote element (inline marker, auto-numbered list at the end, click-back). Mobile display reported to be a popup that can be cut off [14]. | No built-in footnotes per an older Medium help article as summarised by a third party; workaround is superscripts plus a list, or links [15]. Unverified for the current editor. |
| Code blocks | Native, with syntax highlighting, auto language detection, line numbers and copy button [16]. | Code blocks exist; I did not find a current primary source on highlighting. Unverified. |
| Tables | No native table button; HTML tables collapse to text; workarounds are LaTeX tables or embedded Datawrapper charts [17][8]. | No source found either way. Test in a draft. Unverified. |
| Images with captions | Yes (captioned images; `python-substack` maps Markdown image titles to captions) [8]. | Yes in the editor; via API, images in HTML are side-loaded [6]. |
| LaTeX | MathJax-based LaTeX blocks and inline; reported quirks (limited command set, possible block length limit from an undated post) [17]. | Not supported natively as far as I could find. Unverified. |
| Embeds | Yes (native widgets; `python-substack` cannot create them from Markdown) [8]. | Yes (URL embeds). No source retrieved; unverified. |

Net for this essay: the DFW-style footnote and captioned images fit Substack natively. If the essay contains a table, either rewrite it as a list, render it as an image with a caption and alt text, or embed a chart.

### 1.5 Recommendation

Primary: Substack. Secondary: Medium import after a few days, canonical to Substack, if you want Medium's tag-browsing readers. Skip Medium's API entirely (section 2.2). If the essay depends on a table, convert it to an image or list before publishing, or accept that it will not render in either place.

## 2. Automation options

### 2.1 Substack

- There is no official public write API; the options use undocumented internal endpoints or the browser. The project page of `python-substack` states that it "uses undocumented Substack interfaces that may change without notice" and is not affiliated with Substack [8].
- `python-substack` (ma2za/python-substack, on PyPI) [8][18]:
  - Auth: email and password, or browser cookies via a cookies file or string (recommended when captcha or magic-link sign-in is required).
  - CLI: `substack drafts create post.md` always creates an unpublished draft; also `drafts list/get/export/update/schedule/unschedule/publish/delete`; publish and delete require confirmation (`--yes` in noninteractive use); `--dry-run` for updates.
  - Markdown support listed: headings, bold/italic, inline code, strikethrough, super/subscript, links, images, linked images, image captions, code blocks, blockquotes, lists, horizontal rules, footnotes, LaTeX math, pull quotes, callouts.
  - Footnote syntax: `[^1]` with `[^1]: text` definitions, rendered as footnote blocks at the end; image caption via the Markdown title: `![alt](url "caption")`; math with `$...$` and `$$...$$` using Pandoc's opening-delimiter rule [8].
  - Local images are uploaded when the converter is given the API object.
  - Tables: not supported; "Substack has no table renderer or editor UI for them, so GFM table syntax is not converted" [8].
  - Editor-only widgets (buttons, polls, embeds) have no Markdown equivalent but are preserved on update unless explicitly allowed to change [8].
  - I did not verify the current version number or run the library. Check PyPI before pinning [18].
- Substack's own importer (Settings, Import/Export, Import posts) accepts a Medium profile URL, a CSV, or an RSS feed from other platforms; comments and likes do not carry over, and imported posts may keep old subscribe prompts and links to the old platform [19]. It imports existing posts from another platform; it is not a way to publish a new Markdown file. Whether it can import a single arbitrary URL is not stated in the help article.
- Paste: whether pasting rendered HTML (for example from a browser preview of the Markdown) into the Substack editor preserves headings, links, code and footnotes is plausible but I found no primary source. Unverified; test with a throwaway draft.
- Browser automation (Playwright) driving the editor is the approach the `article` package sketches but does not implement [7].

### 2.2 Medium

- The Medium API docs state "The Medium API is no longer supported. We do not recommend using it" and "We don't allow any new integrations with our API"; the repository was archived on 2023-03-02 [6].
- Integration tokens: a StackOne guide says Medium stopped issuing new integration tokens on 2025-01-01 and that existing tokens keep working; n8n's docs say you cannot set up new integrations [20][21]. These are secondary sources; I could not confirm the date from Medium itself, and I could not test whether the Security and apps page still offers tokens for this account. If the "Integration tokens" section is not visible in Medium settings, assume no access [20].
- Format: `contentFormat` accepts `html` or `markdown`; `canonicalUrl` is optional; images in HTML are side-loaded, and there is a separate upload endpoint [6].
- What breaks: I found no authoritative statement on how Medium renders Markdown footnotes or tables. Treat both as likely to fail; unverified.
- So for Medium the realistic automation is the "Import a story" tool pointed at the live Substack URL, which sets the canonical [4], or manual paste with the canonical set by hand.

### 2.3 Other routes

- Pandoc converts Markdown (including footnotes and tables) to HTML; useful for previewing and as an input to copy-paste. No source was fetched for this; it is common knowledge, flagged as unsourced.
- dev.to supports `canonical_url` in front matter [5]; Hashnode and dev.to have real APIs and are the only syndication targets here with documented write APIs (not independently verified in this report; the `article` package's adapters for them are also stubs [7]).
- Aggregator tools claim to publish to dev.to, Medium, Hashnode and Substack automatically [22]. I did not evaluate them, and any that post to Medium must rely on a pre-2025 token or the browser.
- Ghost or a self-hosted blog would give full control of the canonical URL, footnotes, tables and an API, at the cost of Substack's network effects. Out of scope here; mentioned as the long-term alternative.

## 3. The `article` package (thorwhalen/article)

Repository remote: github.com/thorwhalen/article [7]. Inspected at the main branch, version 0.1.3 per the latest commit message.

- Design: an article JSON with `content_markdown`, tags and a `platforms` map; two commands, `publish-primary` (Substack, records the live URL as the SSOT in a state file keyed by slug) then `syndicate-secondary` (Medium, dev.to, Hashnode), injecting the canonical URL. Registry of async adapters, pydantic config, state store [7].
- Substack adapter: a stub. It only builds the URL `<publication>/p/<slug>` and returns a success result flagged `stub: True`; the Playwright login and editor flow exists only as a commented TODO. No network call, no draft creation, no image upload, no footnote handling. It reports "published" or "draft" status as a label only, so a success result from it means nothing about Substack [7].
- Medium adapter: a stub. It builds a correct payload for the old API (`contentFormat: markdown`, `canonicalUrl`, first three tags, `publishStatus` draft or public) and returns it in the result without sending it; the httpx call is commented out. The docstring itself notes the API is deprecated and that no new tokens are issued [7].
- dev.to and Hashnode adapters: also stubs with commented TODOs [7].
- Auth: planned only (Substack email/password or saved Playwright storage state; Medium bearer token); neither is exercised [7].
- Images, footnotes, captions: not handled anywhere. The only image-related field is `cover_image_url` in the config for a secondary platform; the article body is passed through as a Markdown string unchanged [7].
- Tests: ran the suite offline (`pytest`, 27 tests passed). The tests cover config loading, the state store, registry, the CLI wiring and the canonical handoff logic against the stubs; none exercises a real platform [7].
- Verdict: the orchestration layer is sound and tested, but nothing publishes. Missing for this essay: a real Substack adapter (the cleanest fill-in is to call `python-substack`'s draft-from-Markdown path instead of Playwright, which gives local-image upload, captions and footnotes for free [8]), an explicit rule that the Substack adapter creates drafts only, a table pre-processing step (table to image or list), and for Medium a switch from the API adapter to a "print the Import-a-story steps and canonical URL" adapter. Also worth deciding: whether `publish_as_draft` should default to true for the primary.

## 4. Recommended workflow for this essay

Assumes the essay is a Markdown file with relative image paths and a footnote written as `[^1]` definitions.

1. Preprocess (automatable): replace any GFM table with an image or a list (the only construct known not to convert) [8]. Put image captions in the Markdown title slot: `![alt](images/fig1.png "Caption text")` [8]. Convert the DFW-style footnote to `[^1]` form if it is not already. If the essay lives as a GitHub issue body, export it to a local file first (this repo's issue-to-file step is not covered by any tool inspected here).
2. Authenticate once (manual): log in to Substack in a browser and give `python-substack` the cookies (file or string); the project page recommends cookie auth when captcha or magic-link sign-in is used [8]. Keep the cookies out of the repo.
3. Create the draft (automatable): `substack drafts create essay.md` with `--json`; it uploads local images and creates an unpublished draft [8]. Re-run `substack drafts update <id> essay.md --dry-run` then with `--yes` after edits [8]. Note that I have not run this; check the installed version's flags.
4. Review in the Substack editor (manual): check footnotes (including phone display, which has been reported to clip footnotes [14]), the code block language, image captions, LaTeX rendering, the subtitle, cover image, tags, section and the email/web settings. Add embeds and any table image by hand, since these have no Markdown equivalent [8].
5. Publish (manual, one click) in Substack. Only draft creation is automated; this is deliberate.
6. Optional Medium copy (manual, a few minutes), after one to two days [5]: Stories page, "Import a story", paste the live Substack URL; the canonical is applied automatically [4]. Then inspect the imported story for footnotes and tables (unverified how they come through), fix by hand, add up to five tags in the Medium UI, and check page source for the canonical tag [4][5]. If the import fails on the page, paste the content and set the canonical by hand [4].
7. Record the Substack and Medium URLs in the Discussion that the essay moves to, as the existing Discussions do [1].

Which steps stay manual: authentication, editor review, publishing, all embeds, tables, and the Medium import (no supported API). A thin script wrapping steps 1 and 3 is enough; the `article` package's registry and SSOT state are only worth keeping if a second real target (dev.to or Hashnode) is added.

## Unverified or open items

- Current `python-substack` version and exact flag names (not run).
- Substack rich-paste behaviour for footnotes from rendered HTML.
- How Medium's importer treats footnotes, tables, LaTeX and code language.
- Whether this author's Medium account still shows Integration tokens.
- Official Substack statements on canonical URL handling, Recommendations and Notes effectiveness, and current fee schedule.
- Medium table support and current publication/boost routes after Better Programming's hiatus.

## REFERENCES

1. [thorwhalen/essays Discussions (queried via GitHub GraphQL API, 2026-10-08)](https://github.com/thorwhalen/essays/discussions)
2. [Example Discussion linking a Substack post: The Internet After Browsing](https://github.com/thorwhalen/essays/discussions/24)
3. [Substack vs Medium: Which Is Better for Writers in 2026? (Amy Suto)](https://www.amysuto.com/desk-of-amy-suto/substack-vs-medium) and [How to Grow a Substack to 10,000 Subscribers in 2026 (unilink, vendor blog)](https://www.unilink.us/blog/how-to-grow-a-substack-2026)
4. [Medium Help: Importing a post to Medium](https://help.medium.com/hc/en-us/articles/214550207-Import-post)
5. [Can You Post the Same Blog on Substack and Medium? (Anshul Kumar)](https://anshulkumar.substack.com/p/can-you-post-the-same-blog-on-substack) (community-reported behaviour, not official)
6. [Medium/medium-api-docs (archived 2023-03-02)](https://github.com/Medium/medium-api-docs)
7. [thorwhalen/article (README, adapters, tests; inspected locally and run offline)](https://github.com/thorwhalen/article)
8. [ma2za/python-substack (README and docs/markdown.md)](https://github.com/ma2za/python-substack)
9. [Substack vs Medium in 2026 (unilink)](https://www.unilink.us/blog/substack-vs-medium-2026)
10. [Farewell, Better Programming (Medium, 2023-11-10)](https://betterprogramming.pub/farewell-better-programming-395b0cf966ad)
11. [Substack Pricing: What It Costs and How Much Substack Takes (Amy Suto, third party)](https://www.amysuto.com/desk-of-amy-suto/substack-pricing)
12. [Medium Help: Partner Program earnings calculation](https://help.medium.com/hc/en-us/articles/360036691193)
13. [Medium Help: Set a canonical link](https://help.medium.com/hc/en-us/articles/360033930293-Set-a-canonical-link)
14. [Substack user threads on footnotes (comments on "Are footnotes necessary?")](https://missiongenealogy.substack.com/p/are-footnotes-necessary/comments)
15. [Citations and footnotes on Medium (ReadMedium mirror, secondary)](https://readmedium.com/citations-and-footnotes-on-medium-3713cc665722)
16. [New on Substack: code blocks and other updates (on.substack.com)](https://on.substack.com/p/new-on-substack-draft-notes-hide)
17. [LaTeX Tables in Substack, Step by Step](https://sipandsum.substack.com/p/substack-latex-table) and [comments on "How to insert a table in Substack"](https://nsokolsky.substack.com/p/how-to-insert-a-table-in-substack/comments)
18. [python-substack on PyPI](https://pypi.org/project/python-substack/)
19. [Substack Help: How do I import my posts from another platform, such as Mailchimp, WordPress, Medium, or Ghost?](https://support.substack.com/hc/en-us/articles/360037830351-How-do-I-import-my-posts-from-another-platform-such-as-Mailchimp-WordPress-Medium-or-Ghost-)
20. [StackOne: Medium integration token guide (secondary)](https://docs.stackone.com/connectors/medium/guides/link-account/integration-token)
21. [n8n docs: Medium credentials](https://docs.n8n.io/integrations/builtin/credentials/medium/)
22. [Publish to Dev.to, Medium, Hashnode, and Substack Automatically (startuphub.ai, unevaluated)](https://www.startuphub.ai/ai-news/ai-tools/2026/publish-to-devto-medium-hashnode-substack-automatically)
