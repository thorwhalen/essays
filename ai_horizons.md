In tech, there’s way too much focus on the crisp (precision, predictability, exactitude,…) that blinds us from the being able to take advantage of the mush (uncertainty, fuzziness, imprecision, …). In many (but not all) AI approaches, there’s a lot of wiggle room created by so many layers of imprecision between signal and decision. An error between a key stroke and key captured could be fatal for a text editor. 

But in a signal-to-decision, only the last step (the decision) needs to be crisp – on the path there, most errors would lead to the same decision. 

# Ideas to consider

In case the reader doesn't go further, I'd like to bullet point a few ideas at the top.
* Coding: We're shaping our minds according to machines instead of shaping machines to our minds.
* When can coding, or even computers, do the job without bring precise, complete, or event deterministic? Because... we (humans) aren't any of that, and yet are good enough for many tasks.
* Is it worse to not be completely precise or predictable, or just fail categorically? How does this depend on the context?
* Could AI reading comprehension research and the troves of public annotated code help to loosen the rigidity of coding?
* Can the natural semantic emergence (deep learning) be guided towards mind-patterns rather than brain-patterns?
* Self-recovering code
* Self-organizing code

More details on some of these ideas will be discussed below.

## The seldom challenged

Let's consider a radical idea. The three main limitations of the evolution of AI are
* Deep learning
* Coding
* Digital Computers

The above follows the "bad to worse" type of joke structure. 

The punchline is I'm not (totally) kidding. 

Now, of course, I wouldn't be writing this, or have the wonderful job I have, if it weren't for digital computers and the programming languages I use to talk to them. 
And deep learning is the (current) cherry on top of the machine learning cake: Neural networks can do magic.

Yet, we can do better. Of course we can.

So let's have a look at what we could do differently. Let's for a moment challenge precisely those things that seem so unchallengeable.

### Deep learning
Deep learning is greedy, brittle, opaque, and shallow, says Gary Marcus (professor of cognitive psychology at NYU). 

Think outside the hypercube!

We're trying to model the mind by modeling the brain, and modeling the brain with structured layers.
The mind has more of the messy multi-connected structure of language than the neat arrangements of matrices.

### Coding
"_I’m not sure that programming has to exist at all._" -- Bret Victor

I said "coding" instead of "programming languages" on purpose. It's the "programming" that I'm pointing the accusatory finger at, 
not the language part. 
Again, language is close to the mind, and will be involved in our communicating our designs to a computer. 
But does that communication have to be so darn unforgivingly dry and fragile?

The way we talk to computers still carries a heavy "engineering" legacy; switches to flip, wires to connect, etc.
But that's not the way we think, so the idea-to-implementation path is slow and creativity is limited.

### Digital Computers
Intelligence is not binary.

Reality is not precise.

Harness uncertainty instead of fighting it!


# Some ideas to explore

## AI Reading comprehension 

What could we utilize from coding forums and AI reading comprehension to loosen the rigidity of programming languages?

Reading comprehension is a branch of _cognitive AI_ that seeks to teach machines how to "understand" what they read. This "understanding" is defined according to specific actions the machine can perform, given the input text, such as answer questions about it. 

Wouldn't it be conceivable then, to have a machine "understand" a written description of a app to be able to write the code for it?

Lot's of public code produced every day, a lot of it even "annotated" by natural language (for instance on coding forums). Is that not enough supervised data to help us out?

The goal doesn't have to be absolute to make things better than they are now. We don't have to produce a program that generates the perfect app from an unstructured natural language description. But we can probably do better than needing to spell everything out in some frigid programming language. The bar can be placed wherever today's pragmatic AI possibilities dictate. 


## Guide the natural semantic emergence

_"(…) deep learning dynamics can self-organize emergent hidden representations in a manner that recapitulates many empirical phenomena in human development."_ [A mathematical theory of semantic development in deep neural networks](https://www.pnas.org/content/116/23/11537): 

Yet, I still insist that the industry should spend more effort modeling the mind with language-like structures and is spending too much on modeling the brain with rectangular structures.

Let's assume, as the article claims, that language-like organization naturally emerges from the rectangular ones.My guess is that there's many ways to do this, and therefore it will be improbable that the internal representations the machine will develop will be compatible with ours.

Perhaps machines don't need any guidance at all, but if we want the human+machine system to be optimal for humans, we may want to provide this guidance anyway.

One way to do this is to constrain the space of possibilities to resemble to something resembling our minds (rather than our brains).

## Self-organizing code

"_(A programmer...) has to be able to think in terms of conceptual hierarchies that are much deeper than a single mind ever needed to face before._" -- Edsger Dijkstra (1988 - computer scientist)

It's about writing components in such a way that the structure (think DAG) of the program itself can be inferred by the (unordered) set of components itself, with minimized "further questions" to dispel ambiguity. Totally and absolutely possible: You have types, names, tests, and as a last resort, the user, to help reduce any ambiguities.

By projecting language constructs to a common space — for instance, by wrapping their interfaces to a universal one — we make it easier for various components to be connected into a computational structure. 

But how can we make connecting components easier? Identifying patterns and erecting a reusable components out of the composition of other components brings us some of the way — and this is how computation has evolved so far; chunking and aliasing. 

What’s outside that box?

Thought experiment: Leave a developer with a bunch of code she didn’t write, and ask her to build stuff with it (with minimal glue code). How would she approach the problem?

The code itself, along with all it’s meta-data, has a lot of information about how one could, or could not, connect things together. We have documentation, tests, names given to functions and arguments, types, etc. If this information could be extracted and codified properly, it would significantly reduce the entropy or structural possibilities. 

But extracting and codifying this information is hard when you have so many languages, styles, choice of words, etc. That said, if you provide tools for a human to add to guide the system, there may be hope. 

R&D POC: Write components with strongly typed interfaces, throw them in a pool with no further instructions, and get a generator of DAGs that connect these components in a input-output type-compliant manner. Generator would start with the smallest DAGs, and build from there.

But types are too restrictive; their history is tainted with too-close-to-the-machine concerns. We need a more flexible concept to replace type. Duck-typing is one direction, but might be too flexible. Perhaps a more fuzzy approach is needed. Rather than matching types, we associate a scaled matching score to any two components (or any two sets of components). Probabilistic approaches would then guide the search for compliant structures. 


# Quotes:

“_The problem is that software engineers don’t understand the problem they’re trying to solve, and don’t care to_" -- Nancy Leveson (MIT software-safety expert)

"_(A programmer...) has to be able to think in terms of conceptual hierarchies that are much deeper than a single mind ever needed to face before._" -- Edsger Dijkstra (1988 - computer scientist)

"_I’m not sure that programming has to exist at all._" -- Bret Victor

"_Computers had doubled in power every 18 months for the last 40 years. Why hadn’t programming changed?_"
-- https://www.theatlantic.com/technology/archive/2017/09/saving-the-world-from-code/540393/

"Most programmers like code. At least they understand it."
-- https://www.theatlantic.com/technology/archive/2017/09/saving-the-world-from-code/540393/



 
 
