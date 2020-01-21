
# Hurdles to AI

At the time of writing this, I see three main limitations of the evolution of AI:
* Deep learning
* Coding
* Digital Computers

The above follows the "bad to worse" type of joke structure. 

The punchline is I'm not kidding. 

Now, of course, I wouldn't be writing this, or have the wonderful job I have, if it weren't for digital computers and the languages to talk to them. 
And deep learning, the (current) cherry on top of the machine learning cake: They can do magic.

But we can do better. Of course.

So let's have a look at what we could do differently.

## Deep learning
Deep learning is greedy, brittle, opaque, and shallow, says Gary Marcus (professor of cognitive psychology at NYU). 

Think outside the hypercube!

We're trying to model the mind by modeling the brain, and modeling the brain with structured layers.
The mind has more of the messy multi-connected structure of language than the neat arrangements of matrices.

## Coding
I said "coding" instead of "programming languages" on purpose. It's the "programming" that I'm pointing my accusatory finger at, 
not the language part. 
Again, language is close to the mind, and will be involved in our communicating our designs to a computer. 
But does that communication have to be so darn unforgivingly dry and fragile?

The way we talk to computers still carries a heavy "engineering" legacy; switches to flip, wires to connect, etc.
But that's not the way we think, so the idea-to-implementation path is slow and creativity is limited.

## Digital Computers
Intelligence is not binary.

Reality is not precise.

Harness uncertainty instead of fighting it!


# Guide the natural semantic emergence

_"(…) deep learning dynamics can self-organize emergent hidden representations in a manner that recapitulates many empirical phenomena in human development."_ [A mathematical theory of semantic development in deep neural networks](https://www.pnas.org/content/116/23/11537): 

Yet, I still insist that the industry should spend more effort modeling the mind with language-like structures and is spending too much on modeling the brain with rectangular structures.

Let's assume, as the article claims, that language-like organization naturally emerges from the rectangular ones.My guess is that there's many ways to do this, and therefore it will be improbable that the internal representations the machine will develop will be compatible with ours.

Perhaps machines don't need any guidance at all, but if we want the human+machine system to be optimal for humans, we may want to provide this guidance anyway.

One way to do this is to constrain the space of possibilities to resemble to something resembling our minds (rather than our brains).

# Self-organizing code

It's about writing components in such a way that the structure (think DAG) of the program itself can be inferred by the (unordered) set of components itself, with minimized "further questions" to dispel ambiguity. Totally and absolutely possible: You have types, names, tests, and as a last resort, the user, to help reduce any ambiguities.

By projecting language constructs to a common space — for instance, by wrapping their interfaces to a universal one — we make it easier for various components to be connected into a computational structure. 

But how can we make connecting components easier? Identifying patterns and erecting a reusable components out of the composition of other components brings us some of the way — and this is how computation has evolved so far; chunking and aliasing. 

What’s outside that box?

Thought experiment: Leave a developer with a bunch of code she didn’t write, and ask her to build stuff with it (with minimal glue code). How would she approach the problem?

The code itself, along with all it’s meta-data, has a lot of information about how one could, or could not, connect things together. We have documentation, tests, names given to functions and arguments, types, etc. If this information could be extracted and codified properly, it would significantly reduce the entropy or structural possibilities. 

But extracting and codifying this information is hard when you have so many languages, styles, choice of words, etc. That said, if you provide tools for a human to add to guide the system, there may be hope. 

R&D POC: Write components with strongly typed interfaces, throw them in a pool with no further instructions, and get a generator of DAGs that connect these components in a input-output type-compliant manner. Generator would start with the smallest DAGs, and build from there.

But types are too restrictive; their history is tainted with too-close-to-the-machine concerns. We need a more flexible concept to replace type. Duck-typing is one direction, but might be too flexible. Perhaps a more fuzzy approach is needed. Rather than matching types, we associate a scaled matching score to any two components (or any two sets of components). Probabilistic approaches would then guide the search for compliant structures. 


