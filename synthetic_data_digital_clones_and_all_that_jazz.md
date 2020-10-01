# Synthetic data

Hype!

Fake news!

Informational illusion. Beware!

Okay, that was the troll style way of compensating for the currently trending hype around the subject, 
and all the misinformation surrounding it. 
Reality, as always, is a bit more in the grey. (But if you want black and white commandments, here it is: 
Stay away from synthetic data.)

## Nutshell

Synthetic data always has a model/distribution assumption behind it. 
It's **that information** you're actually trying to enhance your model with. 
The only reason you're generating points is that your model can't (shame on it), 
or you don't know how to make it (shame on you), 
ingest any another form of information.

## Use cases

Synthetic data is good if you count synthetic data as being data you generated from the raw data. 
That's data preparation. That's de-identification. 
That's not what I'm really focused on here. 

Synthetic data to balance and unbalanced dataset. You got too many of what category and not enough of another? 
Repeat and remove data until you get the right balance. Repeat and remove, or do something smarter, obviously.

Your model is unstable because it has some holes where you should have somme continuous blobs? 
Well, fill the holes! And if you were right, your model will improve. What does improve mean though? 
Better accuracy? On what? You're training on real_data + synthetic_data, right? What are you testing on?
And how does your model generalize? How does it do when you feed it REAL new points?

If it generalizes well, it's that you filled the holes correctly, or that you modified the category balance correctly. 
Congratulations? The model you're using to general that new data fits reality well!

Erm... So why don't you just use **that** model?

## What's happening?

So that's the main point. 

Synthetic data always has a model behind it. 
It may be simple, such as repeating and removing points, or it may be more sophisticated, 
such as adding noise, or extrapolating between points, computing categorical densities 
and randomly drawing from the inferred distribution.

Still. A model.

So why not just use **that** model?

Well, there lies the rub, and a reason why resorting to synthetic data **could** be a good idea. The reason is that 
- it may be easy to build a generative model, but harder to reverse it (i.e. get a inference engine out of it 
-- but note that there are methods for this)
- the inference model you're building (the purpose of the data in the first place) has a limited interface/language

Limited interface/language? Like... it only takes points as it's data input format. 
Like... most of the models out there.
If your model (learner/fitter) could ingest weighted points, 
it would be ludicrous (and inefficient) to repeat and remove points.
If your model could inject distributions, you wouldn't need to generate extra points drawn from that distribution: 
You'd always get better (and quicker) by feeding the full continuous distribution than a few points drawn from it. 

And that's it. Anytime you use synthetic data, there's some data distribution assumption behind it, 
and the only reason you'd want to draw points from it is because you (or your model) 
doesn't have the language/interface to absorb the distribution in it's entirety. 

## Should I do it?

Sure. Do it. But know what you're doing, and especially know what you're not. 

You are doing it to add information that's not in the data already, 
and you're doing it that way because you have no other way to communicate this information to your model. 

It's not your fault. It's the model's fault. Your only fault is not to know more about Bayesian methods. 

Do ask yourself the independence question too. That information you're supposedly adding. 
Is it independent of your observation of the data? 
If it is, go forth and multiply. If not, you may want to factor out any dependencies you can, because... well you know! 
You may unwillingly be exaggerating some aspects of the data ("double counting") to the expense of others. 

# Conclusion

Creating more points is just a hack, a proxy for something better. 
The ideal: Merge the idea/logic/model behind the data generator to the model. Directly.

Most data science methods work with points only.

If they can ingest weighted points, that's better since you can add more information through these.

Even better would be distributions (having both shape and weight). But that might require some math, 
so might not be as popular as point-injecting multi-layer black-box gluttons. 


