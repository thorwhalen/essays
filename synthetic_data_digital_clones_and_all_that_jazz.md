# Synthetic data

Hype!

Fake news!

Informational illusion. Beware!

Okay, that was the troll style way of compensating for the currently trending hype around the subject, and all the misinformation surrounding it. 
Reality, as always, is a bit more in the grey. (But if you want black and white commandments, here it is: Stay away from synthetic data.)

## The good

Synthetic data is good if you count sythetic data as being data you generated from the raw data. That's data preperation. That's deidentification. 
That's not what I'm really focused on here. 

Synthetic data to balance and unbalanced dataset. You got too many of what category and not enough of another? 
Repeat and remove data until you get the right balance. Repeat and remove, or do something smarter, obviously.

Your model is unstable because it has some holes where you should have somme continuous blobs? 
Well, fill the holes! And if you were right, your model will improve. What does improve mean though? 
Better accuracy? On what? You're training on real_data + synthetic_data, right? What are you testing on?
And how does your model generalize? How does it do when you feed it REAL new points?

If it generalizes well, it's that you filled the holes correctly, or that you modified the category balance correctly. 
Congradulations? The model you're using to general that new data fits reality well!

Erm... So why don't you just use **that** model?

## The bad
