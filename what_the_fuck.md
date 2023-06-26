# What The Fuck - my understanding of this library
## Models dir
Defines the decision transformer models.
There's some boring LSTM stuff that I don't care about.

Some shared stuff is in `TrajectoryTransformer`, the actual model seems to be <span style="color: red">`DecisionTransformer`</span>.
<span style="color: red">I don't know what `CloneTransformer` does,</span> but both the actor and the critic inherit from it.
<span style="color: red;">There's also a ton of embedding stuff that I should probably grok more.</span>

## Environments
Sets up the Memory env and some utils around it

## Decision_transformer dir
* Utils - some useful stuff for inintialize a model given a path to its weights
* Eval - run a bunch of trajectories and see how well the agent acts
* Runner - wrapper to run stuff, <span style="color: red">need to look into this some more</span>
* Calibration - <span style="color: red">IDK</span>

There's also some stuff to train the thing, I don't rly care about that rn

## Patch TransformerLens
The `patching` file has some cool utils for basic activation patching.

## PPO
Who cares ¯\\\_(ツ)\_/¯

## Call w Joseph 21 June
### Dead ends in the repo
there is a ton of code attempting to do online training with the transformers, that's kind of what the CloneTransformer is
### Embeddings
You have an Env with a bunch of stuff in the observation (7x7 grid = view size; usually one-hot encoded)

Dense representation would be Blue-Door-Open -> 3-5-1 (or something like that)

A better way to represent this is to have a one-hot vector with 20 input elements, eg if 1 at index N that means Blue. There's one of these for each element in the view size, so you have $7*7*20=980$ elements (this is all for observations)

For actions we just have the same d\_vocab as the number of actions

For RTG we just have size 1 (we're using RTG because it's a sparse-reward env, and it's like a type of promt telling you right at the beginning how well the trajectory went)

### Calibration
Take a model, do a ton of rollouts, measure RTG vs achieved reward

### Forcing a certain spawn setup
Line 153 in memory.py, you can just override that with your own argument

### Unification
You can make the agent turn around or put it in some weird position and it'll still aim for the same goal - can we find a single vector that predicts behavior in all of these settings?