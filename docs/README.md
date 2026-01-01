I want to design an sdk for a new type of design for deep learning architectures. My idea is this:
* basically, we dont need to train models from scratch; there are lots of models on huggingface that do all sorts of stuff
* however, for difficult tasks we would typically want to combine models. This is sort of designing architectures at a higher level of abstraction. Think more: I use a textvectorizer, combine it with a vectorizer that checks for regexes and outputs a vector of 0/1s, concatenate those, slap on a small two layer neural network.
* The focus is more on combining at the level of high-dimensional vectorspaces and recombining that.

Eg i have 20.000 texts that need to be classified in 160 "lawful facts". Texts are anywhere up to 40k tokens long. So i could just run an embedder, and slap a nn onto that. But probably it is suboptimal. So, why not enhance it with eg regexes for laws, or tfidf vectorizer. Then maybe combine it with a fast LLM like haiku for some details... 

So, if you get my drift; i am thinking about:

a) having the ability to download huggingface models, and build in the GPU acceleration detection etc
b) make sure it is clear how to connect models at the levels of tensors. So it might be important to implement classes like 1DTensor, 2Dtensor, 3dtensor etc such that vectors can be combined better.
c) think of a class of connectors (eg concatenate, but maybe also small NN architectures to "merge" two flows)
d) essentially instead of doing nn.sequential(nn.linear, nn.relu, ...) we want to build something like this but on this higher level. Also, can we wrap things like skipconnections, parallelconnections, etc?
e) integrate more basic vectorizers, eg like regex-based, or tfidf, etc

essentially, i think we often dont need to TRAIN the HFmodels. So, probably, what is much smarter, is to a) set up models b) run the data trough the pipeline one time, and cache all vectors. 

After this, we have a cached version of text -> vectors, for all variations (regex, tfidf, vectors, etc) and then we can start figuring out how to combine this information (eg stack and run a cnn, concatenate, transformer, etc)

- the amount of types of models of huggingface is huge. I would indeed like to simplify this; i just need text -> vector. So, ideally, i just want embedders, however, sometimes models are finetuned eg on specific corpora (eg legal text) so i would want those. Ideally, i would like the people not to have to think about the models and the types. Ideally, we would have a test-first approach, where we can eg test a range of models, and that we can have strict expectations (eg what you summed up about the types of output, with pooling etc). Also, isnt there a huggingface MCP to query for modeltypes?
- In addition to this, we would want to have different ways to extract low-level features, eg regex etc. Maybe we can setup an open-closed architecture, that is optimized for speed, where people can just add their own regexes. Ideally i would have groups of regexes that generate vectors (eg typically, there is something like "art 3.12 Burg. Wetb" and then we should extract this as a "hit" (a 1 in the vector) at 3.12, but for a range of articles (eg 3.13 would be another hit, etc... but maybe we have to be carefull about not exploding the regexes...)
- One major part would be: how to build caches from the different models . Then, another major part would be: how to "play" with these vectors, here it probably looks more like "classic" pytorch modelling, but we can still help with "connectors" eg skiplayer, parallel, etc

Can you help me plan this out? And make a specs-driven document? Can you spot pitfalls, potential difficulties, etc?

Huggingface has an mcp, see https://huggingface.co/learn/mcp-course/unit1/hf-mcp-server but i also installed the mcp server in this session.


