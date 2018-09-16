# Model hyperparameters

* epoch
* learningRate
* learningRateUpdateRate
* wordNgrams

# Model parameters

* wordVecDim = length of vector used to represent each word
* windowSize = length of context around word
* loss = softmax or hierarchical softmax or negative sampling

## Dictionary parameters

* bucket = size of hash table (2M default)
* minn, maxn = compute all character n-grams between this range
* minCount = subsampling threshold. Ignore words less than this count
* minCountLabel = subsampling threshold. Ignore labels less than this count
* samplingThreshold = used while computing discard probability of frequent words

# Model Attributes

1. inputMatrix = Matrix [numWords + bucket] X [wordVecDim]
2. outputMatrix = Matrix [numLabels] X [wordVecDim]
3. hiddenVector = Vector[wordVecDim]
4. outputVector = Vector[wordVecDim]
5. gradient 


# Supervised model defaults

* character n-grams disabled
* loss is softmax
* learning rate set to 0.1
* minCount


# Supervised model training 

```
  read labels + words from file and insert into dictionary
	remove words and labels whose count less than "args.minCount" and "args.minCountLabel"
	for each word/label
	  set Discard Probability such that rare words are discarded
	compute N-grams for each word based on "args.minn and args.maxn"

	if (pretrained vectors available)
	  add them to Dictionary
		set Model.inputMatrix = list of pretrained vecors

	start training threads
	in each training thread

	  initialize hierarchical softmax/negative sampling data based on label counts

    for each line
		  for each word
			  get vector from dictionary
		Model.hiddenVector = average of vectors of each word in line

		*assuming here that loss is set to softmax*
		Model.outputVector = dot product of (outputMatrix . hidden)
		compute softmax over outputVector

		for each row in Model.outputMatrix
			gradient = gradient + derivative of loss * row 
			update weights of outputMatrix row by adding hiddenVec 

    Model.loss = -log(Model.outputVector[targetLabel])

		for each word on input line
			adjust input matrix by adding gradient to Model.inputMatrix[word] 
```

# Supervised model prediction

```
  for each word in line
	  word_vector = dictionary[word]
	hidden vector = average of vectors of all words in line

  Model.outputVector = dot product of (outputMatrix . hidden)
  compute softmax over outputVector

	return top N labels in Model.outputVector above threshold

```

# TODO

1. add hierarchical softmax & negative sampling
2. note all model param
3. quantization
4. wordNgrams

