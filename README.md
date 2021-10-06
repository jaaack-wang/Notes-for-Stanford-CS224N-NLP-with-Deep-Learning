Notes for Stanford [CS224N: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/index.html), a great course that I just discovered. You can also find [the course videos on YouTube](https://youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z), which were recorded in Winter 2019 and contains 22 lecture videos. There are differences between the course slides found on the website (2021 version) and those used in the videos due to certain degrees of revisions. My notes are mainly based on the lecture videos, but also take the latest slides on the course website into accounts.


In each lecture folder, you will find the following materials: my notes, the lecture slide(s) and other course materials, such as course code or assignment. The assignment is left in a way as it is downloaded from the course website. In addition, I also placed the course notes written by Stanford students available on the course website in each lecture folder and denoted them by \[Stanford notes\] to distinguish from my notes, denoted as \[Jack's notes\]. I consciously made my notes as shareable as possible and I hope anyone who views this repository find them helpful.  


You are also welcome to take a loot at my another repository [dl-nlp-using-paddlenlp](https://github.com/jaaack-wang/dl-nlp-using-paddlenlp), which focuses the application of deep learning techniques in natural language processing using the state-of-the-art deep learning frameworks `paddle` and `paddlenlp`. <br>


The "Table of Contents" below are structures of my notes taken for lectures that I finished. 


# Table of Contents

## Lecture 1-Intro and Word Vectors

- 1. Casual takeaways
- 2. WordNet
    - 2.1 Quick facts
    - 2.2 Accessing WordNet
    - 2.3 Limitations
- 3. Word vectors
    - 3.1 One-hot vectors
    - 3.2 Word vectors (embeddings)
    - 3.3 Word2vec algorithm basics
        - 3.3.1 Basic ideas
        - 3.3.2 Math behind
- 4. References
- 5. Appendix: gradients for loss function (2)

## Lecture 2-Word Vectors and Word Senses

- 1. Casual takeaways
- 2. Word2vec
    - 2.1 Recap
    - 2.2 Two model variants
    - 2.3 The skip-gram model with negative sampling (HW2)
        - 2.3.1 Negative sampling
        - 2.3.2 Objective/loss functions
        - 2.3.3 Subsampling
- 3. Co-occurrence counts based method
    - 3.1 Co-occurrence matrix
    - 3.2 Dimensionality reduction
        - 3.2.1 Classic Method: Singular Value Decomposition
        - 3.2.2 Other tricks: scaling, pearson correlations etc.
    - 3.3 Count based method vs. direct prediction (neural network) method
- 4. GloVe: Global Vectors for Word Representation
    - 4.1 Basic idea
    - 4.2 Mathematical realization
- 5. Evaluation of word vectors
    - 5.1 Overview
    - 5.2 Intrinsic word vector evaluation
        - 5.2.1 Models
        - 5.2.2 Performances of different models
    - 5.3 Extrinsic word vector evaluation
    - 5.4 Factors that impact the performances
- 6. Challenges: word sense ambiguity
    - 6.1 Challenges
    - 6.2 Tentative solutions
- 7. References

## Lecture 3-Neural Networks

- 1. Casual takeaways
- 2. Classification review and notation
    - 2.1 Inputs and labels
    - 2.2 Output models
    - 2.3 Objective/loss function and cross entropy
    - 2.4 Traditional ML classifier versus Neural Network Classifier
- 3. Neural Network Basics
    - 3.1 Neuron
    - 3.2 What an artificial neuron can do
    - 3.3 Matrix notation for a layer
- 4. Named Entity Recognition (NER)
    - 4.1 NER task
    - 4.2 Simple NER training methods
        - 4.2.1 Binary classification with averaged context words vectors
        - 4.2.2 Window classification using multi-class softmax classifier
        - 4.2.3 Binary classification with unnormalized scores using shallow neural network
    - 4.3 Challenges for NER
- 5. Gradients computation
    - 5.1 Partial derivatives and gradients
        - 5.1.1 Simple case: placed in a vector
        - 5.1.2 A more complex case: placed in a matrix -- Jacobian Matrix
    - 5.2 Multivariate calculus
        - 5.2.1 The chain rule
        - 5.2.2 Partial derivatives computation in Jacobian Matrix
        - 5.2.3 Shape convention
- 6. References

## Lecture 4-Backpropagation

- 1. Casual takeaways
- 2. Derivative with regard to a weight matrix
    - 2.1 An shallow neural network example
    - 2.2 Let's derive
        - 2.2.1 With regard to W
        - 2.2.2 With regard to b
    - 2.3 Tips for deriving gradients
    - 2.4 Updating weights: gradient descent
    - 2.5 Transfer learning: using pre-trained parameters
- 3. Computation graph and backpropagation
    - 3.1 Overview
    - 3.2 An example
    - 3.3 Generalization of computation graph in neural networks
    - 3.4 Debugging
- 4. General tricks: regularization, nonlinearities, hyperparmeters etc.
    - 4.1 Regularization
    - 4.2 Vectorization
    - 4.3 Non-linearities: activation functions
    - 4.4 Parameter Initialization
    - 4.5 Optimizers
    - 4.6 Learning rate
- 5. References
- 6. Appendix
    - 6.1 Notations
    - 6.2 A quick and general solution
    - 6.3 Sigmoid function
    - 6.4 Softmax function

## Lecture 5-Dependency Parsing

- 1. Casual takeaways
- 2. Syntactic structure: constituency and dependency
    - 2.1 Constituency
    - 2.2 Dependency
    - 2.3 Sources of ambiguity related to dependency
        - 2.3.1 Prepositional phrase attachment ambiguity
        - 2.3.2 PP attachment ambiguities multiply
        - 2.3.3 Coordination scope ambiguity
        - 2.3.4 Adjectival modifier ambiguity
        - 2.3.5 Verb phrase (VP) attachment ambiguity
- 3. Dependency grammar and dependency structure
    - 3.1 A bit history of dependency grammar/parsing
    - 3.2 Dependency grammar and its graphic representation
    - 3.3 Universal dependencies treebanks and annotated data
- 4. Dependency parser
    - 4.1 Dependency conditioning preferences
    - 4.2 General parsing rules
    - 4.3 Projectivity
    - 4.4 Methods of dependency parsing
- 5. Greedy transition-based parsing
    - 5.1 Overview
    - 5.2 Arc-standard transition-based parser
    - 5.3 ML transition-based parser
    - 5.4 Evaluation of accuracy
    - 5.5 Problems
- 6. Neural dependency parser
    - 6.1 General idea
    - 6.2 Preprocessing
    - 6.3 Model architecture
    - 6.4 Comparison
- 7. References
