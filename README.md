# Mechanistic Interpretability Paper Replications
These are replications of some results from prominent papers in mechanistic interpretability, as well as some other mechanistic interpretability exercises

## Paper Replications
### Indirect Object Identification
This replicates some of the results in the paper [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593) by Wang et al

### Grokking and Modular Arithmetic
This completes some of the exercises from from Nanda's [A Mechanistic Interpretability Analysis of Grokking](https://colab.research.google.com/drive/1F6_1_cWXE5M7WocUcpQWp3v8z4b1jL20) and thereby replicates some of the work in [Progress measures for grokking via mechanistic interpretability](https://arxiv.org/abs/2301.05217)

### Othello GPT
This replicates some of the results in the paper [Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task](https://arxiv.org/abs/2210.13382) by Li et al\
It utilizes some work directly from their accompanying library [Othello World](https://github.com/likenneth/othello_world)

## Other Work
### Balanced Bracket Classifier
This reverse engineers one of the circuits a simple transformer uses for bracket classification, or taking a string like "(())()" and predicting whether the parentheses are balanced or not. 

### Basic Transformer Implementation
This creates a basic transformer from scratch

## Credits
These were completed during the summer of 2023 following the [ARENA](https://www.arena.education/) curriculum. I completed the exercises by implementing the functions within the main.py files. The skeletons/outlines of the main.py scripts and the supplementary files are the work of the ARENA team and not mine. \
These exercises rely heavily on the [TransformerLens](https://github.com/neelnanda-io/TransformerLens) library
