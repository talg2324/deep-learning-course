# Exercise 2
# Preprocessing
Penn Tree Bank dataset tokenization - to create both a dictionary and data batches for training / evaluation.

# Models Architecture - RNN model class
The RNN class constructor expects an enumerable recurrent unit type (RecurrentUnit - either LSTM or GRU). 
as well it expects the dropout regularization rate for the recurrent unit, layers dimensions, and weight init value to set the uniform distribution.


# Running the code
## End-to-end
To run the entire code *including* training, run all the cells in series.\

## Evaluation only
To exclude training, pre-trained models can be loaded from the '/saved_models/' folder.\
all sections apart from "Training & Eval - 4 models" must be run prior to evaluation to:
- import the required libraries
- declare the RNN class
- set hyperparams
- load the data
- define the evaluation loop

Once these cells have been run, skip to "Saved Models - Evaluation" and declare the inference evaluation functions.
Finally, models from '/saved_models/' can be loaded and evaluated.