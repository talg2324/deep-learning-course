# Exercise 1
# Training With Regularization
The LeNet5 class constructor expects an enumerable regularization type (RegType).\
By default, no regularization is used (RegType.NONE), but Dropout or Batch Normalization can be requested and the constructor will add the necessary layers accordingly.

In the case of weight decay, no changes are made to the network architecture, but the weight decay parameter must be manually initialized in the optimizer constructor (it defaults to 0).

# Running the code
## End-to-end
To run the entire code *including* training, run all the cells in series.\

## Evaluation only
To exclude training, pre-trained models can be loaded from the '/saved_models/' folder.\
The first four cells of the notebook must be run prior to evaluation to:
- import the required libraries
- declare the LeNet5 class
- initiate the dataloader and download the dataset if necessary

Once these cells have been run, skip to "Testing Saved Models" and declare the inference evaluation functions.
Finally, models from '/saved_models/' can be loaded and evaluated. The final four cells of the notebook show how to do this and can be evaluated without training.