# Alphabet Soup Charity Funding Predictor

Usage
Google CoLab was used instead of Jupyter Notebook
All the Google CoLab files are located in the HDF5 files

# Overview:
This analysis aims to develop a binary classifier for predicting the success of funding applications submitted to Alphabet Soup, a nonprofit foundation, utilizing a dataset containing over 34,000 past applications. The project entails data preprocessing, designing and training a neural network model, and optimizing it to achieve a predictive accuracy exceeding 75%. Despite various optimization efforts, the desired accuracy threshold was not met.

# Results
Data Preprocessing
The model aims to predict the success of applicants if they receive funding. This is indicated by the IS_SUCCESSFUL column in the dataset which is the target variable of the model. The feature variables are every column other than the target variable and the non-relevant variables such as EIN and names. The features capture relevant information about the data and can be used in predicting the target variables, the non-relevant variables that are neither targets nor features will be dropped from the dataset to avoid potential noise that might confuse the model.
During preprocessing, I implemented binning/bucketing for rare occurrences in the APPLICATION_TYPE and CLASSIFICATION columns. Subsequently, I transformed categorical data into numeric data using the one-hot encoding technique. I split the data into separate sets for features and targets, as well as for training and testing. Lastly, I scaled the data to ensure uniformity in the data distribution.

Compiling, Training, and Evaluating the Model
Initial Model: For my initial model, I decided to include 3 layers: an input layer with 80 neurons, a second layer with 30 neurons, and an output layer with 1 neuron. I made this choice to ensure that the total number of neurons in the model was between 2-3 times the number of input features. In this case, 43 input features were remaining after removing 2 irrelevant ones. I selected the relu activation function for the first and second layers, and the sigmoid activation function for the output layer since the goal was binary classification. To start, I trained the model for 100 epochs and achieved an accuracy score of approximately 74% for the training data and 72.9% for the testing data. There was no apparent indication of overfitting or underfitting.

Optimization attempts
I attempted to optimize the model’s performance by first modifying the architecture of the model by adding 2 dropout layers with a rate of 0.5 to enhance generalization and changing the activation function to tanh in the input and hidden layer. With that, I got an accuracy score of 74.1% for my training set and 72.9% for my testing set.
For my second optimization attempt, I used hyperparameter tuning. During this process, Keras identified the optimal hyperparameters, which include using the tanh activation function and setting 41 neurons for the first layer. As a result, the model achieved an accuracy score of 73.3%.

# Summary:
While the neural network model exhibits potential with a maximum accuracy of approximately 73%, it falls short of the targeted 75% accuracy.

# Recommendation for Future Work:
Future iterations may explore alternative machine learning models like Gradient Boosting or Random Forest classifiers, known for their adeptness in handling both categorical and numerical data, potentially capturing complex feature interactions more effectively. Additionally, further data preprocessing and feature engineering could be pursued to enhance predictive accuracy.

