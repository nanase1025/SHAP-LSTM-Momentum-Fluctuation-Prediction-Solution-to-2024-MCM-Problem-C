# SHAP-LSTM-Momentum-Fluctuation-Prediction-Solution-to-2024-MCM-Problem-C
## 1. Description

###  1.1 Background

![image](https://github.com/nanase1025/SHAP-LSTM-Momentum-Fluctuation-Prediction-Solution-to-2024-MCM-Problem-C-/assets/145645749/4e6efb70-dbb0-4e27-bead-8bd4a421ddcd)

In the 2023 Wimbledon men's singles final, Carlos Alcaraz defeated Novak Djokovic, breaking his winning streak and dominance. The match was highly publicized as it represented the rise of a new star and the decline of a veteran.
  
In tennis, wins and losses are not only about point competition, but also about momentum swings. Momentum in sports refers to a player or team's sudden surge of success during a match, which can significantly impact the course and outcome of the game. However, accurately measuring and predicting momentum and its effects on the game has proven to be a challenge.

### 1.2 Restatement of the Problem

We need to model the following questions based on the data：

![image](https://github.com/nanase1025/SHAP-LSTM-Momentum-Fluctuation-Prediction-Solution-to-2024-MCM-Problem-C-/assets/145645749/8b251ff4-12e6-4d02-a69c-a929c9bb3f4f)

Problem 1: Create a model that captures changes in scores during a match, identifies which players perform better at key moments, and provides a model-based visualization of the flow of the match, given that the serve side wins more often in tennis.

Problem 2: Use a model/metrics to assess a tennis coach's suspicions about "momentum" in a match, i.e., whether match twists and player winning streaks occur randomly.

Problem 3: Develop a model to predict match transitions using data from at least one match and identify the most relevant factors.

Problem 4: Advise players on how to prepare for different opponents in response to changes in "momentum" in past matches. Test the accuracy of the model on other matches and identify factors that may need to be added.

Problem 5: Summarize the results of the study and provide coaches with a report, including a one- to two-page memo, on the role of Momentum and recommendations on how to prepare players for game flow events.

This repository is mainly about our solution to Problem 3.

## 2. SHAP-LSTM Momentum Fluctuation Prediction

To develop a model that utilizes data from at least one match to predict turning points in the game and to identify the most relevant factors, we constructed the SHAP-LSTM Momentum Fluctuation Prediction Model, as shown in Figure below.

![image](https://github.com/nanase1025/SHAP-LSTM-Momentum-Fluctuation-Prediction-Solution-to-2024-MCM-Problem-C-/assets/145645749/6a8b8d35-5050-4133-ba2e-ad23b38cc636)

### 2.1 Data Processing

This study utilizes the variables obtained from graph for model prediction. Before employing the model for prediction and evaluation, it is essential to normalize the data to prevent the model from not converging. Normalization is a crucial step in the preprocessing of machine learning, which helps accelerate the convergence of the model and can reduce the bias caused by different magnitudes of features. Using MinMaxScaler for normalization scales all features to a range between 0 and 1.

###  2.2 LSTM Prediction Model

#### 2.2.1 Model Introduction

Long Short-Term Memory networks (LSTMs) are a variant of Recurrent Neural Networks (RNNs) extensively used in deep learning for processing and predicting time series data as well as other sequential data. LSTMs were initially proposed by Hochreiter and Schmidhuber in 1997, aiming to address the vanishing and exploding gradients problems present in traditional RNNs.

The popularity of LSTMs stems from their ability to capture and remember dependencies in long sequences while preventing gradient issues. This is achieved through the introduction of three key gating mechanisms, including the forget gate, input gate, and output gate. Each gate is controlled by a sigmoid activation function, working together to determine which information should be retained, ignored, or output.

In this study, the LSTM model architecture for this task is designed with 512 hidden layers, each containing a large number of hidden units to capture complex patterns in the data. Additionally, it includes a linear layer to map the output of the LSTM layers to the dimensions of the prediction target. Such design aims to provide sufficient model capacity to learn and understand the complex relationships from input features to predicted momentum fluctuations.

![image](https://github.com/nanase1025/SHAP-LSTM-Momentum-Fluctuation-Prediction-Solution-to-2024-MCM-Problem-C-/assets/145645749/f16fd70b-412e-4757-8a5f-3fe24223ac2b)

#### 2.2.2 Model Construction

Model Construction 80% of the data is used as the training set and 20% as the test set. The model's hyperparameters and optimization strategy are set according to the Table below, and the training set is fitted to obtain the loss function shown in Figure below.

![image](https://github.com/nanase1025/SHAP-LSTM-Momentum-Fluctuation-Prediction-Solution-to-2024-MCM-Problem-C-/assets/145645749/ec63717a-54bf-4c5a-8d93-9f2a0b415df7)

![image](https://github.com/nanase1025/SHAP-LSTM-Momentum-Fluctuation-Prediction-Solution-to-2024-MCM-Problem-C-/assets/145645749/ac8ca462-e000-4741-94c5-064f46512ae8)

The LSTM model in this study underwent training over 1000 epochs. The loss curve shows a rapid decline in the model's loss at the early stages of training, indicating that the model could quickly learn from the data. Subsequently, the loss curve gradually flattens, indicating that the model's performance on the training set stabilizes over time, and similar performance is maintained on the test set, implying that the model does not overfit.
