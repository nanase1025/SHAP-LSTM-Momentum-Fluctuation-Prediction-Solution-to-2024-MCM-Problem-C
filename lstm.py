import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import shap
import matplotlib.pyplot as plt

# read Excel file
df = pd.read_csv('path-to-your-data')
print(df.head)

features_columns = ['p1_Leading_sets', 'p1_Unstoppable_balls', 'p1_Single_shot_error_rate', 'p1_Interception_score', 'p1_distance_run', 'p1_rally_count', 'p1_speed_mph', 'p1_serve', 'p1_Leading_games', 'p2_Leading_sets', 'p2_Unstoppable_balls', 'p2_Single_shot_error_rate', 'p2_Interception_score', 'p2_distance_run', 'p2_rally_count', 'p2_speed_mph', 'p2_serve', 'p2_Leading_games', 'p1_momentum_score', 'p2_momentum_score']
target_columns = ['p1_momentum_score', 'p2_momentum_score']

# get the feature and targets
features = df[features_columns].values
targets = df[target_columns].values

# create dataset
def create_dataset(features, targets, time_step=5, future_step=5):
    Xs, ys = [], []
    for i in range(len(features) - time_step - future_step + 1):
        v = features[i:(i + time_step), :]
        Xs.append(v)
        # 只从targets中选取对应的future_step时间点
        ys.append(targets[i + time_step:i + time_step + future_step, :])
    return np.array(Xs), np.array(ys)

# solve the NA data
for column in features_columns + target_columns:
    df[column].fillna(df[column].mean(), inplace=True)
    
time_step = 5 
future_step=5

# data normalization
scaler = MinMaxScaler(feature_range=(0, 1))
features_scaled = scaler.fit_transform(df[features_columns])

targets_scaled = scaler.fit_transform(df[target_columns])

X, y = create_dataset(features_scaled, targets_scaled, time_step, future_step)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)
batch_size = 16
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)  # 注意batch_first=True
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)  
        predictions = self.linear(lstm_out[:, -1, :])  
        return predictions

model = LSTMModel(input_size=20, hidden_layer_size=100, output_size=10) 
loss_function = nn.MSELoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

epochs = 0

for i in range(epochs):
    for seq, labels in train_loader:
        if torch.isnan(seq).any():
            print("NaN found in input data")

        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))
        y_pred = model(seq)
        _, b, c = labels.shape
        labels = labels.view(-1, b * c)
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()
    if i%25 == 0:
        with torch.no_grad():
            model.eval() 
            for seq, labels in test_loader:
                y_pred = model(seq)
                _, b, c = labels.shape
                labels = labels.view(-1, b * c)
                loss = loss_function(y_pred, labels)
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f} Test Loss: {loss.item()}')

