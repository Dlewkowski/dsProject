import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('C:\\Users\\denie\\Downloads\\diabetes - Copy.csv')
predictors = df.drop(columns=['Diabetes'])
num_features = predictors.shape[1]


# q1 perceptron model

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(num_features, 1)
        
    def forward(self, x):
        return self.fc1(x)
torch.manual_seed(14949753)
model = Net()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_p = nn.BCEWithLogitsLoss()

X = predictors.values
y = df['Diabetes'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14949753)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


epochs = 200
for epoch in range(epochs):
    model.train()
    opt.zero_grad()
    out = model(X_train)
    loss = loss_p(out, y_train)
    loss.backward()
    opt.step()
model.eval()
with torch.no_grad():
    logits = model(X_test)
    probs = torch.sigmoid(logits)
    
auc = roc_auc_score(y_test, probs.numpy())
print(f"Perceptron AUC: {auc}")

# figure 1, question 1: roc curve
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test.numpy(), probs.numpy())
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Perceptron ROC Curve')
plt.legend()
plt.show()

#q2 FFNN

class Net(nn.Module):
    def __init__(self, n_layers, activation):
        super().__init__()
        self.activation = activation
        self.layers = nn.ModuleList(
            [nn.Linear(num_features, 64)] +
            [nn.Linear(64, 64) for _ in range(n_layers - 1)] +
            [nn.Linear(64, 1)]
        )

    def forward(self, x):
        for layer in self.layers[:-1]:
            if self.activation == 'relu':
                x = F.relu(layer(x))
            elif self.activation == 'sigmoid':
                x = torch.sigmoid(layer(x))
            else:
                x = layer(x)
        return self.layers[-1](x)

layer_counts = [1, 2, 4, 6, 8, 10, 20, 30]
activations = ['relu', 'sigmoid', 'none']

for activation in activations:
    for n_layers in layer_counts:
        torch.manual_seed(14949753)
        model_f = Net(n_layers, activation)
        opt_f = torch.optim.Adam(model_f.parameters(), lr=1e-3)
        loss_f = nn.BCEWithLogitsLoss()

        for epoch in range(200):
            model_f.train()
            opt_f.zero_grad()
            loss_f(model_f(X_train), y_train).backward()
            opt_f.step()

        model_f.eval()
        with torch.no_grad():
            auc_f = roc_auc_score(y_test, torch.sigmoid(model_f(X_test)).numpy())
        print(f"FFNN AUC ({n_layers} layers, {activation}): {auc_f}")
        
#q2 figure 2, auc as a function of depth of neural netx
layer_counts = [1, 2, 4, 6, 8, 10, 20, 30]

relu_aucs    = [0.8209, 0.8282, 0.8299, 0.8303, 0.8298, 0.8298, 0.8305, 0.5000]
sigmoid_aucs = [0.8099, 0.8003, 0.8056, 0.8122, 0.8062, 0.7100, 0.5000, 0.5001]
none_aucs    = [0.8234, 0.8231, 0.8231, 0.8231, 0.8231, 0.8231, 0.8231, 0.8231]

plt.figure(figsize=(10, 6))
plt.plot(layer_counts, relu_aucs,    marker='o', label='ReLU')
plt.plot(layer_counts, sigmoid_aucs, marker='o', label='Sigmoid')
plt.plot(layer_counts, none_aucs,    marker='o', label='None')
plt.axhline(y=0.7730, color='black', linestyle='--', label='Perceptron Baseline')
plt.xlabel('Number of Hidden Layers')
plt.ylabel('AUC')
plt.title('AUC vs Network Depth by Activation Function')
plt.legend()
plt.grid(True)
plt.show()
        
        
# q3 will use the best model from above
from sklearn.metrics import roc_curve, auc

torch.manual_seed(14949753)
model_q3 = Net(6, 'relu')
opt_q3 = torch.optim.Adam(model_q3.parameters(), lr=1e-3)
loss_q3 = nn.BCEWithLogitsLoss()

for epoch in range(200):
    model_q3.train()
    opt_q3.zero_grad()
    loss_q3(model_q3(X_train), y_train).backward()
    opt_q3.step()

model_q3.eval()
with torch.no_grad():
    probs_q3 = torch.sigmoid(model_q3(X_test))

#figure 3, auc 

fpr, tpr, _ = roc_curve(y_test.numpy(), probs_q3.numpy())
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'DNN (6 layers, ReLU) AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - DNN (6 Layer ReLU)')
plt.legend()
plt.grid(True)
plt.show()

# q4 predicting BMI, with a ffnn
from sklearn.metrics import mean_squared_error

predictors_bmi = df.drop(columns=['BMI', 'Zodiac'])
target_bmi = df['BMI'].values

X_bmi = predictors_bmi.values
y_bmi = target_bmi

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_bmi, y_bmi, test_size=0.2, random_state=14949753)

scaler_b = StandardScaler()
X_train_b = scaler_b.fit_transform(X_train_b)
X_test_b = scaler_b.transform(X_test_b)

scaler_y = StandardScaler()
y_train_b = scaler_y.fit_transform(y_train_b.reshape(-1,1))
y_test_b = scaler_y.transform(y_test_b.reshape(-1,1))

X_train_b = torch.tensor(X_train_b, dtype=torch.float32)
X_test_b = torch.tensor(X_test_b, dtype=torch.float32)
y_train_b = torch.tensor(y_train_b, dtype=torch.float32)
y_test_b = torch.tensor(y_test_b, dtype=torch.float32)

num_features_bmi = predictors_bmi.shape[1]
class bmiNet(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation
        self.layers = nn.ModuleList([
            nn.Linear(num_features_bmi, 256), #generally for 2^n neurons rmse decreased as n increased, but plateaus. 256 to balance model accuracy with computation time 
            nn.Linear(256, 1)])
        
    def forward(self, x):
        if self.activation == 'relu':
            x = F.relu(self.layers[0](x))
        elif self.activation == 'sigmoid':
            x = torch.sigmoid(self.layers[0](x))
        else:
            x = self.layers[0](x)
        return self.layers[1](x)
    
for activation in ['relu', 'sigmoid', 'none']:
    torch.manual_seed(14949753)
    model_reg = bmiNet(activation)
    opt_reg = torch.optim.Adam(model_reg.parameters(), lr=1e-3)
    loss_reg = nn.MSELoss()
    
    for epoch in range(200):
        model_reg.train()
        opt_reg.zero_grad()
        loss_reg(model_reg(X_train_b), y_train_b).backward()
        opt_reg.step()
        
    model_reg.eval()
    with torch.no_grad():
        preds = model_reg(X_test_b)
        preds_actual = scaler_y.inverse_transform(preds.numpy())
        y_actual = scaler_y.inverse_transform(y_test_b.numpy())
        rmse = np.sqrt(mean_squared_error(y_actual, preds_actual))
        
        
    print(f"BMI RMSE({activation}): {rmse: .4f}")

# q4 figure, compares performance by activation function (bar chart)
activations = ['ReLU', 'Sigmoid', 'None']
rmses = [6.0890, 6.2174, 6.2232]

plt.figure(figsize=(8, 6))
bars = plt.bar(activations, rmses, color=['steelblue', 'coral', 'mediumseagreen'], width=0.5)


for bar, rmse in zip(bars, rmses):
    plt.text(bar.get_x() + bar.get_width()/2, 
             bar.get_height() + 0.005, 
             f'{rmse:.4f}', 
             ha='center', va='bottom')

plt.ylabel('RMSE (BMI Units)')
plt.xlabel('Activation Function')
plt.title('BMI Prediction RMSE by Activation Function (1 Hidden Layer)')
plt.ylim(5.8, 6.4)  # zoom in to show differences clearly
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
    
## question 5 building the best model to predict bmi from other data 


class dnnNet(nn.Module):
    def __init__(self, n_layers, n_neurons):
        super().__init__()
        
        self.layers = nn.ModuleList(
            [nn.Linear(num_features_bmi, n_neurons)] +
            [nn.Linear(n_neurons, n_neurons) for _ in range (n_layers - 1)] +
            [nn.Linear(n_neurons, 1)]   
        )
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)
    
configs = [
    (2, 256),
    (4, 256),
    
]
    
for n_layers, n_neurons in configs:
    torch.manual_seed(14949753)
    model_dnn = dnnNet(n_layers, n_neurons)
    opt_dnn = torch.optim.Adam(model_dnn.parameters(), lr=1e-3)
    loss_dnn = nn.MSELoss()
    
    for epoch in range(200):
        model_dnn.train()
        opt_dnn.zero_grad()
        loss_dnn(model_dnn(X_train_b), y_train_b).backward()
        opt_dnn.step()
        
    model_dnn.eval()
    with torch.no_grad():
        preds = model_dnn(X_test_b)
        preds_actual = scaler_y.inverse_transform(preds.numpy())
        y_actual = scaler_y.inverse_transform(y_test_b.numpy())
        rmse = np.sqrt(mean_squared_error(y_actual, preds_actual))
    print(f"DNN BMI RMSE (layers={n_layers}, neurons={n_neurons}): {rmse:.4f}")
        
   
#q5 figure
torch.manual_seed(14949753)
model_q5 = dnnNet(2, 256)
opt_q5 = torch.optim.Adam(model_q5.parameters(), lr=1e-3)
loss_q5 = nn.MSELoss()

for epoch in range(500):
    model_q5.train()
    opt_q5.zero_grad()
    loss_q5(model_q5(X_train_b), y_train_b).backward()
    opt_q5.step()

model_q5.eval()
with torch.no_grad():
    preds_q5 = model_q5(X_test_b)
    preds_actual_q5 = scaler_y.inverse_transform(preds_q5.numpy())
    y_actual_q5 = scaler_y.inverse_transform(y_test_b.numpy())

plt.figure(figsize=(8, 6))
plt.scatter(y_actual_q5, preds_actual_q5, alpha=0.3, color='steelblue', s=5)
plt.plot([y_actual_q5.min(), y_actual_q5.max()], 
         [y_actual_q5.min(), y_actual_q5.max()], 
         'r--', label='Perfect Prediction')
plt.xlabel('Actual BMI')
plt.ylabel('Predicted BMI')
plt.title('Predicted vs Actual BMI - Best DNN (2 layers, 256 neurons, ReLU)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()



#Bonus questions
    
class NetQ3(nn.Module):
    def __init__(self, n_layers, activation):
        super().__init__()
        self.activation = activation
        self.layers = nn.ModuleList(
            [nn.Linear(num_features, 64)] +
            [nn.Linear(64, 64) for _ in range(n_layers - 1)] +
            [nn.Linear(64, 1)]
        )
    def forward(self, x):
        for layer in self.layers[:-1]:
            if self.activation == 'relu':
                x = F.relu(layer(x))
            elif self.activation == 'sigmoid':
                x = torch.sigmoid(layer(x))
            else:
                x = layer(x)
        return self.layers[-1](x)
    
# manual permutation importance
torch.manual_seed(14949753)
model_pi = NetQ3(6, 'relu')
opt_pi = torch.optim.Adam(model_pi.parameters(), lr=1e-3)
loss_pi = nn.BCEWithLogitsLoss()

for epoch in range(200):
    model_pi.train()
    opt_pi.zero_grad()
    loss_pi(model_pi(X_train), y_train).backward()
    opt_pi.step()

model_pi.eval()

# baseline auc
with torch.no_grad():
    baseline_auc = roc_auc_score(y_test.numpy(), torch.sigmoid(model_pi(X_test)).numpy())

feature_names = predictors.columns.tolist()
importances = []

for i in range(X_test.shape[1]):
    aucs = []
    for _ in range(10):  
        X_permuted = X_test.clone()
        X_permuted[:, i] = X_test[torch.randperm(X_test.shape[0]), i]  
        with torch.no_grad():
            permuted_auc = roc_auc_score(y_test.numpy(), torch.sigmoid(model_pi(X_permuted)).numpy())
        aucs.append(baseline_auc - permuted_auc)  
    importances.append(np.mean(aucs))


# print near zero importance features
print("\nFeatures with near-zero or negative importance:")
for i, name in enumerate(feature_names):
    if importances[i] < 0.001:
        print(f"{name}: {importances[i]:.4f}")

