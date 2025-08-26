# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The objective of this project is to develop a Neural Network Regression Model that can accurately predict a target variable based on input features. The model will leverage deep learning techniques to learn intricate patterns from the dataset and provide reliable predictions.

## Neural Network Model

<img width="935" height="678" alt="image" src="https://github.com/user-attachments/assets/fcae90c4-6a9b-4af0-ba5d-34b8ea2f5249" />


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: KANNAN S
### Register Number: 212223230098
```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10)
        self.fc3 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history={'loss': []}
  def forward(self,x):
    x=self.relu(self.fc1(x))
    x=self.relu(self.fc2(x))
    x=self.fc3(x)
    return x



# Initialize the Model, Loss Function, and Optimizer
kannan_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(kannan_brain.parameters(),lr=0.001)



def train_model(kannan_brain, X_train, y_train, criterion, optimizer, epochs=4000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(kannan_brain(X_train), y_train)
        loss.backward()
        optimizer.step()

        # Append loss inside the loop
        kannan_brain.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')



```
## Dataset Information

<img width="224" height="555" alt="image" src="https://github.com/user-attachments/assets/8f865725-006e-468c-8759-a862a31a4d62" />


## OUTPUT

### Training Loss Vs Iteration Plot

<img width="854" height="745" alt="image" src="https://github.com/user-attachments/assets/5e994580-2938-4d75-b4db-5a7d8063d064" />


### New Sample Data Prediction

<img width="998" height="140" alt="image" src="https://github.com/user-attachments/assets/719d625d-64b5-4383-86b4-daed73ea9397" />


## RESULT

The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
