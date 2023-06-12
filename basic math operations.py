import torch
import torch.nn as nn
import torch.optim as optim
import re

# Define the neural network model
class MathNet(nn.Module):
    def __init__(self):
        super(MathNet, self).__init__()
        self.fc = nn.Linear(2, 1)  # Fully connected layer with 2 input units and 1 output unit

    def forward(self, x):
        x = self.fc(x)
        return x

# Create an instance of the model
model = MathNet()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    # Generate random addition and subtraction examples
    x = torch.randint(low=0, high=100, size=(100, 2), dtype=torch.float)
    operations = torch.randint(low=0, high=4, size=(100, 1), dtype=torch.float)
    y = torch.zeros((100, 1), dtype=torch.float)
    for i in range(100):
        if operations[i] == 0:
            y[i] = x[i, 0] + x[i, 1]
        elif operations[i] == 1:
            y[i] = x[i, 0] - x[i, 1]
        elif operations[i] == 2:
            y[i] = x[i, 0] * x[i, 1]
        elif operations[i] == 3:
            y[i] = x[i, 0] / x[i, 1]

    # Forward pass
    output = model(x)

    # Compute the loss
    loss = criterion(output, y)

    # Backpropagation and optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# User interaction loop
while True:
    # Get user input
    input_str = input("Enter a math problem (e.g., '5+3' or '7-2'): ")
    
    # Parse the input string to extract operands and operator
    operands = re.split(r'[+\-*/]', input_str)
    operator = re.findall(r'[+\-*/]', input_str)[0]
    operand1 = float(operands[0])
    operand2 = float(operands[1])
    
    # Prepare input tensor for prediction
    input_tensor = torch.tensor([[operand1, operand2]], dtype=torch.float)
    
    # Perform the forward pass to get the predicted output
    predicted_output = model(input_tensor)
    
    # Perform the arithmetic operation to get the expected output
    if operator == '+':
        result = operand1 + operand2
    elif operator == '-':
        result = operand1 - operand2
    elif operator == '*':
        result = operand1 * operand2
    elif operator == '/':
        result = operand1 / operand2
    else:
        result = "Unknown operator"
    
    print("Predicted Output:", predicted_output.item())
    print("Expected Output:", result)
    print()
