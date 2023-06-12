import torch
import torch.nn as nn
import torch.optim as optim
import re

# Define the neural network model
class EquationSolver(nn.Module):
    def __init__(self):
        super(EquationSolver, self).__init__()
        self.fc = nn.Linear(2, 1)  # Fully connected layer with 2 input units and 1 output unit

    def forward(self, x):
        x = self.fc(x)
        return x

# Create an instance of the model
model = EquationSolver()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    # Generate random equation examples
    x = torch.randint(low=0, high=100, size=(100, 2), dtype=torch.float)
    y = x[:, 0:1] + x[:, 1:2]  # Equation: x + y = z

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
    input_str = input("Enter an equation (e.g., 'x+5=12' or '10=y-3'): ")
    
    try:
        # Remove spaces from the input string
        input_str = input_str.replace(" ", "")
        
        # Parse the input string to extract the equation components
        equation_parts = re.split(r'(\+|=|-)', input_str)
        equation_parts = [part.strip() for part in equation_parts if part.strip() != '']
        variable = None
        operand1 = None
        operand2 = None
        operator = None
        
        # Extract the variable dynamically
        variables = set(['x', 'y'])  # Set of possible variables
        equation_vars = set(equation_parts)  # Variables present in the equation
        variable = list(variables.intersection(equation_vars))
        
        if len(variable) != 1 or len(equation_parts) != 3:
            print("Invalid equation format.")
            continue
        
        variable = variable[0]  # Convert set to a single variable
        
        # Check the equation format and extract the components
        if '+' in equation_parts:
            operator = '+'
            operand1, operand2 = equation_parts
        elif '-' in equation_parts:
            operator = '-'
            operand1, operand2 = equation_parts
        
        # Prepare input tensor for prediction
        if variable == 'x':
            input_tensor = torch.tensor([[float(operand1), float(operand2)]], dtype=torch.float)
        elif variable == 'y':
            input_tensor = torch.tensor([[float(operand2), float(operand1)]], dtype=torch.float)
        
        # Perform the forward pass to get the predicted output
        predicted_output = model(input_tensor)
        
        # Solve the equation to get the value of the variable
        if operator == '+':
            result = predicted_output.item() - float(operand2)
        elif operator == '-':
            result = float(operand1) - predicted_output.item()
        
        print("Predicted Value of", variable + ":", result)
        print()
    except:
        print("Invalid equation format. Please try again.")
