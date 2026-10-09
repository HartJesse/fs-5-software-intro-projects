import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#train_ml.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from run_template import K_P
from pid_template import (make_car, 
                     update, 
                     acceleration_to_throttle_percentage, 
                     calc_error, 
                     calc_proportional_gain)


#vectors to store velocity and acceleration during training data generation
velocity_data= []
acceleration_data = []
STEPS_PER_RUN = 100


# Sample across target velocities (10 to 30 m/s) and current velocities (0 to 25 m/s)
for target_v in np.linspace(10.0, 30.0, 20):

    #initialize a new car for each target velocity and get more data for better coverege of the velocity space
    car = make_car(desired_v=target_v, dt=0.1)
    for step in range(STEPS_PER_RUN):
        
        # PID formula (P-only)
        #we are always comparing the current velocity(car["v"]) to the target 
        # velocity is updated at the end before iterating
        error = calc_error(target_v, car["v"])
        desired_acceleration = calc_proportional_gain(K_P, error)
        
        # save recorded velocities and acceleration
        velocity_data.append([car["v"], target_v])
        acceleration_data.append([desired_acceleration])

        # 3. Update physics for the next frame
        throttle = acceleration_to_throttle_percentage(desired_acceleration)
        update(car, throttle)

# Convert Python lists into PyTorch Tensors
input_features = torch.tensor(velocity_data, dtype=torch.float32)
target_accelerations = torch.tensor(acceleration_data, dtype=torch.float32)

print("\n--- Data Collection Complete ---")
print(f"Total Data Samples Collected: {len(velocity_data)}")
print(f"Input Features Tensor Shape : {input_features.shape}")
print(f"Target Accel Tensor Shape   : {target_accelerations.shape}")

print("\nFirst 5 Samples in Input Features Tensor [current_v, target_v]:")
print(input_features[:5])

print("\nFirst 5 Corresponding Target Accelerations [desired_accel]:")
print(target_accelerations[:5])

# 1. Define the Linear Regression Model
# 2 inputs (current_v, target_v) -> 1 output (desired_acceleration)
model = nn.Linear(in_features=2, out_features=1)

#instantiate loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#Training Loop
epochs = 900
print("\n--- Starting Training ---")
for epoch in range(epochs):
    # Forward Pass: Predict accelerations using current weights
    predictions = model(input_features)
    
    # Calculate Loss: Pass predicted tensors and target tensors into criterion and do MSE
    loss = criterion(predictions, target_accelerations)
    
    # clear out gradients, backpropagate, and update weights
    optimizer.zero_grad()
    loss.backward() #crazy gradient math here gross
    optimizer.step()
    
    # Print progress every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.6f}")

# 3. Extract and Display Learned Parameters
w1, w2 = model.weight[0].detach().numpy()
b = model.bias[0].detach().numpy()

print("\n--- Learned Parameters vs PID Constants ---")
print(f"Actual K_P used in PID data generation : {K_P}")
print(f"Learned Weight 1 (multiplier for current_v) : {w1:.4f}  (Expected: ~{-K_P:.4f})")
print(f"Learned Weight 2 (multiplier for target_v)  : {w2:.4f}  (Expected: ~{K_P:.4f})")
print(f"Learned Bias                                : {b:.4f}  (Expected: ~0.0000)")