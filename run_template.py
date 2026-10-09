import matplotlib.pyplot as plt
from pid_template import make_car
from pid_template import update
from pid_template import calculate_desired_acceleration
from pid_template import acceleration_to_throttle_percentage

K_P = 1.0
K_I = 0.09
K_D = 2.0
 
STEPS = 550
 
car = make_car(desired_v=20.0, dt=0.1)

# Create empty lists to record history over time
times = []
velocities = []
errors = []

if __name__ == "__main__":
    # Move your simulation loop and plt.show() here!
    for step in range(STEPS):
        # 1. Ask controller for desired acceleration and error
        desired_accel, error = calculate_desired_acceleration(car, K_P=K_P, K_I=K_I, K_D=K_D)
    
        # 2. Convert acceleration to throttle percentage (-1 to 1)
        throttle_perc = acceleration_to_throttle_percentage(desired_accel)
    
        # 3. Step physics forward
        update(car, throttle_perc)
    
        # 4. Save current state to our history lists
        times.append(car["t"])
        velocities.append(car["v"])
        errors.append(error)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    #Top Graph Velocity over Time
    ax1.plot(times, velocities, label="Car Velocity", color="green")
    ax1.axhline(y=car["desired_v"], color="blue", linestyle="--", label="Target velocity")
    ax1.set_ylabel("Velocity (m/s)")
    ax1.set_title("Car Velocity vs Time (PID controller)")
    ax1.legend()
    ax1.grid(True)

    #Bottom Graph: Error vs. Time
    ax2.plot(times, errors, label="Error", color="orange")
    ax2.axhline(y=0, color="black", linestyle="--")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Error (m/s)")
    ax2.set_title("Error vs Time (PID controller)")
    ax2.legend()
    ax2.grid(True)
    #show results
    plt.tight_layout()
    plt.show()