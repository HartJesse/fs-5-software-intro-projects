#pid_template.py

import numpy as np


def make_car(desired_v:float=20.0, dt:float=0.1) -> dict:
    """ 
    Generates a dictionary that holds all the car's values. Keeps track of state varaibles.
    """
    car_state_dictionary : dict[str, float | None] = {
        "v" : 0, #velocity of your car 
        "a" : 0, #acceleration of your car
        "t" : 0, #time of your car
        "x" : 0, #position of your car
        "dt" : dt, #time step of your car, how much the time changes every time you update/step
        "desired_v" : desired_v, #desired velocity of your car, the velocity you want to maintain
        "step" : 0,
    
        #hint: use these variables in the integral and derivative portion of your PID control (steps 5 and 6 )
        "error_prev" : None,
        "net_integral" : 0.0
    }
    return car_state_dictionary

def update(car: dict, throttle_perc: float, mass: float = 1000, max_throttle_force: float = 5000, friction: float = 2.0) -> None:
        """
        Updates the car's state variables based on the throttle percentage.
        Use this function after finding throttle percentage to update the car's state variables.

        Inputs:
        car: dictionary containing the car's state variables
        throttle_perc: float, throttle percentage (-1 to 1)

        Outputs:
        None, but updates the car's state variables
        """
        force = throttle_perc * max_throttle_force
        car["a"] = (force / mass) - friction
        car["v"] += car["a"] * car["dt"]
        car["x"] += car["v"] * car["dt"]
        car["t"] += car["dt"]
        car["step"] += 1


def calculate_desired_acceleration(car: dict, K_P: float, K_I: float = 0.0, K_D: float = 0.0) -> tuple[float, float]:
        #input: car["v"], car["desired_v"] (floats)
        #output: desired acceleration and error tuple(float, float)

        # Step 1: Calculate the 'error'
        error = calc_error(car["desired_v"], car["v"])

        # Step 2: Update the integral term
        car["net_integral"] += calc_net_integral(error, car["dt"])

        # Step 3: Calculate the derivative term
        if car["error_prev"] is None:
            car["error_prev"] = error

        #Step 3.1: Calculate the derivative of the error
        de = calc_de(error, car["error_prev"])
        derivative = calc_derivative(de, car["dt"])

        # ALWAYS update error_prev for the next step!
        car["error_prev"] = error

        proportional_gain = calc_proportional_gain(K_P, error)
        integral_gain = calc_integral_gain(K_I, car["net_integral"])
        derivative_gain = calc_derivative_gain(K_D, derivative)

        # Step 2: Calculate the desired acceleration
        desired_acceleration = proportional_gain + integral_gain + derivative_gain

        return desired_acceleration, error




def acceleration_to_throttle_percentage(acceleration_desired: float, mass: float = 1000, max_throttle_force: float = 5000) -> float:
        #input: desired_acceleration(float)

        #determine max potential acceleration for our sim.
        max_acceleration = max_throttle_force / mass

        #Calculate desired throttle percentage based off desired acceleration, mass, and max throttle force
        desired_throttle = acceleration_desired / max_acceleration

        #translate throttle into a percentage between -1 and 1
        throttle_percentage = np.clip(desired_throttle, -1, 1)

        return float(throttle_percentage)


def calc_error(desired_velocity: float, current_velocity: float) -> float:
        return desired_velocity - current_velocity

def calc_net_integral(error: float, dt: float) -> float:
        return error * dt

def calc_de(error: float, error_prev: float) -> float:
        return error - error_prev

def calc_derivative(change_of_error: float, dt: float) -> float:
       return change_of_error / dt

def calc_proportional_gain(K_P: float, error: float) -> float:
        return K_P * error

def calc_integral_gain(K_I: float, net_integral: float) -> float:
        return K_I * net_integral

def calc_derivative_gain(K_D: float, derivative: float) -> float:
        return K_D * derivative