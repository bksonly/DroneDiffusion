import os
import pickle
import numpy as np

def get_file_names(directory):
    """
    directory: path to the directory containing csv files
    Returns a list of file names in the directory that end with '.csv'.
    """
    file_names = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                file_names.append(os.path.join(root, file))
    return file_names

def read_csv_file(M, file_name):
    """
        M: nominal mass of the quadrotor
        file_name: path to the csv file

        Extracts the states and control inputs from the csv file,
        calculates the uncertainty \mathcal{H} (Eq. 3 in the paper),
        and returns the states, actions (uncertainty), and terminal flags.
    """
    data = []
    with open(file_name, 'r') as f:
        for line in f:
            columns = line.strip().split(',')
            data.append(columns)
    
    data = np.array(data)

    try:
        # Convert strings to floats and perform the arithmetic operation
        secs = data[2:, 2].astype(float)
        nsecs = data[2:, 3].astype(float) * 0.000000001
        t = secs + nsecs
        t = t - t[0]  # Offset by the first entry

        x_curr = data[2:, 5].astype(float)
        y_curr = data[2:, 6].astype(float)
        z_curr = data[2:, 7].astype(float)

        x_dot_curr = data[2:, 8].astype(float)
        y_dot_curr = data[2:, 9].astype(float)
        z_dot_curr = data[2:, 10].astype(float)

        x_ddot_curr = data[2:, 11].astype(float)
        y_ddot_curr = data[2:, 12].astype(float)
        z_ddot_curr = data[2:, 13].astype(float)

        roll_curr = data[2:, 32].astype(float)
        pitch_curr = data[2:, 33].astype(float)
        yaw_curr = data[2:, 34].astype(float)

        roll_dot_curr = data[2:, 35].astype(float)
        pitch_dot_curr = data[2:, 36].astype(float)
        yaw_dot_curr = data[2:, 37].astype(float)

        tau_x = data[2:, 50].astype(float)
        tau_y = data[2:, 51].astype(float)
        tau_z = data[2:, 52].astype(float)
        collective_thrust = data[2:, 53].astype(float)

        tau_x_prev = data[1:-1, 50].astype(float)
        tau_y_prev = data[1:-1, 51].astype(float)
        tau_z_prev = data[1:-1, 52].astype(float)
        collective_thrust_prev = data[1:-1, 53].astype(float)

    except IndexError as e:
        raise IndexError(f"Data array is missing required columns: {e}")
    except ValueError as e:
        raise ValueError(f"Data cannot be converted to float: {e}")

    terminal = np.zeros(len(t), dtype='bool')
    terminal[-1]=1
    states = np.array([x_curr, y_curr, z_curr,
                        x_dot_curr, y_dot_curr, z_dot_curr,
                        x_ddot_curr, y_ddot_curr, z_ddot_curr,
                        roll_curr, pitch_curr, yaw_curr,
                        tau_x_prev, tau_y_prev, tau_z_prev,])
    p_ddot = np.array([x_ddot_curr, y_ddot_curr, z_ddot_curr])
    tau = np.array([tau_x, tau_y, tau_z])
    actions = tau - M * np.identity(tau.shape[0]) @ p_ddot

    return states, actions, terminal

def prepare_data():
    dataset_dir = 'dataset'
    file_names = get_file_names(dataset_dir)
    data = {'states': [], 'actions': [], 'terminal': []}
    max_length = 0
    for file_name in file_names:
        states, actions, terminals = read_csv_file(1.0, file_name)
        data['states'].append(states.T)
        data['actions'].append(actions.T)
        data['terminal'].append(terminals)
        max_length = max(max_length, states.shape[-1])

    for key in data.keys():
        data[key] = np.concatenate(data[key], axis=0)
    
    with open(os.path.join(dataset_dir, 'data.pkl'), 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Data preparation complete.\nPlease set `max_path_length` to {max_length} in your config file.")

if __name__ == "__main__":
    prepare_data()