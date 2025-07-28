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

def sigma_to_SI(sigma):
    SI = 0.0656* sigma**2 + 0.088 * sigma - 2.424e-4
    return SI
    #$f_i=0.0656\sigma ^2 +0.088\sigma-2.424\times 10^{-4}$

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
        t = data[1:, 0].astype(float) * 0.001 # Convert time from milliseconds to seconds
        t = t - t[0]  # Offset by the first entry

        x_curr = data[1:, 1].astype(float)
        y_curr = data[1:, 2].astype(float)
        z_curr = data[1:, 3].astype(float)

        x_dot_curr = data[1:, 4].astype(float)
        y_dot_curr = data[1:, 5].astype(float)
        z_dot_curr = data[1:, 6].astype(float)

        x_ddot_curr = data[1:, 7].astype(float)
        y_ddot_curr = data[1:, 8].astype(float)
        z_ddot_curr = data[1:, 9].astype(float)

        roll_curr = data[1:, 10].astype(float) / 180. * np.pi  # Convert degrees to radians
        pitch_curr = data[1:, 11].astype(float) / 180. * np.pi
        yaw_curr = data[1:, 12].astype(float) / 180. * np.pi

        roll_rate_curr = data[1:, 13].astype(float) / 1000. # Convert m rad/s to rad/s
        pitch_rate_curr = data[1:, 14].astype(float) / 1000.
        yaw_rate_curr = data[1:, 15].astype(float) / 1000.

        collective_thrust = data[1:, 16].astype(float) / 65536.
        thrust_SI = sigma_to_SI(collective_thrust)

        # ====================== 新增部分：推力坐标系转换 ======================
        # rpy顺序是zyx (yaw, pitch, roll)
        # 推力在机体坐标系中是垂直向下的 (0, 0, -1)方向
        # 我们需要将其转换到世界坐标系
        
        # 使用scipy的旋转功能 (向量化实现)
        from scipy.spatial.transform import Rotation
        
        # 准备角度数组
        angles = np.column_stack([yaw_curr, pitch_curr, roll_curr])
        
        # 创建旋转对象 (ZYX顺序)
        r = Rotation.from_euler('zyx', angles, degrees=False)
        
        # 获取所有旋转矩阵 (向量化操作)
        R = r.as_matrix()  # 形状为 (n, 3, 3)
        
        # 机体坐标系中的推力向量 (垂直向下)
        thrust_body = np.zeros((len(t), 3))
        thrust_body[:, 2] = -thrust_SI  # z轴方向向下
        
        # 将推力转换到世界坐标系 (向量化操作)
        thrust_world = np.einsum('ijk,ik->ij', R, thrust_body)
        tau_x_prev = tau_x = thrust_world[:, 0]
        tau_y_prev = tau_y = thrust_world[:, 1]
        tau_z_prev = tau_z = thrust_world[:, 2]
        
        # ====================== 新增部分结束 ======================
        m1 = data[1:, 17].astype(float) / 65536.
        m2 = data[1:, 18].astype(float) / 65536.
        m3 = data[1:, 19].astype(float) / 65536.
        m4 = data[1:, 20].astype(float) / 65536.
        # tau_x = data[1:, 50].astype(float)
        # tau_y = data[1:, 51].astype(float)
        # tau_z = data[1:, 52].astype(float)
        

        # tau_x_prev = data[1:-1, 50].astype(float)
        # tau_y_prev = data[1:-1, 51].astype(float)
        # tau_z_prev = data[1:-1, 52].astype(float)
        # collective_thrust_prev = data[1:-1, 53].astype(float)

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