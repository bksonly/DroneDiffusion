import sys
import jax.numpy as jnp
import numpy as np

import rospy

from sensor_msgs.msg import Imu

# 3D point & Stamped Pose msgs
from geometry_msgs.msg import Point, PoseStamped, TwistStamped, Vector3
import tf.transformations as transformations
from tf.transformations import *

# import all mavros messages and services
from mavros_msgs.msg import *
from mavros_msgs.srv import *
from nav_msgs.msg import *
from trajectory_msgs.msg import MultiDOFJointTrajectory as Mdjt
from std_msgs.msg import Float32
from gazebo_msgs.msg import ModelStates
from msg_check.msg import PlotDataMsg

from diffusion.diffusion_inference import Policy


class fcuModes:
    def __init__(self):
        pass

    def setTakeoff(self):
        rospy.wait_for_service("mavros/cmd/takeoff")
        try:
            takeoffService = rospy.ServiceProxy(
                "mavros/cmd/takeoff", mavros_msgs.srv.CommandTOL
            )
            takeoffService(altitude=3)
        except rospy.ServiceException as e:
            print("Service takeoff call failed: %s" % e)

    def setArm(self):
        rospy.wait_for_service("mavros/cmd/arming")
        try:
            armService = rospy.ServiceProxy(
                "mavros/cmd/arming", mavros_msgs.srv.CommandBool
            )
            armService(True)
        except rospy.ServiceException as e:
            print("Service arming call failed: %s" % e)

    def setDisarm(self):
        rospy.wait_for_service("mavros/cmd/arming")
        try:
            armService = rospy.ServiceProxy(
                "mavros/cmd/arming", mavros_msgs.srv.CommandBool
            )
            armService(False)
        except rospy.ServiceException as e:
            print("Service disarming call failed: %s" % e)

    def setStabilizedMode(self):
        rospy.wait_for_service("mavros/set_mode")
        try:
            flightModeService = rospy.ServiceProxy(
                "mavros/set_mode", mavros_msgs.srv.SetMode
            )
            flightModeService(custom_mode="STABILIZED")
        except rospy.ServiceException as e:
            print(
                "service set_mode call failed: %s. Stabilized Mode could not be set."
                % e
            )

    def setOffboardMode(self):
        rospy.wait_for_service("mavros/set_mode")
        try:
            flightModeService = rospy.ServiceProxy(
                "mavros/set_mode", mavros_msgs.srv.SetMode
            )
            flightModeService(custom_mode="OFFBOARD")
        except rospy.ServiceException as e:
            print(
                "service set_mode call failed: %s. Offboard Mode could not be set." % e
            )

    def setAltitudeMode(self):
        rospy.wait_for_service("mavros/set_mode")
        try:
            flightModeService = rospy.ServiceProxy(
                "mavros/set_mode", mavros_msgs.srv.SetMode
            )
            flightModeService(custom_mode="ALTCTL")
        except rospy.ServiceException as e:
            print(
                "service set_mode call failed: %s. Altitude Mode could not be set." % e
            )

    def setPositionMode(self):
        rospy.wait_for_service("mavros/set_mode")
        try:
            flightModeService = rospy.ServiceProxy(
                "mavros/set_mode", mavros_msgs.srv.SetMode
            )
            flightModeService(custom_mode="POSCTL")
        except rospy.ServiceException as e:
            print(
                "service set_mode call failed: %s. Position Mode could not be set." % e
            )

    def setAutoLandMode(self):
        rospy.wait_for_service("mavros/set_mode")
        try:
            flightModeService = rospy.ServiceProxy(
                "mavros/set_mode", mavros_msgs.srv.SetMode
            )
            flightModeService(custom_mode="AUTO.LAND")
        except rospy.ServiceException as e:
            print(
                "service set_mode call failed: %s. Autoland Mode could not be set." % e
            )


class Controller:
    # initialization method
    def __init__(self):
        self.state = State()
        self.sp = PoseStamped()
        self.yaw_angle = Float32()
        self.yaw_angle.data = 0.0

        self.imu = Imu()

        self.sp.pose.position.x = 0
        self.sp.pose.position.y = 0
        self.sp.pose.position.z = 2

        self.local_pos = PoseStamped()
        self.local_vel = TwistStamped()

        self.local_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.desVel = np.zeros(3)
        self.desAcc = np.zeros(3)

        self.att_cmd = PoseStamped()
        self.thrust_cmd = Thrust()

        # Gains
        self.Lam = np.array([2.0, 2.0, 8.0])
        self.Phi = np.array([1.5, 1.5, 1.5])
        self.M_bar = 1
        self.Kp0, self.alpha_0, self.v = (
            np.array([0.1, 0.1, 0.1]),
            np.array([1, 1, 1]),
            0.1,
        )

        self.norm_thrust_const = 0.056
        self.max_th = 16.0
        self.max_throttle = 0.96
        self.gravity = np.array([0, 0, 9.8])
        self.pre_time = rospy.get_time()

        self.armed = False
        self.command = AttitudeTarget()
        self.collective_thrust = 0.0

        # Publishers
        self.att_pub = rospy.Publisher(
            "mavros/setpoint_attitude/attitude", PoseStamped, queue_size=10
        )
        self.thrust_pub = rospy.Publisher(
            "mavros/setpoint_attitude/thrust", Thrust, queue_size=10
        )
        self.body_rate = rospy.Publisher(
            "/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=10
        )

        self.obs = None
        self.last_tau = np.array([0, 0, 0])
        self.last_vel = np.array([0, 0, 0])
        self.last_accel = np.array([0, 0, 0])

        self.policy = Policy(config="experiment.yaml", use_ema=True, rng=42)
        self.policy(jnp.ones((15,), dtype=jnp.float32))  # Warm up the JAX model

    def base_link_pos(self, msg):
        idx = msg.name.index("iris")
        iris_pose = msg.pose[idx]
        iris_twist = msg.twist[idx]

        self.local_pos.pose.position.x = iris_pose.position.x
        self.local_pos.pose.position.y = iris_pose.position.y
        self.local_pos.pose.position.z = iris_pose.position.z

        self.local_pos.pose.orientation.x = iris_pose.orientation.x
        self.local_pos.pose.orientation.y = iris_pose.orientation.y
        self.local_pos.pose.orientation.z = iris_pose.orientation.z
        self.local_pos.pose.orientation.w = iris_pose.orientation.w

        self.local_vel.twist.linear.x = iris_twist.linear.x
        self.local_vel.twist.linear.y = iris_twist.linear.y
        self.local_vel.twist.linear.z = iris_twist.linear.z

        self.local_vel.twist.angular.x = iris_twist.angular.x
        self.local_vel.twist.angular.y = iris_twist.angular.y
        self.local_vel.twist.angular.z = iris_twist.angular.z

    def posCb(self, msg):
        self.local_pos.pose.position.x = msg.pose.position.x
        self.local_pos.pose.position.y = msg.pose.position.y
        self.local_pos.pose.position.z = msg.pose.position.z

        self.local_pos.pose.orientation.x = msg.pose.orientation.x
        self.local_pos.pose.orientation.y = msg.pose.orientation.y
        self.local_pos.pose.orientation.z = msg.pose.orientation.z
        self.local_pos.pose.orientation.w = msg.pose.orientation.w

    def velCb(self, msg):
        self.local_vel.twist.linear.x = msg.twist.linear.x
        self.local_vel.twist.linear.y = msg.twist.linear.y
        self.local_vel.twist.linear.z = msg.twist.linear.z

        self.local_vel.twist.angular.x = msg.twist.angular.x
        self.local_vel.twist.angular.y = msg.twist.angular.y
        self.local_vel.twist.angular.z = msg.twist.angular.z

    def multiDoFCb(self, msg):
        pt = msg.points[0]
        self.sp.pose.position.x = pt.transforms[0].translation.x
        self.sp.pose.position.y = pt.transforms[0].translation.y
        self.sp.pose.position.z = pt.transforms[0].translation.z
        self.desVel = np.array(
            [
                pt.velocities[0].linear.x,
                pt.velocities[0].linear.y,
                pt.velocities[0].linear.z,
            ]
        )
        self.desAcc = np.array(
            [
                pt.accelerations[0].linear.x,
                pt.accelerations[0].linear.y,
                pt.accelerations[0].linear.z,
            ]
        )

    def stateCb(self, msg):
        self.state = msg

    def updateSp(self):
        self.sp.pose.position.x = self.local_pos.pose.position.x
        self.sp.pose.position.y = self.local_pos.pose.position.y
        self.sp.pose.position.z = self.local_pos.pose.position.z

    def accCB(self, msg):
        self.imu.orientation.w = msg.orientation.w
        self.imu.orientation.x = msg.orientation.x
        self.imu.orientation.y = msg.orientation.y
        self.imu.orientation.z = msg.orientation.z

        self.imu.angular_velocity.x = msg.angular_velocity.x
        self.imu.angular_velocity.y = msg.angular_velocity.y
        self.imu.angular_velocity.z = msg.angular_velocity.z

        self.imu.linear_acceleration.x = msg.linear_acceleration.x
        self.imu.linear_acceleration.y = msg.linear_acceleration.y
        self.imu.linear_acceleration.z = msg.linear_acceleration.z

    def newPoseCB(self, msg):
        if self.sp.pose.position != msg.pose.position:
            print("New pose received")
        self.sp.pose.position.x = msg.pose.position.x
        self.sp.pose.position.y = msg.pose.position.y
        self.sp.pose.position.z = msg.pose.position.z

        self.sp.pose.orientation.x = msg.pose.orientation.x
        self.sp.pose.orientation.y = msg.pose.orientation.y
        self.sp.pose.orientation.z = msg.pose.orientation.z
        self.sp.pose.orientation.w = msg.pose.orientation.w

    def yawAngle(self, msg):
        self.yaw_angle.data = msg.data

    def vector2Arrays(self, vector):
        return np.array([vector.x, vector.y, vector.z])

    def vector3Arrays(self, vector):
        return np.array([vector.x, vector.y, vector.z, vector.w])

    def array2Vector3(self, array, vector):
        vector.x = array[0]
        vector.y = array[1]
        vector.z = array[2]

    def array2Vector4(self, array, vector):
        vector.x = array[0]
        vector.y = array[1]
        vector.z = array[2]
        vector.w = array[3]

    def sigmoid(self, s, v):
        if np.absolute(s) > v:
            return s / np.absolute(s)
        else:
            return s / v

    def get_obs(self):
        curPos = self.vector2Arrays(self.local_pos.pose.position)
        desPos = self.vector2Arrays(self.sp.pose.position)
        curVel = self.vector2Arrays(self.local_vel.twist.linear)
        curAcc = self.vector2Arrays(self.imu.linear_acceleration) - self.gravity

        orientation_q = self.local_pos.pose.orientation
        orientation_list = [
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w,
        ]
        (roll_curr, pitch_curr, yaw_curr) = euler_from_quaternion(orientation_list)

        accel = curAcc
        obs = np.array(
            [
                curPos[0],
                curPos[1],
                curPos[2],
                curVel[0],
                curVel[1],
                curVel[2],
                curAcc[0],
                curAcc[1],
                curAcc[2],
                roll_curr,
                pitch_curr,
                yaw_curr,
                self.last_tau[0],
                self.last_tau[1],
                self.last_tau[2],
            ],
            dtype=np.float32,
        )

        return obs, accel

    def get_action(self, curr_act):
        self.currN = curr_act

    def a_des(self):
        _, accel = self.get_obs()

        dt = rospy.get_time() - self.pre_time
        self.pre_time = self.pre_time + dt

        if dt > 0.01:
            dt = 0.01

        curPos = self.vector2Arrays(self.local_pos.pose.position)
        desPos = self.vector2Arrays(self.sp.pose.position)
        curVel = self.vector2Arrays(self.local_vel.twist.linear)
        # print(curVel)
        curAcc = self.vector2Arrays(self.imu.linear_acceleration) - self.gravity

        errPos = curPos - desPos
        errVel = curVel - self.desVel
        errAcc = curAcc - self.desAcc

        sv = errVel + np.multiply(self.Phi, errPos)
        if self.armed:
            self.Kp0 += (np.linalg.norm(sv) - np.multiply(self.alpha_0, self.Kp0)) * dt
            self.Kp0 = np.maximum(self.Kp0, 0.0001 * np.ones(3))
        Rho = self.Kp0
        delTau = np.zeros(3)
        delTau[0] = Rho[0] * self.sigmoid(sv[0], self.v)
        delTau[1] = Rho[1] * self.sigmoid(sv[1], self.v)
        delTau[2] = Rho[2] * self.sigmoid(sv[2], self.v)

        des_a = (
            -np.multiply(self.Lam, sv)
            + self.M_bar * (self.desAcc - np.multiply(self.Phi, errVel))
            + self.currN
            - delTau
        )

        print("N   :", self.currN)
        print("errPos", errPos)
        print("------------")

        if np.linalg.norm(des_a) > self.max_th:
            des_a = (self.max_th / np.linalg.norm(des_a)) * des_a

        self.last_tau = des_a
        self.last_vel = curVel
        self.last_accel = accel

        return des_a

    def geo_con_(self):
        des_a = self.a_des()

        pose = transformations.quaternion_matrix(
            np.array(
                [
                    self.local_pos.pose.orientation.x,
                    self.local_pos.pose.orientation.y,
                    self.local_pos.pose.orientation.z,
                    self.local_pos.pose.orientation.w,
                ]
            )
        )  # 4*4 matrix
        pose_temp1 = np.delete(pose, -1, axis=1)
        rot_curr = np.delete(pose_temp1, -1, axis=0)  # 3*3 current rotation matrix
        zb_curr = rot_curr[:, 2]
        thrust = self.norm_thrust_const * des_a.dot(zb_curr)
        self.collective_thrust = np.maximum(0.0, np.minimum(thrust, self.max_throttle))

        # Desired Euler orientation and Desired Rotation matrix
        rot_des = self.acc2quat(des_a, 0)  # desired yaw = 0
        rot_44 = np.vstack(
            (np.hstack((rot_des, np.array([[0, 0, 0]]).T)), np.array([[0, 0, 0, 1]]))
        )
        quat_des = quaternion_from_matrix(rot_44)

        now = rospy.Time.now()
        self.att_cmd.header.stamp = now
        self.thrust_cmd.header.stamp = now
        self.att_cmd.pose.orientation.x = quat_des[0]
        self.att_cmd.pose.orientation.y = quat_des[1]
        self.att_cmd.pose.orientation.z = quat_des[2]
        self.att_cmd.pose.orientation.w = quat_des[3]
        self.thrust_cmd.thrust = self.collective_thrust

    def acc2quat(self, des_a, des_yaw):
        proj_xb_des = np.array([np.cos(des_yaw), np.sin(des_yaw), 0.0])
        if np.linalg.norm(des_a) == 0.0:
            zb_des = np.array([0, 0, 1])
        else:
            zb_des = des_a / np.linalg.norm(des_a)
        yb_des = np.cross(zb_des, proj_xb_des) / np.linalg.norm(
            np.cross(zb_des, proj_xb_des)
        )
        xb_des = np.cross(yb_des, zb_des) / np.linalg.norm(np.cross(yb_des, zb_des))

        rotmat = np.transpose(np.array([xb_des, yb_des, zb_des]))
        return rotmat

    def pub_att(self):
        self.geo_con_()
        self.thrust_pub.publish(self.thrust_cmd)
        self.att_pub.publish(self.att_cmd)


# Main function
def main(argv):

    rospy.init_node("setpoint_node", anonymous=True)
    modes = fcuModes()  # flight modes
    cnt = Controller()  # controller object
    rate = rospy.Rate(30)
    rospy.Subscriber("mavros/state", State, cnt.stateCb)
    rospy.Subscriber("/gazebo/model_states", ModelStates, cnt.base_link_pos)

    # rospy.Subscriber('mavros/local_position/pose', PoseStamped, cnt.posCb)
    # rospy.Subscriber('mavros/local_position/velocity_local', TwistStamped, cnt.velCb)
    rospy.Subscriber("mavros/imu/data", Imu, cnt.accCB)

    rospy.Subscriber("command/trajectory", Mdjt, cnt.multiDoFCb)
    rospy.Subscriber("new_pose", PoseStamped, cnt.newPoseCB)
    rospy.Subscriber("yaw_in_deg", Float32, cnt.yawAngle)
    sp_pub = rospy.Publisher(
        "mavros/setpoint_position/local", PoseStamped, queue_size=10
    )

    print("ARMING")

    while not cnt.state.armed:
        modes.setArm()
        cnt.armed = True
        rate.sleep()

    cnt.armed = True
    k = 0

    while k < 20:
        sp_pub.publish(cnt.sp)
        rate.sleep()
        k = k + 1

    modes.setOffboardMode()
    print("---------")
    print("OFFBOARD")
    print("---------")

    # ROS main loop
    i = 0
    PLANNING_BUDGET = 8

    while not rospy.is_shutdown():

        if i % PLANNING_BUDGET == 0:
            obs = cnt.get_obs()[0]
            action = cnt.policy(obs)

        for j in range(PLANNING_BUDGET):
            curr_act = action[j]
            cnt.get_action(curr_act)

            cnt.pub_att()
            rate.sleep()


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except rospy.ROSInterruptException:
        pass
