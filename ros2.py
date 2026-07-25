# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-2-Clause

import json
import time
from pathlib import Path

import numpy as np
import omni
import omni.replicator.core as rep
import isaaclab.sim as sim_utils
from nav_msgs.msg import Odometry
from pxr import Gf
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rosgraph_msgs.msg import Clock
from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import Imu, JointState, PointCloud2, PointField
from std_msgs.msg import Float32MultiArray, Header

from isaaclab.sensors import Camera, CameraCfg


_L1_PROFILE_PATH = Path(__file__).parent / "Isaac_sim" / "Unitree" / "Unitree_L1.json"
_LIDAR_PUBLISH_PERIOD_S = 0.1
_GO2_LIDAR_X_M = 0.29515
_GO2_LIDAR_Y_M = -0.00003
_GO2_LIDAR_Z_M = -0.06597
_GO2_LIDAR_PITCH_RAD = -0.2
_LAST_LIDAR_PUBLISH_S = None


def _load_l1_attributes():
    """Translate the legacy L1 JSON profile into Isaac Sim OmniLidar attributes."""
    with _L1_PROFILE_PATH.open(encoding="utf-8") as profile_file:
        profile = json.load(profile_file)["profile"]

    attributes = {
        "omni:sensor:Core:scanType": profile["scanType"].upper(),
        "omni:sensor:Core:intensityProcessing": profile["intensityProcessing"].upper(),
        "omni:sensor:Core:rayType": profile["rayType"].upper(),
        "omni:sensor:Core:nearRangeM": profile["nearRangeM"],
        "omni:sensor:Core:farRangeM": profile["farRangeM"],
        "omni:sensor:Core:validStartAzimuthDeg": profile["startAzimuthDeg"],
        "omni:sensor:Core:validEndAzimuthDeg": profile["endAzimuthDeg"],
        "omni:sensor:Core:rangeResolutionM": profile["rangeResolutionM"],
        "omni:sensor:Core:rangeAccuracyM": profile["rangeAccuracyM"],
        "omni:sensor:Core:avgPowerW": profile["avgPowerW"],
        "omni:sensor:Core:minReflectance": profile["minReflectance"],
        "omni:sensor:Core:minReflectionRangeM": profile["minReflectanceRange"],
        "omni:sensor:Core:waveLengthNm": profile["wavelengthNm"],
        "omni:sensor:Core:pulseTimeNs": profile["pulseTimeNs"],
        "omni:sensor:Core:azimuthErrorMean": profile["azimuthErrorMean"],
        "omni:sensor:Core:azimuthErrorStd": profile["azimuthErrorStd"],
        "omni:sensor:Core:elevationErrorMean": profile["elevationErrorMean"],
        "omni:sensor:Core:elevationErrorStd": profile["elevationErrorStd"],
        "omni:sensor:Core:maxReturns": profile["maxReturns"],
        "omni:sensor:Core:scanRateBaseHz": profile["scanRateBaseHz"],
        "omni:sensor:Core:reportRateBaseHz": profile["reportRateBaseHz"],
        "omni:sensor:Core:numberOfEmitters": profile["numberOfEmitters"],
        "omni:sensor:Core:numberOfChannels": profile["numberOfEmitters"],
        "omni:sensor:Core:intensityMappingType": profile["intensityMappingType"],
        "omni:sensor:Core:emitterState:s001:azimuthDeg": profile["emitters"]["azimuthDeg"],
        "omni:sensor:Core:emitterState:s001:elevationDeg": profile["emitters"]["elevationDeg"],
        "omni:sensor:Core:emitterState:s001:fireTimeNs": profile["emitters"]["fireTimeNs"],
    }
    return attributes


def _to_numpy(arr):
    """warp.array / torch.Tensor / numpy / list -> numpy.ndarray."""
    if hasattr(arr, "numpy"):
        try:
            return arr.numpy()
        except Exception:
            return arr.detach().cpu().numpy()
    return np.asarray(arr)


def lidar_points_in_sensor_frame(position_array):
    """Keep lidar points in the lidar sensor frame."""
    pts = np.asarray(position_array, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        pts = pts.reshape(-1, 3).astype(np.float32)
    return pts


def _create_point_cloud2(header, points):
    """Build a sensor_msgs/PointCloud2 from an (N,3) float32 array."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        pts = pts.reshape(-1, 3).astype(np.float32)
    xyz_points = [tuple(map(float, row)) for row in pts]
    return point_cloud2.create_cloud_xyz32(header, xyz_points)


def _to_ros_time(sim_time_s):
    sec = int(sim_time_s)
    nanosec = int((sim_time_s - sec) * 1_000_000_000)
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return sec, nanosec


def add_rtx_lidar(num_envs, robot_type, debug=False):
    annotators = []
    lidar_attributes = _load_l1_attributes()
    for i in range(num_envs):
        if robot_type == "g1":
            parent = f"/World/envs/env_{i}/Robot/head_link"
            translation = Gf.Vec3d(0.0, 0.0, 0.0)
        else:
            parent = f"/World/envs/env_{i}/Robot/base"
            translation = Gf.Vec3d(_GO2_LIDAR_X_M, _GO2_LIDAR_Y_M, _GO2_LIDAR_Z_M)

        if robot_type == "go2":
            orientation = Gf.Quatd(
                float(np.cos(_GO2_LIDAR_PITCH_RAD / 2.0)),
                0.0,
                float(np.sin(_GO2_LIDAR_PITCH_RAD / 2.0)),
                0.0,
            )
        else:
            orientation = Gf.Quatd(1.0, 0.0, 0.0, 0.0)

        _, lidar_prim = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path="/lidar_sensor",
            parent=parent,
            translation=translation,
            orientation=orientation,
            config=None,
            variant=None,
            force_camera_prim=False,
            **lidar_attributes,
        )
        lidar_path = f"{parent}/lidar_sensor"
        created_path = lidar_prim.GetPath().pathString
        if created_path != lidar_path:
            omni.kit.commands.execute("MovePrim", path_from=created_path, path_to=lidar_path)
            lidar_prim = omni.usd.get_context().get_stage().GetPrimAtPath(lidar_path)
        render_product = rep.create.render_product(lidar_prim.GetPath(), [1, 1], name="Isaac")
        annotator = rep.AnnotatorRegistry.get_annotator("IsaacCreateRTXLidarScanBuffer")
        annotator.attach(render_product)
        if debug:
            writer = rep.writers.get("RtxLidarDebugDrawPointCloudBuffer")
            writer.attach([render_product])
        annotators.append(annotator)
    return annotators


def add_camera(num_envs, robot_type):
    for i in range(num_envs):
        camera_cfg = CameraCfg(
            prim_path=f"/World/envs/env_{i}/Robot/base/front_cam",
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.32487, -0.00095, 0.05362),
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            ),
        )

        if robot_type == "g1":
            camera_cfg.prim_path = f"/World/envs/env_{i}/Robot/head_link/front_cam"
            camera_cfg.offset = CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            )

        Camera(camera_cfg)


def pub_robo_data_ros2(robot_type, num_envs, base_node, env, annotator_lst, start_time, sim_time_s):
    global _LAST_LIDAR_PUBLISH_S
    robot_data = env.unwrapped.scene["robot"].data
    joint_pos = _to_numpy(robot_data.joint_pos)
    root_state = _to_numpy(robot_data.root_state_w)
    lin_vel_b = _to_numpy(robot_data.root_lin_vel_b)
    ang_vel_b = _to_numpy(robot_data.root_ang_vel_b)

    if _LAST_LIDAR_PUBLISH_S is None:
        _LAST_LIDAR_PUBLISH_S = time.monotonic()

    base_node.publish_clock(sim_time_s)

    for i in range(num_envs):
        base_node.publish_joints(robot_data.joint_names, joint_pos[i], i, sim_time_s)
        base_node.publish_odom(root_state[i, :3], root_state[i, 3:7], i, sim_time_s)
        base_node.publish_imu(root_state[i, 3:7], lin_vel_b[i, :], ang_vel_b[i, :], i, sim_time_s)

        if robot_type == "go2":
            net_forces = _to_numpy(env.unwrapped.scene["contact_forces"].data.net_forces_w)
            base_node.publish_robot_state(
                [
                    net_forces[i][4][2],
                    net_forces[i][8][2],
                    net_forces[i][14][2],
                    net_forces[i][18][2],
                ],
                i,
            )

        try:
            if (time.monotonic() - _LAST_LIDAR_PUBLISH_S) >= _LIDAR_PUBLISH_PERIOD_S:
                for j in range(num_envs):
                    data = annotator_lst[j].get_data()
                    point_cloud = lidar_points_in_sensor_frame(data["data"])
                    base_node.publish_lidar(point_cloud, j, sim_time_s)
                _LAST_LIDAR_PUBLISH_S = time.monotonic()
        except Exception as exc:
            print(f"[go2_omniverse] lidar publish failed: {exc}", flush=True)
    return _LAST_LIDAR_PUBLISH_S


class RobotBaseNode(Node):
    def __init__(self, num_envs):
        super().__init__("go2_driver_node")
        qos_profile = QoSProfile(depth=10)

        self.joint_pub = []
        self.go2_state_pub = []
        self.go2_lidar_pub = []
        self.odom_pub = []
        self.imu_pub = []
        self.clock_pub = self.create_publisher(Clock, "/clock", qos_profile)

        for i in range(num_envs):
            self.joint_pub.append(self.create_publisher(JointState, f"robot{i}/joint_states", qos_profile))
            self.go2_state_pub.append(
                self.create_publisher(Float32MultiArray, f"robot{i}/foot_force", qos_profile)
            )
            lidar_topic = "/utlidar/cloud" if i == 0 else f"robot{i}/point_cloud2"
            odom_topic = "/utlidar/robot_odom" if i == 0 else f"robot{i}/odom"
            self.go2_lidar_pub.append(self.create_publisher(PointCloud2, lidar_topic, qos_profile))
            self.odom_pub.append(self.create_publisher(Odometry, odom_topic, qos_profile))
            self.imu_pub.append(self.create_publisher(Imu, f"robot{i}/imu", qos_profile))

    def _base_frame(self, robot_num):
        return "base_link" if robot_num == 0 else f"robot{robot_num}/base_link"

    def _lidar_frame(self, robot_num):
        return "utlidar_lidar" if robot_num == 0 else f"robot{robot_num}/utlidar_lidar"

    def publish_clock(self, sim_time_s):
        msg = Clock()
        sec, nanosec = _to_ros_time(sim_time_s)
        msg.clock.sec = sec
        msg.clock.nanosec = nanosec
        self.clock_pub.publish(msg)

    def publish_joints(self, joint_names_lst, joint_state_lst, robot_num, sim_time_s):
        joint_state = JointState()
        sec, nanosec = _to_ros_time(sim_time_s)
        joint_state.header.stamp.sec = sec
        joint_state.header.stamp.nanosec = nanosec
        joint_state.name = [f"robot{robot_num}/{n}" for n in joint_names_lst]
        joint_state.position = [float(v.item()) for v in joint_state_lst]
        self.joint_pub[robot_num].publish(joint_state)

    def publish_odom(self, base_pos, base_rot, robot_num, sim_time_s):
        base_frame = self._base_frame(robot_num)

        odom_topic = Odometry()
        sec, nanosec = _to_ros_time(sim_time_s)
        odom_topic.header.stamp.sec = sec
        odom_topic.header.stamp.nanosec = nanosec
        odom_topic.header.frame_id = "odom"
        odom_topic.child_frame_id = base_frame
        odom_topic.pose.pose.position.x = base_pos[0].item()
        odom_topic.pose.pose.position.y = base_pos[1].item()
        odom_topic.pose.pose.position.z = base_pos[2].item()
        odom_topic.pose.pose.orientation.x = base_rot[1].item()
        odom_topic.pose.pose.orientation.y = base_rot[2].item()
        odom_topic.pose.pose.orientation.z = base_rot[3].item()
        odom_topic.pose.pose.orientation.w = base_rot[0].item()
        self.odom_pub[robot_num].publish(odom_topic)

    def publish_imu(self, base_rot, base_lin_vel, base_ang_vel, robot_num, sim_time_s):
        imu_msg = Imu()
        sec, nanosec = _to_ros_time(sim_time_s)
        imu_msg.header.stamp.sec = sec
        imu_msg.header.stamp.nanosec = nanosec
        imu_msg.header.frame_id = self._base_frame(robot_num)
        imu_msg.linear_acceleration.x = base_lin_vel[0].item()
        imu_msg.linear_acceleration.y = base_lin_vel[1].item()
        imu_msg.linear_acceleration.z = base_lin_vel[2].item()
        imu_msg.angular_velocity.x = base_ang_vel[0].item()
        imu_msg.angular_velocity.y = base_ang_vel[1].item()
        imu_msg.angular_velocity.z = base_ang_vel[2].item()
        imu_msg.orientation.x = base_rot[1].item()
        imu_msg.orientation.y = base_rot[2].item()
        imu_msg.orientation.z = base_rot[3].item()
        imu_msg.orientation.w = base_rot[0].item()
        self.imu_pub[robot_num].publish(imu_msg)

    def publish_robot_state(self, foot_force_lst, robot_num):
        msg = Float32MultiArray()
        msg.data = [float(v.item()) for v in foot_force_lst]
        self.go2_state_pub[robot_num].publish(msg)

    def publish_lidar(self, points, robot_num, sim_time_s):
        header = Header(frame_id=self._lidar_frame(robot_num))
        sec, nanosec = _to_ros_time(sim_time_s)
        header.stamp.sec = sec
        header.stamp.nanosec = nanosec
        msg = _create_point_cloud2(header, points)
        self.go2_lidar_pub[robot_num].publish(msg)
