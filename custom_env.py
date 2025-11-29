import numpy as np
from gym.envs.registration import register
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


class HighwayEnv(AbstractEnv):
    """
    优化版 HighwayEnv (ROI + GRPO Aggressive):
    1. ROI策略: 物理生成15辆车，Obs只看5辆车。
    2. 激进奖励: 即使有高额碰撞惩罚，也通过超高速度奖励诱导Agent加速。
    """
    metadata = {'render.modes': ['human', 'rgb_array'], "video.frames_per_second": 15}

    def __init__(self, config: dict = None, render_mode: str = None):
        super().__init__(config, render_mode)
        self.reward_range = (-float('inf'), float('inf'))

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                # [关键变量 B] 神经网络输入维度
                # 5 = 1(Ego) + 4(Neighbors)
                # 作用：ROI (Region of Interest)，只关注身边最近的威胁，过滤远处的噪声
                "vehicles_count": 5,
                "features": ["x", "y", "vx", "vy", "heading"],
                # 移除手动 range，防止归一化截断
                "features_range": {
                    "x": [-100, 100], "y": [-100, 100], "vx": [-20, 20], "vy": [-20, 20]
                },
                "absolute": False,
                "order": "fixed"
            },
            "action": {"type": "DiscreteMetaAction"},
            "lanes_count": 4,

            # [关键变量 A] 物理世界生成车辆数
            # 15 = 相当拥堵。配合 short spawn length，制造高难度博弈环境
            "vehicles_count": 10,

            # [关键] 生成范围缩短，增加密度
            # 15辆车挤在300米内 = 真正的“泥石流”路况
            "traffic_spawn_length": 600,

            "initial_ego_speed": 25,
            "initial_traffic_speed": 20,
            "duration": 500,  # 缩短单局时间，加快采样节奏

            # [关键] 奖励重塑
            "collision_reward": -200.0,  # 只要撞车，直接判负
            "reward_speed_range": [20, 35],
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        self.road = Road(
            network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30, length=5000),
            np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        self.controlled_vehicles = []
        vehicle_class = self.action_type.vehicle_class
        if hasattr(vehicle_class, 'func'): vehicle_class = vehicle_class.func

        start_lane = 1
        ego_lane = self.road.network.get_lane(("0", "1", start_lane))
        ego_pos = 100
        controlled_vehicle = vehicle_class(
            self.road, position=ego_lane.position(ego_pos, 0),
            heading=ego_lane.heading_at(ego_pos), speed=self.config["initial_ego_speed"]
        )
        self.controlled_vehicles.append(controlled_vehicle)
        self.road.vehicles.append(controlled_vehicle)
        self.vehicle = self.controlled_vehicles[0]

        # [变量 A] 在这里生效：生成 15 辆车
        vehicles_to_create = self.config["vehicles_count"] - 1
        spawn_len = self.config.get("traffic_spawn_length", 400)

        for _ in range(vehicles_to_create):
            for _ in range(50):  # 尝试 50 次防止重叠
                lid = self.np_random.integers(0, self.config["lanes_count"])
                # 在 Ego 前后随机生成
                x = self.np_random.uniform(0, spawn_len) + ego_pos - 50
                valid = True
                for v in self.road.vehicles:
                    if np.linalg.norm(v.position - [x, 0]) < 10:  # 稍微允许紧凑一点
                        valid = False
                        break
                if valid:
                    lane = self.road.network.get_lane(("0", "1", lid))
                    # 随机速度，增加博弈
                    spd = 20 + self.np_random.uniform(-5, 5)
                    veh = other_vehicles_type(self.road, position=lane.position(x, 0), heading=lane.heading_at(x),
                                              speed=spd)
                    veh.randomize_behavior()
                    self.road.vehicles.append(veh)
                    break

    def _reward(self, action: Action) -> float:
        # 1. 碰撞惩罚 (重罚)
        if self.vehicle.crashed:
            return self.config["collision_reward"]

        # 2. 高速奖励 (GRPO 激进诱导)
        # 归一化到 [0, 1]
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])

        # [关键修改] 权重设为 5.0！
        # 告诉 Agent：如果你不全速跑，就算活着也没意义！
        r_speed = np.clip(scaled_speed, 0, 1) * 4.0

        # 3. 存活奖励 (降低低保)
        # 逼迫它动起来
        r_survival = 1

        # 4. 变道惩罚 (轻微惩罚，防止抽搐)
        r_lane_change = 0
        if action in [0, 2]:
            r_lane_change = -0.05

        # 5. 距离保持 (前车安全距离)
        r_dist = 0
        front_vehicle = self._get_front_vehicle()
        if front_vehicle:
            d = np.linalg.norm(self.vehicle.position - front_vehicle.position)
            # 安全距离阈值设为 30米
            if d < 30:
                # 距离越近扣分越多，最大扣 1.0
                r_dist = -1.0 * (1 - d / 30.0)

        return r_speed + r_survival + r_lane_change + r_dist

    def _get_front_vehicle(self) -> Vehicle:
        if not self.vehicle.lane: return None
        ego_lane = self.vehicle.lane_index
        fronts = [v for v in self.road.vehicles if
                  v is not self.vehicle and v.lane_index == ego_lane and v.position[0] > self.vehicle.position[0]]
        if fronts: return min(fronts, key=lambda v: v.position[0] - self.vehicle.position[0])
        return None

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _cost(self, action: int) -> float:
        return float(self.vehicle.crashed)


register(
    id='highway-v0',
    entry_point='custom_env:HighwayEnv',
)