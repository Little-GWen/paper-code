import numpy as np
from gymnasium.envs.registration import register, registry
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from sympy.series.gruntz import rewrite



class HighwayEnv(AbstractEnv):
    metadata = {'render_modes': ['human', 'rgb_array'],
                'render_fps': 15
    }

    def __init__(self, config: dict = None, render_mode: str = None):
        super().__init__(config, render_mode)
        self.reward_range = (-float('inf'), float('inf'))

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["x", "y", "vx", "vy", "heading"],
                # 注意：这里环境通常会将观测值（Observation）归一化到 [-1, 1] 之间
                # 如果要修改，记得同时修改代码中的数值！
                "features_range": {
                    "x": [-300, 300], "y": [-40, 40], "vx": [-40, 40], "vy": [-40, 40]
                },
                "absolute": False,
                "order": "sorted"    # 按照与自车的距离进行排序
            },
            "action": {"type": "DiscreteMetaAction"},
            "lanes_count": 4,
            "vehicles_count": 20,
            "traffic_spawn_length": 600,
            "initial_ego_speed": 25,
            "initial_traffic_speed": [20, 30],  # 随机生成这个速度范围的车辆
            "duration": 40,
            "collision_reward": -20.0,
            "target_speed": 30.0,
            "offroad_terminal": False,
            "simulation_frequency": 15,  # 物理仿真频率 (保持每秒15次)
            "policy_frequency": 5,  # 策略决策频率 (每秒做5次决策，相当于

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

        spawn_len = self.config.get("traffic_spawn_length", 400)

        for _ in range(self.config["vehicles_count"]):
            for _ in range(50):
                lid = self.np_random.integers(0, self.config["lanes_count"])
                x = self.np_random.uniform(0, spawn_len) + ego_pos - 50
                valid = True
                for v in self.road.vehicles:
                    if np.linalg.norm(v.position - [x, 0]) < 20:
                        valid = False
                        break
                if valid:
                    lane = self.road.network.get_lane(("0", "1", lid))
                    min_spd = self.config["initial_traffic_speed"][0]
                    max_spd = self.config["initial_traffic_speed"][1]
                    spd = self.np_random.integers(min_spd, max_spd + 1)
                    veh = other_vehicles_type(self.road, position=lane.position(x, 0), heading=lane.heading_at(x),
                                              speed=spd)
                    veh.randomize_behavior()
                    self.road.vehicles.append(veh)
                    break

    def _reward(self, action: Action) -> float:
        # 1. 碰撞惩罚
        if self.vehicle.crashed:
            return self.config["collision_reward"]

        # 2. 高速奖励（负向偏移线性）
        current_speed = self.vehicle.velocity[0]
        target_speed = self.config["target_speed"]
        r_speed = -np.abs(current_speed - target_speed) / target_speed

        # 3. 存活奖励
        r_survival = 0.5

        return r_speed + r_survival

    def _get_front_vehicle(self) -> Vehicle:
        if not self.vehicle.lane: return None
        ego_lane = self.vehicle.lane_index
        fronts = [v for v in self.road.vehicles if
                  v is not self.vehicle and v.lane_index == ego_lane and v.position[0] > self.vehicle.position[0]]
        if fronts: return min(fronts, key=lambda v: v.position[0] - self.vehicle.position[0])
        return None

    def _compute_risk_penalty(self) -> float:
        """
        完全复刻 Agent_GRPO.calculate_risk 的物理逻辑
        """
        risk_values = []

        # 1. 寻找周围车辆
        # 这里我们只关心前方的车，或者一定范围内的所有车
        for other in self.road.vehicles:
            if other is self.vehicle:
                continue

            # 1. 获取相对位置和相对速度 vector
            # delta_p = other_pos - ego_pos
            # 注意：Agent代码里是用 batch View 计算的相对位置，这里直接用物理坐标相减
            delta_pos = other.position - self.vehicle.position

            # rel_v = other_v - ego_v
            # 注意：Agent代码里 rel_vx 是 (other - ego)，所以这里保持一致
            # 但 Agent 代码里的 closing_speed 计算用了 -dot_product，
            # 意味着它定义 "靠近" 为正。
            rel_vel = other.velocity - self.vehicle.velocity

            # 2. 计算欧式距离 (dists)
            dist = np.linalg.norm(delta_pos)
            dist = max(dist, 1e-6)  # 防止除0

            # 3. 计算接近速度 (Closing Speed)
            # 投影: (pos · vel)
            # 如果 delta_pos 和 rel_vel 方向相反 (点积为负)，说明在靠近
            dot_product = np.dot(delta_pos, rel_vel)

            # closing_speed > 0 代表正在靠近
            closing_speed = -dot_product / dist

            # 4. 筛选逻辑 (Valid Mask)
            # 只有距离小于一定范围才计算风险 (模拟 Agent 的 mask)
            # 且只有正在靠近 (closing_speed > 0.05) 才计算 TTC
            if dist < 60.0:  # 视野范围，类似 Agent 的 Observation 范围
                current_risk = 0.0

                # A. TTC 风险 (TTC Threshold = 8.0 from Agent Code)
                if closing_speed > 0.05:
                    # TTC = dist / closing_speed
                    # Risk = 8.0 / TTC = 8.0 * closing_speed / dist
                    # 加上 min=0.5 保护
                    safe_dist = max(dist, 0.5)
                    ttc_risk = 8.0 * closing_speed / safe_dist
                    current_risk = max(current_risk, ttc_risk)

                # B. 绝对距离风险 (from Agent Code: 20.0 / dists)
                # 即使不靠近，贴太近也是危险
                dist_risk = 20.0 / max(dist, 0.5)
                current_risk = max(current_risk, dist_risk)

                risk_values.append(current_risk)

        # 如果视野内没车
        if not risk_values:
            return 0.0

        # 5. 取最大风险 (Max over neighbors)
        max_risk = max(risk_values)

        # 6. 归一化 (tanh + scaling from Agent Code)
        # Agent 代码: torch.tanh(max_total_risk * 0.2)
        normalized_risk = np.tanh(max_risk * 0.2)

        return normalized_risk

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _cost(self, action: int) -> float:
        return float(self.vehicle.crashed)

# 改用新名字，并添加防重复注册检查
env_id = 'my-highway-v0'
if env_id not in registry:
    register(
        id=env_id,
        entry_point='envs.custom_highway_env:HighwayEnv',
    )
    #print(f"[CustomEnv] Successfully registered {env_id}")