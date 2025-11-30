import numpy as np
from gymnasium.envs.registration import register, registry
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

class HighwayEnv(AbstractEnv):
    """
    优化版 HighwayEnv:
    """
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
                "features_range": {
                    "x": [-100, 100], "y": [-100, 100], "vx": [-20, 20], "vy": [-20, 20]
                },                  # 注意：这里环境通常会将观测值（Observation）归一化到 [-1, 1] 之间
                "absolute": False,
                "order": "fixed"    # 保证 obs[0] 永远是 Ego Vehicle
            },
            "action": {"type": "DiscreteMetaAction"},
            "lanes_count": 4,
            "vehicles_count": 10,
            "traffic_spawn_length": 600,
            "initial_ego_speed": 20,        # 与车流同速，防止追尾
            "initial_traffic_speed": 20,
            "duration": 150,                # 减小 duration, 让智能体更容易活到最后, 建立正向反馈循环
            "collision_reward": -100.0,
            "reward_speed_range": [10, 30], # 让低速区也有梯度
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

        vehicles_to_create = self.config["vehicles_count"] - 1
        spawn_len = self.config.get("traffic_spawn_length", 400)

        for _ in range(vehicles_to_create):
            for _ in range(50):
                lid = self.np_random.integers(0, self.config["lanes_count"])
                x = self.np_random.uniform(0, spawn_len) + ego_pos - 50
                valid = True
                for v in self.road.vehicles:
                    if np.linalg.norm(v.position - [x, 0]) < 15:
                        valid = False
                        break
                if valid:
                    lane = self.road.network.get_lane(("0", "1", lid))
                    spd = 20 + self.np_random.uniform(-5, 5)
                    veh = other_vehicles_type(self.road, position=lane.position(x, 0), heading=lane.heading_at(x),
                                              speed=spd)
                    veh.randomize_behavior()
                    self.road.vehicles.append(veh)
                    break

    def _reward(self, action: Action) -> float:
        # 1. 碰撞惩罚
        if self.vehicle.crashed:
            return self.config["collision_reward"]

        # 2. 高速奖励
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        r_speed = np.clip(scaled_speed, 0, 1) * 1.0

        # 3. 存活奖励
        r_survival = 1.0  # 只要活着就一直加分

        # 4. 变道惩罚
        r_lane_change = 0
        if action in [0, 2]:
            r_lane_change = -0.1

        return r_speed + r_survival + r_lane_change

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

# 改用新名字，并添加防重复注册检查
env_id = 'my-highway-v0'
if env_id not in registry:
    register(
        id=env_id,
        entry_point='custom_env:HighwayEnv',
    )
    # print(f"[CustomEnv] Successfully registered {env_id}")