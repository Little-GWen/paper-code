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
    基于动态约束GRPO算法的四车道高速公路环境 (优化版)
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,  # 观测 15 辆车 -> State Dim = 75
                "features": ["x", "y", "vx", "vy", "heading"],
                "features_range": {
                    "x": [-100, 100], "y": [-100, 100], "vx": [-20, 20], "vy": [-20, 20]
                },
                "absolute": False, "order": "sorted"
            },
            "action": {"type": "DiscreteMetaAction"},
            "lanes_count": 4,

            # --- [难度降低版：先学会跑，再加难度] ---
            "vehicles_count": 20,  # 40 -> 20 (降低密度)
            "traffic_spawn_length": 600,  # 400 -> 600 (拉大间距)
            "initial_ego_speed": 25,
            "initial_traffic_speed": 20,
            "traffic_speed_variance": 5,
            "duration": 500,
            "ego_spacing": 20,
            # -------------------------

            "vehicles_density": 1,
            "collision_reward": -50.0,
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        self.controlled_vehicles = []
        vehicle_class = self.action_type.vehicle_class
        if hasattr(vehicle_class, 'func'): vehicle_class = vehicle_class.func

        # 1. 生成自车
        start_lane = 1
        ego_lane = self.road.network.get_lane(("0", "1", start_lane))
        ego_pos = 50
        controlled_vehicle = vehicle_class(
            self.road, position=ego_lane.position(ego_pos, 0),
            heading=ego_lane.heading_at(ego_pos), speed=self.config["initial_ego_speed"]
        )
        self.controlled_vehicles.append(controlled_vehicle)
        self.road.vehicles.append(controlled_vehicle)
        self.vehicle = self.controlled_vehicles[0]

        # 2. 生成周围车辆
        vehicles_to_create = self.config["vehicles_count"] - 1
        min_spacing = 20
        spawn_len = self.config.get("traffic_spawn_length", 400)
        base_spd = self.config.get("initial_traffic_speed", 20)
        spd_var = self.config.get("traffic_speed_variance", 5)

        for _ in range(vehicles_to_create):
            for _ in range(50):
                lid = self.np_random.integers(0, self.config["lanes_count"])
                lane = self.road.network.get_lane(("0", "1", lid))
                x = self.np_random.uniform(0, spawn_len)

                lane_y = lane.position(0, 0)[1]
                ego_y = ego_lane.position(0, 0)[1]

                # 避开自车 (前后50米)
                if abs(lane_y - ego_y) < 2.0 and abs(x - ego_pos) < 50: continue

                # 避开其他车
                valid = True
                for v in self.road.vehicles:
                    if abs(v.position[1] - lane_y) < 2.0 and abs(v.position[0] - x) < min_spacing:
                        valid = False;
                        break

                if valid:
                    spd = base_spd + self.np_random.uniform(-spd_var, spd_var)
                    veh = other_vehicles_type(self.road, position=lane.position(x, 0), heading=lane.heading_at(x),
                                              speed=spd)
                    veh.randomize_behavior()
                    self.road.vehicles.append(veh)
                    break

    def _reward(self, action: Action) -> float:
        if self.vehicle.crashed: return self.config["collision_reward"]

        # 存活奖励 (关键！只要活着就给分，鼓励苟住)
        r_survival = 1.0

        # 速度奖励
        target_speed = 30
        speed_diff = abs(self.vehicle.speed - target_speed)
        r_speed = 0.5 * (1 - min(speed_diff, 30) / 30)  # 归一化到 0~0.5

        # 安全距离惩罚
        r_dist = 0
        front = self._get_front_vehicle()
        if front:
            d = np.linalg.norm(self.vehicle.position - front.position)
            if d < 25: r_dist = -0.5 * (25 - d) / 25.0  # 线性惩罚

        return r_speed + r_survival + r_dist

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
        return self.steps >= self.config["duration"]

    def _cost(self, action: int) -> float:
        return float(self.vehicle.crashed)


class HighwayEnvFast(HighwayEnv):
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({"simulation_frequency": 10, "lanes_count": 2, "vehicles_count": 10, "duration": 30})
        return cfg


register(id='highway-v0', entry_point='custom_env:HighwayEnv')
register(id='highway-fast-v0', entry_point='custom_env:HighwayEnvFast')