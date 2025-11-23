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
    基于动态约束 GRPO 算法的四车道高速公路环境 (修复版)
    """

    # --- [修复] 显式定义 metadata，防止 Gym 报错 ---
    metadata = {'render.modes': ['human', 'rgb_array'], "video.frames_per_second": 15}

    def __init__(self, config: dict = None, render_mode: str = None):
        super().__init__(config, render_mode)
        # --- [修复] 显式定义 reward_range ---
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
                },
                "absolute": False, "order": "sorted"
            },
            "action": {"type": "DiscreteMetaAction"},
            "lanes_count": 4,
            "vehicles_count": 25,
            "traffic_spawn_length": 600,
            "initial_ego_speed": 15,
            "initial_traffic_speed": 20,
            "traffic_speed_variance": 5,
            "duration": 600,
            "ego_spacing": 30,
            "vehicles_density": 1,
            "collision_reward": -300.0,
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        self.road = Road(
            network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30, length=3000),
            np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        self.controlled_vehicles = []
        vehicle_class = self.action_type.vehicle_class
        if hasattr(vehicle_class, 'func'): vehicle_class = vehicle_class.func

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

        vehicles_to_create = self.config["vehicles_count"] - 1
        min_spacing = 15
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
                if abs(lane_y - ego_y) < 2.0 and abs(x - ego_pos) < 20: continue

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
        r_survival = 0.2
        target_speed = 30
        speed_diff = abs(self.vehicle.speed - target_speed)
        r_speed = 1.0 * (1 - min(speed_diff, 30) / 30)
        r_low_speed = 0.0
        if self.vehicle.speed < 20: r_low_speed = -0.5 * (1 - self.vehicle.speed / 20)
        r_dist = 0
        front = self._get_front_vehicle()
        if front:
            d = np.linalg.norm(self.vehicle.position - front.position)
            if d < 25: r_dist = -0.5 * (25 - d) / 25.0
        r_compliance = 0.0
        if hasattr(self.vehicle, 'lane_index') and len(self.vehicle.lane_index) == 3:
            if self.vehicle.lane_index[2] >= 2 and self.vehicle.speed > 25: r_compliance = 0.2
        return r_speed + r_low_speed + r_survival + r_dist + r_compliance

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


# --- [关键修复] 移除 max_episode_steps ---
# 这样 Gym 就不会自动加 TimeLimit 包装器，避免了 attribute 缺失问题
# 时长控制完全由我们自己的 _is_truncated 函数接管
register(
    id='highway-v0',
    entry_point='custom_env:HighwayEnv',
)
register(
    id='highway-fast-v0',
    entry_point='custom_env:HighwayEnvFast',
)