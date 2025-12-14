import numpy as np
from gymnasium.envs.registration import register, registry
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Obstacle


class MergeEnv(AbstractEnv):
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
                    "x": [-1000, 1000],
                    "y": [-40, 40],
                    "vx": [-40, 40],
                    "vy": [-40, 40],
                    "heading": [-3.14159, 3.14159]
                },
                "absolute": False,
                "order": "sorted"
            },
            "action": {"type": "DiscreteMetaAction"},
            "lanes_count": 2,
            "vehicles_count": 20,
            "initial_ego_speed": 25,
            "initial_traffic_speed": [20, 30],
            "duration": 40,
            # [修改 1] 核弹级撞车惩罚，迫使 PPO 学会刹车
            "collision_reward": -500.0,
            "target_speed": 35.0,
            "offroad_terminal": False,
            "scaling": 5.0,
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        net = RoadNetwork()
        ends = [300, 200, 80, 1000]
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]
        for i in range(2):
            net.add_lane("a", "b", StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]))
            net.add_lane("b", "c",
                         StraightLane([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_types=line_type_merge[i]))
            net.add_lane("c", "d", StraightLane([sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]))

        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True)
        lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2 * ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.position(ends[1], 0), lkb.position(ends[1], 0) + [ends[2], 0],
                           line_types=[n, c], forbidden=True)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _create_vehicles(self) -> None:
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        self.controlled_vehicles = []
        ego_lane = self.road.network.get_lane(("j", "k", 0))
        controlled_vehicle = self.action_type.vehicle_class(
            self.road,
            position=ego_lane.position(0, 0),
            heading=ego_lane.heading_at(0),
            speed=self.config.get("initial_ego_speed", 25)
        )
        self.controlled_vehicles.append(controlled_vehicle)
        self.road.vehicles.append(controlled_vehicle)
        self.vehicle = self.controlled_vehicles[0]

        vehicles_count = self.config["vehicles_count"]
        real_merge_pos = 500
        for _ in range(vehicles_count):
            for _ in range(20):
                lane_idx = self.np_random.choice([0, 1], p=[0.5, 0.5])
                lane = self.road.network.get_lane(("a", "b", lane_idx))
                x_pos = self.np_random.uniform(0, lane.length)
                if (real_merge_pos - 50) < x_pos < (real_merge_pos + 50): continue
                valid = True
                for v in self.road.vehicles:
                    if np.linalg.norm(v.position - lane.position(x_pos, 0)) < 30:
                        valid = False
                        break
                if valid:
                    min_spd = self.config["initial_traffic_speed"][0]
                    max_spd = self.config["initial_traffic_speed"][1]
                    spd = self.np_random.integers(min_spd, max_spd + 1)
                    veh = other_vehicles_type(self.road, position=lane.position(x_pos, 0),
                                              heading=lane.heading_at(x_pos), speed=spd)
                    veh.randomize_behavior()
                    self.road.vehicles.append(veh)
                    break

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs, reward, terminated, truncated, info

    def _reward(self, action: Action) -> float:
        # [修改 2] 撞车熔断机制：撞车直接返回重罚，不计算其他奖励
        if self.vehicle.crashed:
            return self.config["collision_reward"]

        sigma = 5.0
        # 速度奖励
        r_speed = 2.0 * np.exp(- (self.vehicle.velocity[0] - self.config["target_speed"]) ** 2 / (2 * sigma ** 2))

        # 存活奖励
        r_survival = 0.2

        current_lane_index = self.vehicle.lane_index
        is_on_ramp = (current_lane_index[0] in ["j", "k"]) or (current_lane_index[2] == 2)
        is_merged = (not is_on_ramp) and (current_lane_index[0] in ["a", "b", "c", "d"])

        # 风险惩罚 (TTC)
        r_risk_penalty = 0.0
        if is_merged:
            r_risk_penalty = -3.0 * self._compute_risk_penalty()

        # 换道惩罚
        r_lane_change = 0.0
        if action == 2:
            r_lane_change = -1.0
        elif is_merged and action in [0, 2]:
            r_lane_change = -0.1
        else:
            r_lane_change = 0.5

        r_merged = 0.5 if is_merged else 0.0
        r_stuck = 0.0
        if is_on_ramp and self.vehicle.speed < 5:
            r_stuck = -2.0

        # 车距与相对速度惩罚
        r_headway = 0.0
        front_vehicle = self._get_front_vehicle()

        if front_vehicle:
            dist = np.linalg.norm(front_vehicle.position - self.vehicle.position)
            safe_margin = 30.0

            # 初始化变量，防止 UnboundLocalError
            r_dist = 0.0
            r_rel_speed = 0.0

            if dist < safe_margin:
                # A. 距离惩罚
                penalty_ratio = (safe_margin - dist) / safe_margin
                r_dist = -10.0 * penalty_ratio

                # B. [修改 3] 相对速度惩罚：系数从 -0.5 降为 -0.1
                # 这样它敢于加速，不会因为怕扣分而龟速行驶
                rel_speed = self.vehicle.speed - front_vehicle.speed
                if rel_speed > 0:
                    r_rel_speed = -0.1 * rel_speed

            # 合并惩罚，并在 -5.0 处截断
            r_headway = max(r_dist + r_rel_speed, -5.0)

        # 总和
        return r_headway + r_speed + r_survival + r_lane_change + r_risk_penalty + r_merged + r_stuck

    def _get_front_vehicle(self) -> Vehicle:
        vehicle = self.vehicle
        if not vehicle.lane: return None
        # 使用官方 API 获取前车，更准确
        front_vehicle, _ = self.road.neighbour_vehicles(vehicle, vehicle.lane_index)
        return front_vehicle

    def _compute_risk_penalty(self) -> float:
        risk_values = []
        for other in self.road.vehicles:
            if other is self.vehicle: continue
            delta_pos = other.position - self.vehicle.position
            rel_vel = other.velocity - self.vehicle.velocity
            dist = np.linalg.norm(delta_pos)
            dist = max(dist, 1e-6)
            dot_product = np.dot(delta_pos, rel_vel)
            closing_speed = -dot_product / dist
            if dist < 60.0:
                current_risk = 0.0
                if closing_speed > 0.05:
                    safe_dist = max(dist, 0.5)
                    ttc_risk = 8.0 * closing_speed / safe_dist
                    current_risk = max(current_risk, ttc_risk)
                dist_risk = 20.0 / max(dist, 0.5)
                current_risk = max(current_risk, dist_risk)
                risk_values.append(current_risk)
        if not risk_values: return 0.0
        max_risk = max(risk_values)
        normalized_risk = np.tanh(max_risk * 0.2)
        return normalized_risk

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _cost(self, action: int) -> float:
        return float(self.vehicle.crashed)


env_id = 'my-merge-v0'
if env_id not in registry:
    register(id=env_id, entry_point='custom_merge_env:MergeEnv')