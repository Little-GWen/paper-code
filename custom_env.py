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
    优化版 HighwayEnv: 调整奖励函数以保证正向收益
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
            "initial_ego_speed": 25,  # 提高初始速度，让模型更容易跟上车流
            "initial_traffic_speed": 20,
            "duration": 1000,  # 增加最大步数
            "collision_reward": -50.0,  # [关键修改] 降低碰撞惩罚，避免分数过低
            "reward_speed_range": [20, 35],  # 速度奖励区间
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
        ego_pos = 100  # 初始位置稍微靠后一点，给点反应时间
        controlled_vehicle = vehicle_class(
            self.road, position=ego_lane.position(ego_pos, 0),
            heading=ego_lane.heading_at(ego_pos), speed=self.config["initial_ego_speed"]
        )
        self.controlled_vehicles.append(controlled_vehicle)
        self.road.vehicles.append(controlled_vehicle)
        self.vehicle = self.controlled_vehicles[0]

        # 简单的交通流生成逻辑
        vehicles_to_create = self.config["vehicles_count"] - 1
        spawn_len = self.config.get("traffic_spawn_length", 400)

        for _ in range(vehicles_to_create):
            # 尝试 50 次找到一个不重叠的位置
            for _ in range(50):
                lid = self.np_random.integers(0, self.config["lanes_count"])
                x = self.np_random.uniform(0, spawn_len) + ego_pos - 50

                # 简单的防碰撞检查
                valid = True
                for v in self.road.vehicles:
                    if np.linalg.norm(v.position - [x, 0]) < 15:  # 15米安全半径
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
        # 1. 碰撞惩罚 (Collision)
        if self.vehicle.crashed:
            return self.config["collision_reward"]

        # 2. 高速奖励 (High Speed Reward)
        # 归一化速度：(当前速度 - 20) / (30 - 20)。 20m/s以下得0分，30m/s以上得1分
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        r_speed = np.clip(scaled_speed, 0, 1)

        # 3. 存活奖励 (Survival Reward) - 鼓励活得久
        r_survival = 0.5

        # 4. 变道惩罚 (Lane Change Penalty) - 避免频繁变道
        r_lane_change = 0
        if action in [0, 2]:  # LANE_LEFT, LANE_RIGHT
            r_lane_change = -0.1

        # 5. 距离保持 (Safety Distance)
        r_dist = 0
        front_vehicle = self._get_front_vehicle()
        if front_vehicle:
            d = np.linalg.norm(self.vehicle.position - front_vehicle.position)
            if d < 25:  # 稍微增加安全距离阈值
                # 线性惩罚：距离越近扣分越多，最大扣 1.0
                r_dist = -1.0 * (1 - d / 25.0)

        # 总分公式
        reward = r_speed + r_survival + r_lane_change + r_dist
        return reward

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


# 注册环境
# [关键修复] 移除 max_episode_steps，防止旧版 gym 自动添加不兼容的 TimeLimit 包装器
# 我们的环境内部已经通过 _is_truncated 实现了时间限制逻辑
register(
    id='highway-v0',
    entry_point='custom_env:HighwayEnv',
)