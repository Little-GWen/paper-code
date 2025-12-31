from logging import lastResort

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

# ================== 关键坐标定义 ==================
# 主路关键点
X_MAIN_START = 0        # a点: 主路起点
X_MERGE_START = 600    # b点: 汇入开始 (匝道接入点)
X_MERGE_END = 700     # c点: 汇入结束 (平行段结束)
X_DESTINATION = 1300    # d点: 最终终点
# 匝道关键点
X_RAMP_START = 400     # j点: 匝道起点
LEN_RAMP_CURVE = 50   # S形弯道长度 (决定了切入的缓急)
# 匝道弯道起点 (k点)
X_RAMP_CURVE_START = X_MERGE_START - LEN_RAMP_CURVE

# 校验几何合理性
if X_RAMP_CURVE_START < X_RAMP_START:
    raise ValueError(f"配置错误：匝道太短了！起点{X_RAMP_START} 晚于弯道起点{X_RAMP_CURVE_START}")

# ================== 增加状态特征 ==================
_original_to_dict = Vehicle.to_dict
def _new_to_dict(self, origin_vehicle=None, observe_intentions=True):
    """
    这是一个 Wrapper，用来拦截原始的 to_dict，并强行插入自定义数据
    """
    d = _original_to_dict(self, origin_vehicle, observe_intentions)
    distance_to_merge = X_MERGE_END - self.position[0]
    lane_id = self.lane_index[2]
    on_main_lane = (lane_id in [0, 1])      # 在主路
    if distance_to_merge < 0 or on_main_lane:
        d["distance_to_merge"] = 0.0        # 认为已汇入：归零
    else:
        d["distance_to_merge"] = distance_to_merge
    return d
Vehicle.to_dict = _new_to_dict


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
                "vehicles_count": 10,
                "features": ["x", "y", "vx", "vy", "heading", "distance_to_merge"],
                # 注意：这里环境通常会将观测值（Observation）归一化到 [-1, 1] 之间
                # 如果要修改，记得同时修改代码中的数值！
                "features_range": {
                    "x": [-500, 500], "y": [-40, 40], "vx": [-40, 40], "vy": [-40, 40], "distance_to_merge": [0, 700]
                },
                "absolute": False,
                "order": "sorted"       # 按照与自车的距离进行排序
            },
            "action": {"type": "DiscreteMetaAction"},
            "lanes_count": 2,
            "vehicles_count": 20,
            "initial_ego_speed": 20,
            "initial_traffic_speed": [20, 30],  # 随机生成这个速度范围的车辆
            "duration": 25,
            "collision_reward": -50.0,
            "cooldown_steps": 0,                # 换道冷却时间
            "frequent_lc_reward": 0,        # 频繁换道惩罚
            "target_speed": 30.0,               # 目标速度
            "offroad_terminal": False,
            #"simulation_frequency": 15,  # 物理仿真频率 (保持每秒15次)
            #"policy_frequency": 5,  # 策略决策频率 (每秒做5次决策，相当于

            "scaling": 4.0,     # 渲染缩放
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        # 汇入状态标志
        self.has_merged = False
        # 频繁换道记录
        self.steps_since_change = self.config["cooldown_steps"]     # 初始设为已冷却
        self.last_lane_index = self.vehicle.lane_index              # 记录初始车道
        self.frequent_change = False                                # 频繁换道标志

    def _create_road(self) -> None:
        """
            Make a road composed of a straight highway and a merging lane.
        """
        net = RoadNetwork()

        # ==================车道限速 ==================
        # Lane 0: 30 m/s (108 km/h)
        # Lane 1: 25 m/s (90  km/h)
        speed_limits = [25, 30]

        # Highway lanes
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_pre = [[c, s], [n, s]]
        line_merge = [[c, s], [n, s]]
        line_post = [[c, s], [n, s]]
        for i in range(2):
            net.add_lane("a", "b", StraightLane([X_MAIN_START, y[i]], [X_MERGE_START, y[i]],
                                                line_types=line_pre[i], speed_limit=speed_limits[i]))
            net.add_lane("b", "c", StraightLane([X_MERGE_START, y[i]], [X_MERGE_END, y[i]],
                                                line_types=line_merge[i], speed_limit=speed_limits[i]))
            net.add_lane("c", "d", StraightLane([X_MERGE_END, y[i]], [X_DESTINATION, y[i]],
                                                line_types=line_post[i], speed_limit=speed_limits[i]))

        # Merging lane
        amplitude = 3.25
        ramp_y = y[1] + 6.5 + 4
        # j->k (直行)
        ljk = StraightLane([X_RAMP_START, ramp_y], [X_RAMP_CURVE_START, ramp_y],
                           line_types=[c, c], forbidden=True)
        # k->b (S弯)
        lkb = SineLane(
            ljk.position(ljk.length, -amplitude),
            ljk.position(ljk.length + LEN_RAMP_CURVE, -amplitude),
            amplitude,
            2 * np.pi / (2 * LEN_RAMP_CURVE),
            np.pi / 2,
            line_types=[c, c], forbidden=True
        )
        # b->c (汇入)
        lbc = StraightLane(lkb.position(lkb.length, 0),
                           lkb.position(lkb.length, 0) + [X_MERGE_END - X_MERGE_START, 0],
                           line_types=[n, c], forbidden=True)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, lbc.position(X_MERGE_END - X_MERGE_START, 0)))
        self.road = road

    def _create_vehicles(self) -> None:
        """
            在合流场景中生成车辆
            自车：在匝道起点 (j -> k)
            他车：在主路起点 (a -> b)
        """
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        self.controlled_vehicles = []

        # 1. 生成自车
        ego_lane = self.road.network.get_lane(("j", "k", 0))
        controlled_vehicle = self.action_type.vehicle_class(
            self.road,
            position=ego_lane.position(0, 0),
            heading=ego_lane.heading_at(0),
            speed=self.config["initial_ego_speed"]
        )
        self.controlled_vehicles.append(controlled_vehicle)
        self.road.vehicles.append(controlled_vehicle)
        self.vehicle = self.controlled_vehicles[0]

        # 2. 生成背景交通
        vehicles_count = self.config["vehicles_count"]
        for _ in range(vehicles_count):
            for _ in range(20):  # retry
                # ================= 基于驾驶习惯的车道分配 =================
                lane_idx = self.np_random.choice([0, 1], p=[0.6, 0.4])
                lane = self.road.network.get_lane(("a", "b", lane_idx))
                x_pos = self.np_random.uniform(X_MAIN_START, X_MERGE_START)

                valid = True
                for v in self.road.vehicles:
                    if abs(v.position[0] - lane.position(x_pos, 0)[0]) < 15:  # 前后车辆间距不小于 15
                        valid = False
                        break

                if valid:
                    # ================= 基于车道的速度分配 =================
                    # lane_idx 0 -> 25~30 m/s (约 90~108 km/h)
                    # lane_idx 1 -> 20~25 m/s (约 72~90 km/h)
                    if lane_idx == 0:
                        spd = self.np_random.integers(25, 31)
                    elif lane_idx == 1:
                        spd = self.np_random.integers(20, 26)
                    else:
                        raise ValueError(f"Unexpected lane_idx: {lane_idx}")

                    veh = other_vehicles_type(
                        self.road,
                        position=lane.position(x_pos, 0),
                        heading=lane.heading_at(x_pos),
                        speed=spd
                    )
                    veh.randomize_behavior()
                    veh.target_speed = spd  # 锁定速度
                    self.road.vehicles.append(veh)
                    break

    def _reward(self, action: Action) -> float:
        # ======== 获取当前车辆状态 ========
        current_lane_index = self.vehicle.lane_index    # 当前车道
        last_lane_index = self.last_lane_index          # 上一时刻车道
        current_lane_id = current_lane_index[2]         # 当前车道ID
        on_main_lane = (current_lane_id in [0, 1])      # 在主路
        current_speed = self.vehicle.velocity[0]        # 当前车速
        target_speed = self.config["target_speed"]      # 目标车速

        # 是否有换道操作
        lane_changed = self.vehicle.lane_index[2] != self.last_lane_index[2]

        # =========== 奖励计算 ===========
        # 1. 碰撞惩罚
        if self.vehicle.crashed:
            return self.config["collision_reward"]

        # 2. 汇入奖励
        r_merged = 0
        # 汇入瞬间给予奖励
        if on_main_lane and not self.has_merged:
            r_merged = 10.0
            self.has_merged = True  # 标记为已汇入

        # 2. 行驶效率奖励
        if self.has_merged:
            # [主路巡航]：r_efficiency = r_speed - risk
            current_risk = self._calculate_risk(current_lane_index, mode="front")
            r_speed = 0.8 - np.abs(current_speed - target_speed) / 10     # 20m/s 时奖励为零
            r_efficiency = 2 * (r_speed - current_risk)   # 当 risk > 0.8 时，智能体开的再快也有惩罚
        else:
            # [匝道加速]：惩罚低速
            r_efficiency = -0.1 * max(0, 20 - current_speed)

        # 3. 换道奖励 & 惩罚
        r_lane_change = 0
        if lane_changed:
            if self.steps_since_change < self.config["cooldown_steps"]:
                # 频繁换道惩罚
                r_lane_change = self.config["frequent_lc_reward"]
                self.frequent_change = True
            else:
                # 正常换道
                # 换道收益 = 换道前的车道风险 - 当前车道的风险
                current_risk = self._calculate_risk(current_lane_index, mode="back")
                last_risk = self._calculate_risk(last_lane_index, mode="front")
                r_lane_change = last_risk - current_risk
                self.frequent_change = False

            # 只要换了道，就重置计时器
            self.steps_since_change = 0
            self.last_lane_index = self.vehicle.lane_index
        else:
            # 没换道，计时器增加
            self.steps_since_change += 1
            self.frequent_change = False

        return r_merged + r_efficiency + r_lane_change

    def _get_front_vehicle(self, target_lane_index) -> Vehicle:
        """
            获取指定 target_lane 车道上，位于自车前方且距离最近的车辆
        """
        if not self.vehicle: return None
        ego_x = self.vehicle.position[0]
        ego_y = self.vehicle.position[1]
        fronts = [
            v for v in self.road.vehicles
            if v is not self.vehicle
               and v.lane_index[2] == target_lane_index[2]
               and abs(v.position[1] - ego_y) < 5.0 # 防止主路和匝道使用重复的 id
               and v.position[0] > ego_x
        ]
        if fronts: return min(fronts, key=lambda v: v.position[0])
        return None

    def _get_back_vehicle(self, target_lane_index) -> Vehicle:
        """
            获取指定 target_lane 车道上，位于自车后方且距离最近的车辆
        """
        if not self.vehicle: return None
        ego_x = self.vehicle.position[0]
        ego_y = self.vehicle.position[1]
        backs = [
            v for v in self.road.vehicles
            if v is not self.vehicle
               and v.lane_index[2] == target_lane_index[2]
               and abs(v.position[1] - ego_y) < 5.0  # 防止主路和匝道使用重复的 id
               and v.position[0] < ego_x
        ]
        if backs: return max(backs, key=lambda v: v.position[0])
        return None

    def _get_ttc_risk(self, target_lane_index, mode="front"):
        """
            计算 TTC (Time-To-Collision) 风险
            公式:
                TTC = Distance / Closing_Speed
                TTC_Risk = Threshold / TTC
        """
        # 获取目标车道前车、后车
        front_vehicle = self._get_front_vehicle(target_lane_index)
        back_vehicle = self._get_back_vehicle(target_lane_index)

        # ---- 计算风险 ----
        front_ttc_risk = 0.0
        back_ttc_risk = 0.0
        # 前车 ttc 风险 (所有场景)
        if front_vehicle:
            dist_x = front_vehicle.position[0] - self.vehicle.position[0] - 5
            closing_speed = self.vehicle.velocity[0] - front_vehicle.velocity[0]
            if closing_speed > 1e-6:
                ttc_threshold = 8.0         # 同车道与侧车道阈值相同，方便对比
                ttc = max(dist_x, 0.1) / closing_speed
                front_ttc_risk = ttc_threshold / ttc
        # 后车 ttc 风险 (仅变道场景)
        if mode == "back":
            if back_vehicle:
                dist_x = self.vehicle.position[0] - back_vehicle.position[0] - 5
                closing_speed = back_vehicle.velocity[0] - self.vehicle.velocity[0]
                if closing_speed > 1e-6:
                    ttc_threshold = 5.0     # 变道稍微放宽一点
                    ttc = max(dist_x, 0.1) / closing_speed
                    back_ttc_risk = ttc_threshold / ttc

        ttc_risk = max(front_ttc_risk, back_ttc_risk)
        return ttc_risk

    def _get_dist_risk(self, target_lane_index, mode="front"):
        """
            计算 Headway Risk (车距保持风险)
            公式:
                Dist_isk = ((Safe_Dist - Dist) / k) ^ 2
        """
        # 获取目标车道前车、后车
        front_vehicle = self._get_front_vehicle(target_lane_index)
        back_vehicle = self._get_back_vehicle(target_lane_index)

        # ---- 计算风险 ----
        front_dist_risk = 0.0
        back_dist_risk = 0.0
        # 前车空间风险 (所有场景)
        if front_vehicle:
            dist_x = front_vehicle.position[0] - self.vehicle.position[0] - 5
            safe_dist_front = 30.0      # 跟车安全距离 (较长)
            if dist_x < safe_dist_front:
                intrusion = safe_dist_front - dist_x
                front_dist_risk = (intrusion / 5.0) ** 2

        # 后车空间风险 (仅变道场景)
        if mode == "back":
            if back_vehicle:
                dist_x = self.vehicle.position[0] - back_vehicle.position[0] - 5
                safe_dist_back = 15.0   # 变道适当放宽些
                if dist_x < safe_dist_back:
                    intrusion = safe_dist_back - dist_x
                    back_dist_risk = (intrusion / 5.0) ** 2

        dist_risk = max(front_dist_risk, back_dist_risk)
        return dist_risk

    def _calculate_risk(self, target_lane_index, mode="front"):
        """
            综合风险计算器
            输出: 0.0 (安全) ~ 1.0 (极度危险, tanh饱和)
        """
        # 1. 分别计算两种物理风险
        ttc_risk = self._get_ttc_risk(target_lane_index, mode)
        dist_risk = self._get_dist_risk(target_lane_index, mode)

        # 2. 取最大值作为当前车道的瓶颈风险
        raw_risk = max(ttc_risk, dist_risk)

        # 3. 归一化映射 (使用 tanh 防止数值无限大)
        # 0.1 是缩放系数，使得:
        # raw_risk = 5.0  -> tanh(0.5) = 0.46
        # raw_risk = 10.0 -> tanh(1.0) = 0.76
        # raw_risk = 20.0 -> tanh(2.0) = 0.96
        normalized_risk = np.tanh(raw_risk * 0.1)

        return normalized_risk

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _cost(self, action: int) -> float:
        return float(self.vehicle.crashed)

# 改用新名字，并添加防重复注册检查
env_id = 'my-merge-v0'
if env_id not in registry:
    register(
        id=env_id,
        entry_point='envs.custom_merge_env:MergeEnv',
    )
    #print(f"[CustomEnv] Successfully registered {env_id}")