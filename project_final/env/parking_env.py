"""
Custom Preference-Aware Parking Environment

Base:
    highway-env parking-v0

Main customizations:
    1. Dense parking lot with multiple empty candidate slots
    2. Randomly misaligned parked vehicles
       - lateral offset
       - longitudinal offset
       - heading offset
    3. Pillar obstacles
    4. Shade row
       - one parking row is treated as a shaded area
    5. High-value vehicles
       - only a few parked cars are high-value vehicles
       - the agent receives penalty when it enters the radius around them
    6. Flattened MDP state
       - vehicle state
       - goal state
       - goal-relative position
       - shade / high-value risk
       - nearest obstacle distance
       - 8-direction obstacle distances
       - nearest high-value vehicle relative position
       - previous action
       - selected goal slot attributes
    7. Custom reward
       - distance / progress / heading
       - collision / time / smoothness
       - near-obstacle penalty
       - high-value vehicle radius penalty
       - success bonus
       - terminal shade/high-value preference bonus
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401, registers highway-env environments

from highway_env.envs.parking_env import ParkingEnv as HighwayParkingEnv
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Landmark, Obstacle

try:
    from highway_env.vehicle.graphics import VehicleGraphics
except Exception:
    VehicleGraphics = None


# ============================================================
# 1. Core custom environment
#    - 실제 주차장 구조를 바꾸는 부분
# ============================================================

class PreferenceParkingCoreEnv(HighwayParkingEnv):
    """
    Custom core environment based on highway-env ParkingEnv.

    This class changes the physical scenario:
        - dense parked vehicles
        - misaligned parked vehicles
        - pillar obstacles
        - multiple empty candidate slots
        - selected target slot based on shade and high-value vehicle radius
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            # Parking lot layout
            "spots": 14,
            "vehicles_count": 10,
            "empty_slot_count": 3,
            "add_walls": True,

            # Misaligned parked vehicle parameters
            "misalign_lateral_max": 0.45,
            "misalign_longitudinal_max": 0.35,
            "misalign_heading_max": np.deg2rad(10),

            # Pillar obstacles
            "pillar_count": 4,
            "pillar_size": 0.8,

            # High-value vehicle setting
            "high_value_vehicle_count": 2,
            "high_value_radius": 5.0,

            # Preference-aware target selection
            # utility = w_shade * shade_score - w_high_value * high_value_risk
            "use_preference_goal": True,
            "w_shade": 0.7,
            "w_high_value": 0.3,

            # Simulation
            "duration": 200,
            "collision_reward": 0,
        })
        return config

    def _reset(self) -> None:
        self._create_road(spots=self.config["spots"])
        self._create_vehicles()

    def _create_vehicles(self) -> None:
        """
        Create:
            1. ego vehicle
            2. multiple empty candidate slots
            3. dense parked vehicles with misalignment
            4. a few high-value parked vehicles
            5. target goal landmark
            6. pillar obstacles

        This method replaces highway-env ParkingEnv._create_vehicles().
        """
        all_spots = list(self.road.network.lanes_dict().keys())
        self.np_random.shuffle(all_spots)

        self.controlled_vehicles = []
        self.slot_metadata: Dict[Tuple, Dict] = {}
        self.parked_vehicle_metadata: List[Dict] = []
        self.selected_slot_metadata: Optional[Dict] = None
        self.empty_slot_infos: List[Dict] = []
        self.occupied_spots: List[Tuple] = []

        # ----------------------------------------------------
        # 1. Ego vehicle 생성
        # ----------------------------------------------------
        ego_x = 0.0
        ego_y = 0.0
        ego_heading = 2.0 * np.pi * self.np_random.uniform()

        ego = self.action_type.vehicle_class(
            self.road,
            [ego_x, ego_y],
            ego_heading,
            0.0,
        )

        if VehicleGraphics is not None:
            ego.color = VehicleGraphics.EGO_COLOR

        self.road.vehicles.append(ego)
        self.controlled_vehicles.append(ego)

        # ego와 너무 가까운 주차칸은 후보에서 제외
        candidate_spots = []
        for lane_index in all_spots:
            lane = self.road.network.get_lane(lane_index)
            center = np.asarray(lane.position(lane.length / 2, 0), dtype=np.float32)
            if np.linalg.norm(center - np.asarray([ego_x, ego_y], dtype=np.float32)) > 5.0:
                candidate_spots.append(lane_index)

        if len(candidate_spots) < self.config["empty_slot_count"] + 1:
            candidate_spots = all_spots.copy()

        # ----------------------------------------------------
        # 2. 빈 주차칸 후보 선택
        # ----------------------------------------------------
        shuffled_candidates = candidate_spots.copy()
        self.np_random.shuffle(shuffled_candidates)

        empty_count = min(self.config["empty_slot_count"], len(shuffled_candidates))
        empty_spots = shuffled_candidates[:empty_count]

        occupied_spots = [
            spot for spot in shuffled_candidates
            if spot not in empty_spots
        ]

        occupied_spots = occupied_spots[: min(
            self.config["vehicles_count"],
            len(occupied_spots),
        )]

        self.occupied_spots = occupied_spots

        # ----------------------------------------------------
        # 3. 고가 차량으로 지정할 occupied spot 선택
        # ----------------------------------------------------
        high_value_count = min(
            int(self.config["high_value_vehicle_count"]),
            len(occupied_spots),
        )

        high_value_candidates = occupied_spots.copy()
        self.np_random.shuffle(high_value_candidates)
        high_value_spots = set(high_value_candidates[:high_value_count])

        # ----------------------------------------------------
        # 4. 주차된 차량 생성
        #    - 좌우 치우침
        #    - 앞뒤 치우침
        #    - 대각선 틀어짐
        # ----------------------------------------------------
        for lane_index in occupied_spots:
            lane = self.road.network.get_lane(lane_index)

            center = np.asarray(lane.position(lane.length / 2, 0), dtype=np.float32)
            heading = self._lane_heading(lane)

            shifted_position, shifted_heading = self._sample_misaligned_pose(
                center,
                heading,
            )

            parked_car = Vehicle(
                self.road,
                position=shifted_position,
                heading=shifted_heading,
                speed=0.0,
            )

            is_high_value = lane_index in high_value_spots
            parked_car.is_high_value = bool(is_high_value)

            self.road.vehicles.append(parked_car)

            self.parked_vehicle_metadata.append({
                "lane_index": lane_index,
                "position": shifted_position,
                "heading": shifted_heading,
                "is_high_value": bool(is_high_value),
            })

        # ----------------------------------------------------
        # 5. 빈칸 metadata 계산
        #    - shade_score
        #    - high_value_risk
        # ----------------------------------------------------
        empty_slot_infos = []

        for lane_index in empty_spots:
            lane = self.road.network.get_lane(lane_index)
            center = np.asarray(lane.position(lane.length / 2, 0), dtype=np.float32)
            heading = self._lane_heading(lane)

            shade_score = self._shade_score_at(center)
            high_value_risk = self._high_value_risk_at(center)

            utility = (
                self.config["w_shade"] * shade_score
                - self.config["w_high_value"] * high_value_risk
            )

            info = {
                "lane_index": lane_index,
                "center": center,
                "heading": heading,
                "shade_score": float(shade_score),
                "high_value_risk": float(high_value_risk),
                "utility": float(utility),
            }

            self.slot_metadata[lane_index] = info
            empty_slot_infos.append(info)

        self.empty_slot_infos = empty_slot_infos

        # ----------------------------------------------------
        # 6. 선호도 기반 goal slot 선택
        # ----------------------------------------------------
        if not empty_slot_infos:
            lane_index = candidate_spots[0]
            lane = self.road.network.get_lane(lane_index)
            selected_info = {
                "lane_index": lane_index,
                "center": np.asarray(lane.position(lane.length / 2, 0), dtype=np.float32),
                "heading": self._lane_heading(lane),
                "shade_score": 0.0,
                "high_value_risk": 0.0,
                "utility": 0.0,
            }
        elif self.config["use_preference_goal"]:
            selected_info = max(empty_slot_infos, key=lambda x: x["utility"])
        else:
            idx = int(self.np_random.integers(len(empty_slot_infos)))
            selected_info = empty_slot_infos[idx]

        self.selected_slot_metadata = selected_info

        # ego vehicle goal 설정
        for vehicle in self.controlled_vehicles:
            vehicle.goal = Landmark(
                self.road,
                selected_info["center"],
                heading=selected_info["heading"],
            )
            self.road.objects.append(vehicle.goal)

        # ----------------------------------------------------
        # 7. 기둥 obstacle 추가
        # ----------------------------------------------------
        self._add_pillars()

    def _lane_heading(self, lane) -> float:
        """
        highway-env 버전에 따라 lane heading을 얻는 방식이 다를 수 있어서
        robust하게 처리.
        """
        try:
            return float(lane.heading_at(lane.length / 2))
        except Exception:
            heading = getattr(lane, "heading", 0.0)
            if callable(heading):
                return float(heading(lane.length / 2))
            return float(heading)

    def _sample_misaligned_pose(
        self,
        slot_center: np.ndarray,
        slot_heading: float,
    ) -> Tuple[np.ndarray, float]:
        """
        주차칸 중심에서 조금 왼쪽/오른쪽, 앞/뒤로 치우치고
        heading도 약간 비틀어진 차량 pose 생성.
        """
        forward = np.array(
            [np.cos(slot_heading), np.sin(slot_heading)],
            dtype=np.float32,
        )
        lateral = np.array(
            [-np.sin(slot_heading), np.cos(slot_heading)],
            dtype=np.float32,
        )

        lateral_offset = self.np_random.uniform(
            -self.config["misalign_lateral_max"],
            self.config["misalign_lateral_max"],
        )
        longitudinal_offset = self.np_random.uniform(
            -self.config["misalign_longitudinal_max"],
            self.config["misalign_longitudinal_max"],
        )
        heading_offset = self.np_random.uniform(
            -self.config["misalign_heading_max"],
            self.config["misalign_heading_max"],
        )

        shifted_position = (
            slot_center
            + lateral_offset * lateral
            + longitudinal_offset * forward
        )

        shifted_heading = slot_heading + heading_offset

        return shifted_position.astype(np.float32), float(shifted_heading)

    def _add_pillars(self) -> None:
        """
        주차장 내부에 작은 고정 장애물 기둥 추가.
        원형 기둥 대신 작은 사각형 obstacle로 근사.
        """
        pillar_count = int(self.config["pillar_count"])
        pillar_size = float(self.config["pillar_size"])

        if pillar_count <= 0:
            return

        candidate_positions = [
            [-18.0, 0.0],
            [-6.0, 0.0],
            [6.0, 0.0],
            [18.0, 0.0],
            [-12.0, 4.5],
            [12.0, -4.5],
        ]

        self.np_random.shuffle(candidate_positions)

        for pos in candidate_positions[:pillar_count]:
            obstacle = Obstacle(self.road, pos)
            obstacle.LENGTH = pillar_size
            obstacle.WIDTH = pillar_size
            obstacle.diagonal = np.sqrt(obstacle.LENGTH ** 2 + obstacle.WIDTH ** 2)
            obstacle.is_pillar = True
            self.road.objects.append(obstacle)

    def _shade_score_at(self, position: np.ndarray) -> float:
        """
        그늘 영역 계산.
        단순화:
            - y >= 6.0인 한쪽 주차 row 전체를 그늘로 설정.
            - 필요하면 y <= -6.0으로 바꾸면 반대쪽 row가 그늘이 됨.
        """
        _, y = float(position[0]), float(position[1])

        if y >= 6.0:
            return 1.0

        return 0.0

    def _high_value_risk_at(self, position: np.ndarray) -> float:
        """
        고가 차량 반경 risk field.

        고가 차량과의 거리가 high_value_radius 안이면:
            risk = 1 - distance / high_value_radius

        반경 밖이면 risk = 0.
        """
        radius = float(self.config["high_value_radius"])
        if radius <= 0:
            return 0.0

        max_risk = 0.0

        for meta in self.parked_vehicle_metadata:
            if not meta.get("is_high_value", False):
                continue

            car_pos = np.asarray(meta["position"], dtype=np.float32)
            dist = float(np.linalg.norm(car_pos - position))

            if dist < radius:
                risk = 1.0 - dist / radius
                max_risk = max(max_risk, risk)

        return float(np.clip(max_risk, 0.0, 1.0))


# ============================================================
# 2. RL wrapper
#    - agent가 실제로 사용하는 MDP state/action/reward
# ============================================================

class ParkingEnv(gym.Env):
    """
    Wrapper around PreferenceParkingCoreEnv.

    Observation: 31-dim continuous vector

    Layout:
        agent:
            0  x
            1  y
            2  vx
            3  vy
            4  cos_h
            5  sin_h

        goal:
            6  goal_x
            7  goal_y
            8  goal_cos_h
            9  goal_sin_h

        relative goal:
            10 dx
            11 dy
            12 distance_to_goal

        current location attributes:
            13 current_shade_score
            14 current_high_value_risk

        obstacle:
            15 nearest_obstacle_distance
            16 front_dist
            17 rear_dist
            18 left_dist
            19 right_dist
            20 front_left_dist
            21 front_right_dist
            22 rear_left_dist
            23 rear_right_dist

        nearest high-value vehicle in ego-body coordinate:
            24 nearest_high_value_dx_body
            25 nearest_high_value_dy_body
            26 nearest_high_value_dist

        previous action:
            27 previous_steering
            28 previous_throttle

        selected goal slot attributes:
            29 goal_shade_score
            30 goal_high_value_risk
    """

    DISCRETE_ACTIONS = [
        (-1.0,  0.6), (-0.5,  0.6), (0.0,  0.6), (0.5,  0.6), (1.0,  0.6),
        (-1.0,  0.3), (-0.5,  0.3), (0.0,  0.3), (0.5,  0.3), (1.0,  0.3),
        (-1.0,  0.0),                 (0.0,  0.0),                 (1.0,  0.0),
        (-1.0, -0.3), (-0.5, -0.3), (0.0, -0.3), (0.5, -0.3), (1.0, -0.3),
        (-1.0, -0.6), (-0.5, -0.6), (0.0, -0.6), (0.5, -0.6), (1.0, -0.6),
    ]
    N_DISCRETE_ACTIONS = len(DISCRETE_ACTIONS)

    SUCCESS_DIST = 0.04
    SUCCESS_HEADING = 0.10
    SUCCESS_SPEED = 0.05

    def __init__(
        self,
        discrete: bool = False,
        noise_std: float = 0.02,
        max_steps: int = 200,
        n_other_vehicles: int = 10,
        empty_slot_count: int = 3,
        pillar_count: int = 4,
        high_value_vehicle_count: int = 2,
        high_value_radius: float = 5.0,
        obstacle_dist_scale: float = 12.0,
        ray_dist_scale: float = 12.0,
        use_preference_goal: bool = True,
        high_value_penalty_coef: float = 0.3,
        smoothness_coef: float = 0.02,
        preference_bonus_scale: float = 20.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.discrete = discrete
        self.noise_std = noise_std
        self.max_steps = max_steps
        self.obstacle_dist_scale = obstacle_dist_scale
        self.ray_dist_scale = ray_dist_scale
        self.high_value_penalty_coef = high_value_penalty_coef
        self.smoothness_coef = smoothness_coef
        self.preference_bonus_scale = preference_bonus_scale
        self.high_value_radius = high_value_radius

        self._base_env = PreferenceParkingCoreEnv(
            config={
                "vehicles_count": n_other_vehicles,
                "empty_slot_count": empty_slot_count,
                "pillar_count": pillar_count,
                "duration": max_steps,
                "high_value_vehicle_count": high_value_vehicle_count,
                "high_value_radius": high_value_radius,
                "use_preference_goal": use_preference_goal,
            },
            render_mode=render_mode,
        )

        self.observation_space = gym.spaces.Box(
            low=-np.ones(31, dtype=np.float32),
            high=np.ones(31, dtype=np.float32),
            dtype=np.float32,
        )

        if discrete:
            self.action_space = gym.spaces.Discrete(self.N_DISCRETE_ACTIONS)
        else:
            self.action_space = gym.spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )

        self._step_count = 0
        self._prev_dist = None
        self._prev_action = np.zeros(2, dtype=np.float32)

    # ------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        raw_obs, info = self._base_env.reset(seed=seed, options=options)
        self._step_count = 0
        self._prev_action = np.zeros(2, dtype=np.float32)

        true_obs = self._process_obs(raw_obs, add_noise=False)
        self._prev_dist = float(true_obs[12])

        obs = self._process_obs(raw_obs, add_noise=True)

        info.update(self._make_info(true_obs))
        return obs, info

    def step(self, action):
        if self.discrete:
            steering, throttle = self.DISCRETE_ACTIONS[int(action)]
            cont_action = np.array([steering, throttle], dtype=np.float32)
        else:
            cont_action = np.clip(action, -1.0, 1.0).astype(np.float32)

        raw_obs, _, terminated, truncated, info = self._base_env.step(cont_action)
        self._step_count += 1

        # reward/success는 true observation 기준
        true_obs = self._process_obs(raw_obs, add_noise=False)
        reward = self._compute_reward(true_obs, info, cont_action)

        # agent에게 제공되는 observation은 noisy observation
        obs = self._process_obs(raw_obs, add_noise=True)

        ego = self._get_ego_vehicle()
        crashed = bool(info.get("crashed", False))
        if ego is not None and hasattr(ego, "crashed"):
            crashed = crashed or bool(ego.crashed)

        info["crashed"] = crashed
        success = self.is_success(true_obs)

        failure_type = None

        if crashed:
            terminated = True
            failure_type = "collision"

        if success:
            terminated = True
            info["is_success"] = True
            failure_type = None

        if self._step_count >= self.max_steps:
            truncated = True
            if not success and not crashed:
                failure_type = "timeout"

        info.update(self._make_info(true_obs))
        info["failure_type"] = failure_type

        self._prev_action = cont_action.copy()

        return obs, reward, terminated, truncated, info

    def render(self):
        return self._base_env.render()

    def close(self):
        self._base_env.close()

    # ------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------

    def _process_obs(self, raw_obs: dict, add_noise: bool) -> np.ndarray:
        agent = raw_obs["observation"].astype(np.float32)
        goal = raw_obs["desired_goal"].astype(np.float32)

        dx = float(goal[0] - agent[0])
        dy = float(goal[1] - agent[1])
        distance = float(np.sqrt(dx ** 2 + dy ** 2))

        ego_pos = np.array([agent[0], agent[1]], dtype=np.float32)

        current_shade = self._shade_score_at_position(ego_pos)
        current_high_value_risk = self._high_value_risk_at_position(ego_pos)

        nearest_obs_dist = self._nearest_obstacle_distance()
        ray_dists = self._directional_obstacle_distances(agent)

        nearest_hv_rel = self._nearest_high_value_relative(agent)
        goal_attrs = self._goal_slot_attributes()

        obs = np.concatenate([
            agent,                                               # 6
            goal[[0, 1, 4, 5]],                                  # 4
            np.array([dx, dy, distance], dtype=np.float32),        # 3
            np.array([current_shade, current_high_value_risk], dtype=np.float32),  # 2
            np.array([nearest_obs_dist], dtype=np.float32),        # 1
            ray_dists.astype(np.float32),                         # 8
            nearest_hv_rel.astype(np.float32),                    # 3
            self._prev_action.astype(np.float32),                 # 2
            goal_attrs.astype(np.float32),                        # 2
        ]).astype(np.float32)

        if add_noise and self.noise_std > 0:
            # noise는 물리 관측과 거리 센서류에만 부여
            # shade/high-value risk scalar, previous action, goal preference metadata에는 noise를 주지 않음
            noise_indices = list(range(0, 13)) + list(range(15, 27))
            obs[noise_indices] += np.random.normal(
                0,
                self.noise_std,
                size=len(noise_indices),
            ).astype(np.float32)

        return np.clip(obs, -1.0, 1.0)

    def _goal_slot_attributes(self) -> np.ndarray:
        meta = getattr(self._base_env, "selected_slot_metadata", None)

        if meta is None:
            return np.array([0.0, 0.0], dtype=np.float32)

        return np.array([
            meta.get("shade_score", 0.0),
            meta.get("high_value_risk", 0.0),
        ], dtype=np.float32)

    # ------------------------------------------------------------
    # Obstacle / shade / high-value helpers
    # ------------------------------------------------------------

    def _get_ego_vehicle(self):
        env_unwrapped = self._base_env.unwrapped

        if hasattr(env_unwrapped, "controlled_vehicles") and env_unwrapped.controlled_vehicles:
            return env_unwrapped.controlled_vehicles[0]

        if hasattr(env_unwrapped, "vehicle"):
            return env_unwrapped.vehicle

        return None

    def _all_obstacle_positions(self) -> List[np.ndarray]:
        env_unwrapped = self._base_env.unwrapped
        positions = []

        if not hasattr(env_unwrapped, "road") or env_unwrapped.road is None:
            return positions

        ego = self._get_ego_vehicle()

        # Other vehicles
        for veh in getattr(env_unwrapped.road, "vehicles", []):
            if veh is ego:
                continue
            if hasattr(veh, "position"):
                positions.append(np.asarray(veh.position, dtype=np.float32))

        # Obstacles such as walls and pillars
        for obj in getattr(env_unwrapped.road, "objects", []):
            if isinstance(obj, Landmark):
                continue
            if hasattr(obj, "position"):
                positions.append(np.asarray(obj.position, dtype=np.float32))

        return positions

    def _nearest_obstacle_distance(self) -> float:
        ego = self._get_ego_vehicle()
        if ego is None:
            return 1.0

        ego_pos = np.asarray(ego.position, dtype=np.float32)
        obstacle_positions = self._all_obstacle_positions()

        if not obstacle_positions:
            return 1.0

        min_dist = min(float(np.linalg.norm(pos - ego_pos)) for pos in obstacle_positions)
        norm_dist = min_dist / max(self.obstacle_dist_scale, 1e-6)

        return float(np.clip(norm_dist, 0.0, 1.0))

    def _directional_obstacle_distances(self, agent: np.ndarray) -> np.ndarray:
        """
        8-direction obstacle distances in ego-centered coordinate.

        Output order:
            front, rear, left, right,
            front_left, front_right, rear_left, rear_right

        This is sector-based approximation, not exact LiDAR ray casting.
        """
        ego = self._get_ego_vehicle()
        if ego is None:
            return np.ones(8, dtype=np.float32)

        ego_pos = np.asarray(ego.position, dtype=np.float32)
        heading = float(getattr(ego, "heading", math.atan2(float(agent[5]), float(agent[4]))))

        forward = np.array([np.cos(heading), np.sin(heading)], dtype=np.float32)
        left = np.array([-np.sin(heading), np.cos(heading)], dtype=np.float32)

        sector_angles = np.array([
            0.0,
            np.pi,
            np.pi / 2,
            -np.pi / 2,
            np.pi / 4,
            -np.pi / 4,
            3 * np.pi / 4,
            -3 * np.pi / 4,
        ], dtype=np.float32)

        min_dists = np.ones(8, dtype=np.float32) * self.ray_dist_scale
        tolerance = np.pi / 8

        for obs_pos in self._all_obstacle_positions():
            rel = obs_pos - ego_pos
            dist = float(np.linalg.norm(rel))

            if dist < 1e-6:
                continue

            x_body = float(np.dot(rel, forward))
            y_body = float(np.dot(rel, left))
            angle = math.atan2(y_body, x_body)

            for i, center_angle in enumerate(sector_angles):
                diff = self._angle_diff(angle, float(center_angle))
                if abs(diff) <= tolerance:
                    min_dists[i] = min(min_dists[i], dist)

        norm = min_dists / max(self.ray_dist_scale, 1e-6)
        return np.clip(norm, 0.0, 1.0).astype(np.float32)

    def _shade_score_at_position(self, position: np.ndarray) -> float:
        if hasattr(self._base_env, "_shade_score_at"):
            return float(self._base_env._shade_score_at(position))
        return 0.0

    def _high_value_risk_at_position(self, position: np.ndarray) -> float:
        if hasattr(self._base_env, "_high_value_risk_at"):
            return float(self._base_env._high_value_risk_at(position))
        return 0.0

    def _nearest_high_value_relative(self, agent: np.ndarray) -> np.ndarray:
        """
        Return nearest high-value vehicle relative position in ego-body coordinate.

        Output:
            [dx_body_norm, dy_body_norm, dist_norm]

        If there is no high-value vehicle:
            [0, 0, 1]
        """
        ego = self._get_ego_vehicle()
        if ego is None:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)

        ego_pos = np.asarray(ego.position, dtype=np.float32)
        heading = float(getattr(ego, "heading", math.atan2(float(agent[5]), float(agent[4]))))

        forward = np.array([np.cos(heading), np.sin(heading)], dtype=np.float32)
        left = np.array([-np.sin(heading), np.cos(heading)], dtype=np.float32)

        metas = getattr(self._base_env, "parked_vehicle_metadata", [])
        high_value_positions = [
            np.asarray(meta["position"], dtype=np.float32)
            for meta in metas
            if meta.get("is_high_value", False)
        ]

        if not high_value_positions:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)

        nearest_pos = min(
            high_value_positions,
            key=lambda pos: float(np.linalg.norm(pos - ego_pos)),
        )

        rel = nearest_pos - ego_pos
        dist = float(np.linalg.norm(rel))

        dx_body = float(np.dot(rel, forward))
        dy_body = float(np.dot(rel, left))

        scale = max(float(self.high_value_radius), 1e-6)

        return np.array([
            np.clip(dx_body / scale, -1.0, 1.0),
            np.clip(dy_body / scale, -1.0, 1.0),
            np.clip(dist / scale, 0.0, 1.0),
        ], dtype=np.float32)

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        return (a - b + np.pi) % (2 * np.pi) - np.pi

    # ------------------------------------------------------------
    # Reward / success
    # ------------------------------------------------------------

    def _compute_reward(self, obs: np.ndarray, info: dict, action: np.ndarray) -> float:
        distance = float(obs[12])

        cos_h, sin_h = float(obs[4]), float(obs[5])
        gcos_h, gsin_h = float(obs[8]), float(obs[9])
        heading_error = 1.0 - (cos_h * gcos_h + sin_h * gsin_h)

        progress = 0.0
        if self._prev_dist is not None:
            progress = self._prev_dist - distance
        self._prev_dist = distance

        reward = 0.0

        # Dense reward
        reward -= 1.0 * distance
        reward += 2.0 * progress
        reward -= 0.3 * heading_error
        reward -= 0.1  # time penalty

        # Smoothness penalty
        action_change = float(np.linalg.norm(action - self._prev_action))
        reward -= self.smoothness_coef * action_change

        # Near-obstacle penalty
        nearest_obs_dist = float(obs[15])
        if nearest_obs_dist < 0.15:
            reward -= 0.5 * (0.15 - nearest_obs_dist)

        # High-value vehicle radius penalty
        current_high_value_risk = float(obs[14])
        reward -= self.high_value_penalty_coef * current_high_value_risk

        # Collision penalty
        if info.get("crashed", False):
            reward -= 50.0

        # Success bonus + terminal preference bonus
        if self.is_success(obs):
            reward += 100.0

            goal_shade = float(obs[29])
            goal_high_value_risk = float(obs[30])

            preference_score = (
                0.7 * goal_shade
                - 0.3 * goal_high_value_risk
            )

            reward += self.preference_bonus_scale * preference_score

        return float(reward)

    def is_success(self, obs: np.ndarray, info: dict | None = None) -> bool:
        distance = float(obs[12])

        cos_h, sin_h = float(obs[4]), float(obs[5])
        gcos_h, gsin_h = float(obs[8]), float(obs[9])
        heading_error = 1.0 - (cos_h * gcos_h + sin_h * gsin_h)

        speed = np.sqrt(float(obs[2]) ** 2 + float(obs[3]) ** 2)

        return (
            distance < self.SUCCESS_DIST
            and heading_error < self.SUCCESS_HEADING
            and speed < self.SUCCESS_SPEED
        )
    # ------------------------------------------------------------
    # Info for evaluation
    # ------------------------------------------------------------

    def _make_info(self, obs: np.ndarray) -> dict:
        meta = getattr(self._base_env, "selected_slot_metadata", None)

        distance = float(obs[12])
        cos_h, sin_h = float(obs[4]), float(obs[5])
        gcos_h, gsin_h = float(obs[8]), float(obs[9])
        heading_error = 1.0 - (cos_h * gcos_h + sin_h * gsin_h)

        info = {
            "distance_to_goal": distance,
            "heading_error": float(heading_error),
            "current_shade_score": float(obs[13]),
            "current_high_value_risk": float(obs[14]),
            "nearest_obstacle_distance": float(obs[15]),
            "nearest_high_value_dx_body": float(obs[24]),
            "nearest_high_value_dy_body": float(obs[25]),
            "nearest_high_value_dist": float(obs[26]),
            "success": bool(self.is_success(obs)),
        }

        if meta is not None:
            info.update({
                "goal_shade_score": float(meta.get("shade_score", 0.0)),
                "goal_high_value_risk": float(meta.get("high_value_risk", 0.0)),
                "goal_preference_utility": float(meta.get("utility", 0.0)),
            })

        # high-value 차량 위치도 평가/시각화용으로 저장
        metas = getattr(self._base_env, "parked_vehicle_metadata", [])
        high_value_positions = [
            np.asarray(meta["position"], dtype=np.float32).tolist()
            for meta in metas
            if meta.get("is_high_value", False)
        ]
        info["high_value_vehicle_positions"] = high_value_positions

        return info