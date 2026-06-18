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
    import pygame
except Exception:
    pygame = None

try:
    from highway_env.vehicle.graphics import VehicleGraphics
except Exception:
    VehicleGraphics = None

try:
    from highway_env.road.graphics import RoadObjectGraphics
except Exception:
    RoadObjectGraphics = None


def _install_shade_area_graphics_patch() -> None:
    """
    highway-env에서는 Landmark도 RoadObjectGraphics로 그려지는 버전이 많다.
    그래서 LandmarkGraphics가 아니라 RoadObjectGraphics.display를 patch해서
    is_shade_area=True인 object만 투명 검은색 polygon으로 직접 그린다.

    이 방식이면 goal Landmark는 기존 파란색/초록색 표시를 그대로 유지하고,
    그늘 영역만 초록색 Landmark가 아니라 투명 검은색 overlay처럼 보인다.
    """
    if pygame is None or RoadObjectGraphics is None:
        return

    if getattr(RoadObjectGraphics, "_shade_area_patch_installed", False):
        return

    old_display = RoadObjectGraphics.display

    @classmethod
    def display(cls, object_, surface, transparent: bool = False, offscreen: bool = False):
        if getattr(object_, "is_shade_area", False):
            color = tuple(getattr(object_, "color", (0, 0, 0)))
            alpha = float(getattr(object_, "alpha", 0.30))
            alpha_i = int(np.clip(alpha, 0.0, 1.0) * 255)
            rgba = (int(color[0]), int(color[1]), int(color[2]), alpha_i)

            pos = np.asarray(object_.position, dtype=np.float32)
            heading = float(getattr(object_, "heading", 0.0))
            length = float(getattr(object_, "LENGTH", 1.0))
            width = float(getattr(object_, "WIDTH", 1.0))

            forward = np.array([np.cos(heading), np.sin(heading)], dtype=np.float32)
            lateral = np.array([-np.sin(heading), np.cos(heading)], dtype=np.float32)

            corners = [
                pos + 0.5 * length * forward + 0.5 * width * lateral,
                pos + 0.5 * length * forward - 0.5 * width * lateral,
                pos - 0.5 * length * forward - 0.5 * width * lateral,
                pos - 0.5 * length * forward + 0.5 * width * lateral,
            ]

            def world_to_pixel(v):
                if hasattr(surface, "vec2pix"):
                    return surface.vec2pix(v)
                if hasattr(surface, "pos2pix"):
                    return surface.pos2pix(float(v[0]), float(v[1]))
                return int(v[0]), int(v[1])

            points = [world_to_pixel(corner) for corner in corners]

            overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            pygame.draw.polygon(overlay, rgba, points, 0)
            surface.blit(overlay, (0, 0))
            return

        return old_display(object_, surface, transparent=transparent, offscreen=offscreen)

    RoadObjectGraphics.display = display
    RoadObjectGraphics._shade_area_patch_installed = True


_install_shade_area_graphics_patch()


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
            # 총 주차 차량 수: 일반 차량 8대 + 고가 차량 2대 = 총 10대
            # train.py에서 --n_vehicles 값을 넘기면 이 값이 덮어써질 수 있음
            "vehicles_count": 10,
            # use_all_empty_slots_as_goals=True이면 아래 값은 사실상 사용하지 않고,
            # 차량/기둥이 없는 모든 주차칸을 valid goal slot으로 사용함
            "empty_slot_count": 3,
            "use_all_empty_slots_as_goals": True,
            "add_walls": True,

            # Misaligned parked vehicle parameters
            "misalign_lateral_max": 0.45,
            "misalign_longitudinal_max": 0.35,
            "misalign_heading_max": np.deg2rad(10),

            # Pillar obstacles
            # 실제 주차장처럼 "주차칸 4개 + 기둥 1개" 패턴으로 배치
            # 기둥은 별도 랜덤 좌표가 아니라 parking slot 하나를 차지하도록 생성
            "pillar_count": 6,
            "pillar_size": 2.2,
            "parking_slots_before_pillar": 4,

            # Visual settings
            # 일반 주차 차량은 흰색, 고가 차량은 남색 계열로 표시
            "normal_vehicle_color": (245, 245, 245),
            "high_value_vehicle_color": (20, 35, 90),

            # Shade row visualization
            # 기존 로직과 동일하게 y >= shade_y_threshold인 row를 그늘로 처리
            "shade_y_threshold": 4.5,
            "show_shade_area": True,
            "shade_color": (0, 0, 0),
            "shade_visual_alpha": 0.30,
            "shade_visual_padding_x": 0.0,
            # 그늘 시각화 영역
            # wall 쪽은 그대로 붙이고, 통로 쪽 경계만 줄여서
            # 주차칸 깊이의 약 3/4 정도만 덮도록 함
            "shade_visual_padding_y": 0.0,
            "shade_visual_expand_to_walls": True,
            "shade_visual_coverage_ratio": 0.75,
            "shade_wall_inset": 0.15,

            # Boundary walls
            # highway-env 기본 wall이 렌더링/충돌에서 약하게 보일 때를 대비해
            # 직접 긴 Obstacle wall을 추가
            "add_custom_boundary_walls": True,
            "wall_thickness": 1.2,
            "wall_margin_x": 6.0,
            "wall_margin_y": 6.0,

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
        all_spots_raw = list(self.road.network.lanes_dict().keys())
        self.shade_visual_bounds = None

        # 그늘 영역은 시각화용 object이므로 차량/목표 생성 전에 먼저 추가
        # 실제 reward/state의 그늘 판정은 _shade_score_at()에서 동일하게 수행됨
        self._add_shade_visuals(all_spots_raw)

        # "주차칸 4개 + 기둥 1개" 패턴으로 기둥이 차지할 slot을 먼저 고름
        # 이 slot들은 차량/목표 후보에서 제외되어, 차와 기둥이 겹치지 않음
        self.pillar_spots = self._select_pillar_spots(all_spots_raw)

        all_spots = [
            spot for spot in all_spots_raw
            if spot not in set(self.pillar_spots)
        ]
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
        # 2. 주차 차량 / valid goal slot 선택
        # ----------------------------------------------------
        shuffled_candidates = candidate_spots.copy()
        self.np_random.shuffle(shuffled_candidates)

        use_all_empty_slots_as_goals = bool(
            self.config.get("use_all_empty_slots_as_goals", True)
        )

        if use_all_empty_slots_as_goals:
            # 먼저 차량이 차지할 slot을 고른다.
            # 나머지 non-pillar slot은 전부 valid goal slot이 된다.
            vehicle_count = min(
                int(self.config["vehicles_count"]),
                len(shuffled_candidates),
            )
            occupied_spots = shuffled_candidates[:vehicle_count]
            occupied_set = set(occupied_spots)

            # 차량/기둥이 없는 모든 주차칸을 성공 가능한 빈칸으로 사용
            empty_spots = [
                spot for spot in all_spots
                if spot not in occupied_set
            ]
        else:
            # 기존 방식: empty_slot_count개만 목표 후보로 사용
            empty_count = min(
                int(self.config["empty_slot_count"]),
                len(shuffled_candidates),
            )
            empty_spots = shuffled_candidates[:empty_count]
            empty_set = set(empty_spots)

            occupied_spots = [
                spot for spot in shuffled_candidates
                if spot not in empty_set
            ]

            occupied_spots = occupied_spots[: min(
                int(self.config["vehicles_count"]),
                len(occupied_spots),
            )]

        self.occupied_spots = occupied_spots
        self.valid_goal_spots = empty_spots

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

            # 시각화: 고가 차량은 남색 계열, 일반 차량은 흰색
            self._apply_parked_vehicle_color(parked_car, bool(is_high_value))

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
        # all-empty-goal 모드에서는 이 리스트 전체가 성공 가능한 주차칸이다.
        self.valid_goal_slot_infos = empty_slot_infos

        # ----------------------------------------------------
        # 6. 대표 goal slot 선택
        #    all-empty-goal 모드에서도 highway-env 내부 desired_goal 호환성과
        #    시각화용 marker를 위해 대표 goal 하나는 유지한다.
        #    실제 success 판정은 wrapper에서 모든 empty slot 기준으로 수행된다.
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

        # ----------------------------------------------------
        # 8. 주차장 외곽 boundary wall 추가
        # ----------------------------------------------------
        self._add_boundary_walls(all_spots_raw)

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

    def _apply_parked_vehicle_color(self, vehicle: Vehicle, is_high_value: bool) -> None:
        """
        주차된 차량 색상 지정.
        - 고가 차량: 남색 계열
        - 일반 차량: 흰색 계열

        highway-env 렌더러는 vehicle.color 속성을 사용하는 버전이 많아서
        여기에 직접 색을 넣어준다.
        """
        if is_high_value:
            vehicle.color = tuple(self.config.get("high_value_vehicle_color", (20, 35, 90)))
        else:
            vehicle.color = tuple(self.config.get("normal_vehicle_color", (245, 245, 245)))

    def _group_spots_by_row(self, all_spots: List[Tuple], row_group_tol: float = 1.0) -> List[List[Dict]]:
        """
        parking lane들을 y좌표 기준으로 row 단위로 묶고,
        각 row 내부는 x좌표 기준으로 정렬.
        """
        rows_dict: Dict[float, List[Dict]] = {}

        for lane_index in all_spots:
            lane = self.road.network.get_lane(lane_index)
            center = np.asarray(lane.position(lane.length / 2, 0), dtype=np.float32)
            heading = self._lane_heading(lane)

            y = float(center[1])
            y_key = round(y / row_group_tol) * row_group_tol

            rows_dict.setdefault(y_key, []).append({
                "lane_index": lane_index,
                "center": center,
                "heading": heading,
            })

        rows = []
        for _, row in rows_dict.items():
            row_sorted = sorted(row, key=lambda s: float(s["center"][0]))
            rows.append(row_sorted)

        rows = sorted(
            rows,
            key=lambda row: float(np.mean([s["center"][1] for s in row])),
        )
        return rows

    def _select_pillar_spots(self, all_spots: List[Tuple]) -> List[Tuple]:
        """
        "주차칸 4개 + 기둥 1개" 패턴을 만들기 위해
        각 row에서 특정 parking slot 자체를 pillar slot으로 예약한다.

        예: parking_slots_before_pillar = 4이면
            1,2,3,4번 칸은 일반 주차칸
            5번 칸은 기둥
            6,7,8,9번 칸은 일반 주차칸
            10번 칸은 기둥
        """
        pillar_count = int(self.config.get("pillar_count", 6))
        parking_slots_before_pillar = int(self.config.get("parking_slots_before_pillar", 4))

        if pillar_count <= 0:
            return []

        if parking_slots_before_pillar < 1:
            parking_slots_before_pillar = 4

        period = parking_slots_before_pillar + 1
        rows = self._group_spots_by_row(all_spots)

        row_pillar_spots: List[List[Tuple]] = []

        for row in rows:
            selected_for_row = []

            # index 기준으로 4칸 뒤의 5번째 칸을 기둥 slot으로 사용
            for k in range(parking_slots_before_pillar, len(row), period):
                selected_for_row.append(row[k]["lane_index"])

            if selected_for_row:
                row_pillar_spots.append(selected_for_row)

        # 한 row에 기둥이 몰리지 않도록 row별로 번갈아 선택
        pillar_spots: List[Tuple] = []
        if row_pillar_spots:
            max_len = max(len(spots) for spots in row_pillar_spots)
            for i in range(max_len):
                for spots in row_pillar_spots:
                    if i < len(spots):
                        pillar_spots.append(spots[i])

        return pillar_spots[:pillar_count]

    def _make_obstacle(
        self,
        position: np.ndarray | List[float],
        length: float,
        width: float,
        heading: float = 0.0,
        **attrs,
    ) -> Obstacle:
        """
        highway-env 버전에 따라 Obstacle 생성자의 heading 지원 여부가 다를 수 있어
        최대한 안전하게 생성하는 helper.
        """
        pos = np.asarray(position, dtype=np.float32)

        try:
            obstacle = Obstacle(self.road, pos, heading=heading)
        except TypeError:
            obstacle = Obstacle(self.road, pos)
            obstacle.heading = heading

        obstacle.LENGTH = float(length)
        obstacle.WIDTH = float(width)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH ** 2 + obstacle.WIDTH ** 2)

        for key, value in attrs.items():
            setattr(obstacle, key, value)

        return obstacle

    def _make_landmark(
        self,
        position: np.ndarray | List[float],
        length: float,
        width: float,
        heading: float = 0.0,
        **attrs,
    ) -> Landmark:
        """
        시각화용 Landmark 생성 helper.
        Landmark는 기본적으로 충돌 장애물이 아니므로 그늘 영역 표시용으로 사용한다.
        """
        pos = np.asarray(position, dtype=np.float32)

        try:
            landmark = Landmark(self.road, pos, heading=heading)
        except TypeError:
            landmark = Landmark(self.road, pos)
            landmark.heading = heading

        landmark.LENGTH = float(length)
        landmark.WIDTH = float(width)
        landmark.diagonal = np.sqrt(landmark.LENGTH ** 2 + landmark.WIDTH ** 2)
        landmark.COLLISIONS_ENABLED = False

        for key, value in attrs.items():
            setattr(landmark, key, value)

        return landmark

    def _add_pillars(self) -> None:
        """
        실제 주차장 느낌의 기둥 배치.

        기존 방식:
            - 임의 후보 좌표에서 랜덤 선택
            - 주차선 위에 점처럼 흩어진 느낌

        변경 방식:
            - parking slot 자체를 기둥이 차지하게 함
            - 각 row에서 "주차칸 4개 + 기둥 1개" 패턴
            - 기둥 slot은 차량/목표 후보에서 제외되므로 차와 겹치지 않음
        """
        pillar_size = float(self.config.get("pillar_size", 2.2))
        pillar_spots = list(getattr(self, "pillar_spots", []))

        if not pillar_spots:
            return

        for lane_index in pillar_spots:
            lane = self.road.network.get_lane(lane_index)
            center = np.asarray(lane.position(lane.length / 2, 0), dtype=np.float32)
            heading = self._lane_heading(lane)

            obstacle = self._make_obstacle(
                center,
                length=pillar_size,
                width=pillar_size,
                heading=heading,
                is_pillar=True,
            )
            self.road.objects.append(obstacle)

    def _add_shade_visuals(self, all_spots: List[Tuple]) -> None:
        """
        그늘 영역을 투명 검은색 overlay로 표시.

        이번 버전의 의도:
            - 벽 쪽 끝까지 닿는 느낌은 유지
            - 통로 쪽 경계는 안쪽으로 줄임
            - 즉, 주차장 절반 전체가 아니라 주차칸 깊이의 일부만 그늘로 덮음
            - 기본값은 wall 쪽에서부터 약 75%만 덮는 구조
        """
        if not bool(self.config.get("show_shade_area", True)):
            return

        if not all_spots:
            return

        shade_threshold = float(self.config.get("shade_y_threshold", 4.5))
        shade_color = tuple(self.config.get("shade_color", (0, 0, 0)))
        alpha = float(self.config.get("shade_visual_alpha", 0.30))
        padding_y = float(self.config.get("shade_visual_padding_y", 0.0))
        expand_to_walls = bool(self.config.get("shade_visual_expand_to_walls", True))
        wall_inset = float(self.config.get("shade_wall_inset", 0.15))
        coverage_ratio = float(self.config.get("shade_visual_coverage_ratio", 0.75))
        coverage_ratio = float(np.clip(coverage_ratio, 0.05, 1.0))

        centers = []
        for lane_index in all_spots:
            lane = self.road.network.get_lane(lane_index)
            centers.append(np.asarray(lane.position(lane.length / 2, 0), dtype=np.float32))

        centers = np.asarray(centers, dtype=np.float32)
        if centers.size == 0:
            return

        min_x, max_x = float(np.min(centers[:, 0])), float(np.max(centers[:, 0]))
        min_y, max_y = float(np.min(centers[:, 1])), float(np.max(centers[:, 1]))

        # shade_y_threshold를 기준으로 그늘 row를 고름.
        # 현재 환경은 한쪽 row만 그늘로 두는 구조라서 y >= threshold 쪽을 사용한다.
        shaded = centers[centers[:, 1] >= shade_threshold]
        if shaded.size == 0:
            return

        if expand_to_walls:
            # 외곽 벽 안쪽까지 x 방향은 넓게 확장
            margin_x = float(self.config.get("wall_margin_x", 6.0))
            margin_y = float(self.config.get("wall_margin_y", 6.0))
            wall_thickness = float(self.config.get("wall_thickness", 1.2))

            left = min_x - margin_x + 0.5 * wall_thickness + wall_inset
            right = max_x + margin_x - 0.5 * wall_thickness - wall_inset
            wall_side = max_y + margin_y - 0.5 * wall_thickness - wall_inset

            # 기존에는 bottom = shade_threshold - padding_y라서
            # 통로 쪽까지 너무 넓게 덮였다.
            # 이제는 wall_side에서 통로 방향으로 coverage_ratio만큼만 내려온다.
            aisle_side = shade_threshold + padding_y
            inner_boundary = wall_side - coverage_ratio * (wall_side - aisle_side)

            bottom = inner_boundary
            top = wall_side
        else:
            # fallback: 그늘 row 주변만 표시하되 row 깊이의 일부만 덮음
            padding_x = float(self.config.get("shade_visual_padding_x", 2.0))
            left = float(np.min(shaded[:, 0])) - padding_x
            right = float(np.max(shaded[:, 0])) + padding_x

            wall_side = float(np.max(shaded[:, 1])) + padding_y
            aisle_side = shade_threshold + padding_y
            inner_boundary = wall_side - coverage_ratio * (wall_side - aisle_side)

            bottom = inner_boundary
            top = wall_side

        if right <= left or top <= bottom:
            return

        # 이후 _shade_score_at()에서도 같은 그늘 영역을 쓰도록 저장한다.
        self.shade_visual_bounds = {
            "left": float(left),
            "right": float(right),
            "bottom": float(bottom),
            "top": float(top),
        }

        length = right - left
        width = top - bottom
        position = np.array([
            0.5 * (left + right),
            0.5 * (bottom + top),
        ], dtype=np.float32)

        shade = self._make_landmark(
            position,
            length=length,
            width=width,
            heading=0.0,
            is_shade_area=True,
            color=shade_color,
            alpha=alpha,
        )
        self.road.objects.append(shade)

    def _add_boundary_walls(self, all_spots: List[Tuple]) -> None:
        """
        차량이 주차장 밖으로 멀리 이탈하지 못하도록 외곽 벽을 추가.
        긴 사각형 Obstacle 4개를 주차장 외곽에 배치한다.
        """
        if not bool(self.config.get("add_custom_boundary_walls", True)):
            return

        if not all_spots:
            return

        centers = []
        for lane_index in all_spots:
            lane = self.road.network.get_lane(lane_index)
            centers.append(np.asarray(lane.position(lane.length / 2, 0), dtype=np.float32))

        centers = np.asarray(centers, dtype=np.float32)

        min_x, max_x = float(np.min(centers[:, 0])), float(np.max(centers[:, 0]))
        min_y, max_y = float(np.min(centers[:, 1])), float(np.max(centers[:, 1]))

        margin_x = float(self.config.get("wall_margin_x", 6.0))
        margin_y = float(self.config.get("wall_margin_y", 6.0))
        thickness = float(self.config.get("wall_thickness", 1.2))

        left = min_x - margin_x
        right = max_x + margin_x
        bottom = min_y - margin_y
        top = max_y + margin_y

        center_x = 0.5 * (left + right)
        center_y = 0.5 * (bottom + top)
        width_x = right - left
        height_y = top - bottom

        wall_specs = [
            # bottom wall
            ([center_x, bottom], width_x, thickness, 0.0),
            # top wall
            ([center_x, top], width_x, thickness, 0.0),
            # left wall
            ([left, center_y], height_y, thickness, np.pi / 2),
            # right wall
            ([right, center_y], height_y, thickness, np.pi / 2),
        ]

        for pos, length, width, heading in wall_specs:
            wall = self._make_obstacle(
                pos,
                length=length,
                width=width,
                heading=heading,
                is_boundary_wall=True,
            )
            self.road.objects.append(wall)

    def _shade_score_at(self, position: np.ndarray) -> float:
        """
        그늘 영역 계산.

        이전 버전은 단순히 y >= shade_y_threshold이면 전부 그늘로 처리했다.
        그러면 시각적으로 그늘 영역을 줄여도 state/reward에서는 여전히
        주차장 절반이 그늘로 인식될 수 있다.

        그래서 shade_visual_bounds가 있으면 실제 화면에 그려진
        투명 검은색 영역과 같은 범위를 그늘로 처리한다.
        """
        x, y = float(position[0]), float(position[1])

        bounds = getattr(self, "shade_visual_bounds", None)
        if bounds is not None:
            if (
                bounds["left"] <= x <= bounds["right"]
                and bounds["bottom"] <= y <= bounds["top"]
            ):
                return 1.0
            return 0.0

        shade_y_threshold = float(self.config.get("shade_y_threshold", 6.0))
        if y >= shade_y_threshold:
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
        use_all_empty_slots_as_goals: bool = True,
        pillar_count: int = 6,
        pillar_size: float = 2.2,
        parking_slots_before_pillar: int = 4,
        high_value_vehicle_count: int = 2,
        high_value_radius: float = 5.0,
        obstacle_dist_scale: float = 12.0,
        ray_dist_scale: float = 12.0,
        use_preference_goal: bool = True,
        high_value_penalty_coef: float = 0.3,
        smoothness_coef: float = 0.02,
        preference_bonus_scale: float = 20.0,
        success_slot_longitudinal_coverage: float = 0.75,
        success_slot_lateral_coverage: float = 0.75,
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
        # Success 판정용: ego 차량 중심이 주차칸 중앙 영역 안에 들어와야 한다.
        # 0.75는 각 slot의 중앙 75% 영역을 허용한다는 뜻이다.
        self.success_slot_longitudinal_coverage = float(success_slot_longitudinal_coverage)
        self.success_slot_lateral_coverage = float(success_slot_lateral_coverage)
        self._last_success_slot_metadata = None

        self._base_env = PreferenceParkingCoreEnv(
            config={
                "vehicles_count": n_other_vehicles,
                "empty_slot_count": empty_slot_count,
                "use_all_empty_slots_as_goals": use_all_empty_slots_as_goals,
                "pillar_count": pillar_count,
                "pillar_size": pillar_size,
                "parking_slots_before_pillar": parking_slots_before_pillar,
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
        self._last_success_slot_metadata = None

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
        raw_goal = raw_obs["desired_goal"].astype(np.float32)

        # all-empty-goal 모드에서는 하나의 고정 goal이 아니라
        # 현재 ego 차량에서 가장 가까운 빈 주차칸을 active goal로 사용한다.
        # 따라서 state의 goal/distance는 매 step 가장 가까운 valid empty slot 기준이 된다.
        active_goal_meta = self._active_goal_metadata(agent)

        if active_goal_meta is not None:
            goal_center = np.asarray(active_goal_meta["center"], dtype=np.float32)
            goal_heading = float(active_goal_meta.get("heading", 0.0))
            goal_vec = np.array([
                goal_center[0],
                goal_center[1],
                np.cos(goal_heading),
                np.sin(goal_heading),
            ], dtype=np.float32)
            goal_attrs = np.array([
                active_goal_meta.get("shade_score", 0.0),
                active_goal_meta.get("high_value_risk", 0.0),
            ], dtype=np.float32)
        else:
            # fallback: highway-env가 제공하는 대표 desired_goal 사용
            goal_vec = raw_goal[[0, 1, 4, 5]].astype(np.float32)
            goal_attrs = self._goal_slot_attributes()

        dx = float(goal_vec[0] - agent[0])
        dy = float(goal_vec[1] - agent[1])
        distance = float(np.sqrt(dx ** 2 + dy ** 2))

        ego_pos = np.array([agent[0], agent[1]], dtype=np.float32)

        current_shade = self._shade_score_at_position(ego_pos)
        current_high_value_risk = self._high_value_risk_at_position(ego_pos)

        nearest_obs_dist = self._nearest_obstacle_distance()
        ray_dists = self._directional_obstacle_distances(agent)

        nearest_hv_rel = self._nearest_high_value_relative(agent)

        obs = np.concatenate([
            agent,                                               # 6
            goal_vec,                                            # 4
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

    def _active_goal_metadata(self, agent: np.ndarray) -> Optional[Dict]:
        """
        현재 state/reward/success 기준이 되는 active goal slot을 반환.

        - use_all_empty_slots_as_goals=True:
            차량/기둥이 없는 모든 empty slot 중 ego와 가장 가까운 slot을 사용.
            따라서 어느 빈칸에 들어가도 success가 될 수 있다.
        - False:
            기존처럼 selected_slot_metadata 하나만 goal로 사용.
        """
        use_all = bool(
            getattr(self._base_env, "config", {}).get(
                "use_all_empty_slots_as_goals", True
            )
        )

        if not use_all:
            return getattr(self._base_env, "selected_slot_metadata", None)

        metas = getattr(self._base_env, "valid_goal_slot_infos", None)
        if metas is None:
            metas = getattr(self._base_env, "empty_slot_infos", [])

        if not metas:
            return getattr(self._base_env, "selected_slot_metadata", None)

        ego_pos = np.asarray([agent[0], agent[1]], dtype=np.float32)

        return min(
            metas,
            key=lambda meta: float(
                np.linalg.norm(np.asarray(meta["center"], dtype=np.float32) - ego_pos)
            ),
        )

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
        reward -= 0.05 # time penalty

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
        success_meta = self._success_slot_metadata(obs)
        if success_meta is not None:
            reward += 100.0

            # all-empty-goal 모드에서는 실제로 주차 성공한 slot의 속성을 사용한다.
            # fallback으로 observation의 active goal 속성을 사용한다.
            goal_shade = float(success_meta.get("shade_score", obs[29]))
            goal_high_value_risk = float(success_meta.get("high_value_risk", obs[30]))

            preference_score = (
                0.3 * goal_shade
                - 0.7 * goal_high_value_risk
            )

            reward += self.preference_bonus_scale * preference_score

        return float(reward)

    def _candidate_success_slots(self) -> List[Dict]:
        """
        Success 판정을 할 수 있는 주차칸 목록을 반환.

        use_all_empty_slots_as_goals=True이면 차량/기둥이 없는 모든 빈 주차칸,
        False이면 기존 selected goal 하나만 성공 후보로 사용한다.
        """
        use_all = bool(
            getattr(self._base_env, "config", {}).get(
                "use_all_empty_slots_as_goals", True
            )
        )

        if use_all:
            metas = getattr(self._base_env, "valid_goal_slot_infos", [])
            return list(metas)

        meta = getattr(self._base_env, "selected_slot_metadata", None)
        return [meta] if meta is not None else []

    def _ego_center_in_slot(self, meta: Dict) -> Tuple[bool, float, float]:
        """
        ego 차량 중심이 해당 parking slot의 내부/중앙 영역 안에 있는지 확인한다.

        단순히 goal center와의 거리만 보는 것이 아니라, highway-env lane의
        local coordinate를 사용해서 실제 slot rectangle 안에 들어왔는지 확인한다.

        Returns:
            inside_slot, longitudinal_error_from_center, lateral_error
        """
        ego = self._get_ego_vehicle()
        if ego is None or meta is None:
            return False, float("inf"), float("inf")

        lane_index = meta.get("lane_index", None)
        if lane_index is None:
            return False, float("inf"), float("inf")

        try:
            lane = self._base_env.road.network.get_lane(lane_index)
        except Exception:
            return False, float("inf"), float("inf")

        ego_pos = np.asarray(ego.position, dtype=np.float32)

        try:
            longitudinal, lateral = lane.local_coordinates(ego_pos)
        except Exception:
            # fallback: slot heading 기준으로 직접 body-coordinate 계산
            center = np.asarray(meta.get("center", [0.0, 0.0]), dtype=np.float32)
            heading = float(meta.get("heading", 0.0))
            forward = np.array([np.cos(heading), np.sin(heading)], dtype=np.float32)
            left = np.array([-np.sin(heading), np.cos(heading)], dtype=np.float32)
            rel = ego_pos - center
            longitudinal = float(np.dot(rel, forward)) + 0.5 * float(getattr(lane, "length", 5.0))
            lateral = float(np.dot(rel, left))

        slot_length = float(getattr(lane, "length", 5.0))
        slot_width = float(getattr(lane, "width", 2.5))

        center_longitudinal = 0.5 * slot_length
        longitudinal_error = float(longitudinal - center_longitudinal)
        lateral_error = float(lateral)

        longitudinal_coverage = float(np.clip(self.success_slot_longitudinal_coverage, 0.1, 1.0))
        lateral_coverage = float(np.clip(self.success_slot_lateral_coverage, 0.1, 1.0))

        # 중앙 영역 기준. coverage=0.75이면 slot 중앙 75% 영역에 ego center가 들어와야 함.
        longitudinal_limit = 0.5 * slot_length * longitudinal_coverage
        lateral_limit = 0.5 * slot_width * lateral_coverage

        inside_center_area = (
            abs(longitudinal_error) <= longitudinal_limit
            and abs(lateral_error) <= lateral_limit
        )

        # 혹시 중앙 영역 coverage가 커져도 slot 바깥은 success가 되지 않도록 raw bounds도 함께 확인.
        inside_raw_slot = (
            0.0 <= float(longitudinal) <= slot_length
            and abs(float(lateral)) <= 0.5 * slot_width
        )

        return bool(inside_center_area and inside_raw_slot), longitudinal_error, lateral_error

    def _success_slot_metadata(self, obs: np.ndarray) -> Optional[Dict]:
        """
        현재 ego 차량이 실제로 주차 성공한 빈 slot metadata를 반환한다.
        성공한 slot이 없으면 None을 반환한다.

        성공 조건:
            1. ego 차량 중심이 빈 주차칸 rectangle 내부, 더 정확히는 중앙 허용 영역 안에 있음
            2. 차량 heading이 해당 주차칸 heading과 정렬됨
            3. 차량 속도가 거의 0에 가까움
        """
        cos_h, sin_h = float(obs[4]), float(obs[5])
        speed = float(np.sqrt(float(obs[2]) ** 2 + float(obs[3]) ** 2))

        if speed >= self.SUCCESS_SPEED:
            self._last_success_slot_metadata = None
            return None

        best_meta = None
        best_score = float("inf")

        for meta in self._candidate_success_slots():
            inside_slot, longitudinal_error, lateral_error = self._ego_center_in_slot(meta)
            if not inside_slot:
                continue

            slot_heading = float(meta.get("heading", 0.0))
            gcos_h = float(np.cos(slot_heading))
            gsin_h = float(np.sin(slot_heading))
            heading_error = 1.0 - (cos_h * gcos_h + sin_h * gsin_h)

            if heading_error >= self.SUCCESS_HEADING:
                continue

            # 여러 slot 조건을 동시에 만족하는 드문 경우에는 더 중심에 가까운 slot 선택
            score = abs(longitudinal_error) + abs(lateral_error) + heading_error
            if score < best_score:
                best_score = score
                best_meta = meta

        self._last_success_slot_metadata = best_meta
        return best_meta

    def is_success(self, obs: np.ndarray, info: dict | None = None) -> bool:
        return self._success_slot_metadata(obs) is not None
    # ------------------------------------------------------------
    # Info for evaluation
    # ------------------------------------------------------------

    def _make_info(self, obs: np.ndarray) -> dict:
        meta = getattr(self._base_env, "selected_slot_metadata", None)

        distance = float(obs[12])
        cos_h, sin_h = float(obs[4]), float(obs[5])
        gcos_h, gsin_h = float(obs[8]), float(obs[9])
        heading_error = 1.0 - (cos_h * gcos_h + sin_h * gsin_h)

        success_meta = self._success_slot_metadata(obs)

        info = {
            "distance_to_active_goal": distance,
            "heading_error_to_active_goal": float(heading_error),
            "current_shade_score": float(obs[13]),
            "current_high_value_risk": float(obs[14]),
            "nearest_obstacle_distance": float(obs[15]),
            "nearest_high_value_dx_body": float(obs[24]),
            "nearest_high_value_dy_body": float(obs[25]),
            "nearest_high_value_dist": float(obs[26]),
            "success": bool(success_meta is not None),
            "success_slot_lane_index": str(success_meta.get("lane_index")) if success_meta is not None else None,
            "valid_goal_count": int(len(getattr(self._base_env, "valid_goal_slot_infos", []))),
            "use_all_empty_slots_as_goals": bool(
                getattr(self._base_env, "config", {}).get(
                    "use_all_empty_slots_as_goals", True
                )
            ),
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