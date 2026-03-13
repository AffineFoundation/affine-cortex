"""
ELO Rating System for Miner Scoring.

Implements Codeforces-style ELO with seniority bonus.
"""

import math
from typing import Dict, Optional, Tuple


def expected_losses(
    rating_i: float,
    other_ratings: list[float],
    D: float
) -> float:
    """计算矿工 i 的期望 losses 数。

    Args:
        rating_i: 矿工 i 的当前 rating
        other_ratings: 其他所有矿工的 rating 列表
        D: 积分差尺度参数

    Returns:
        期望 losses（有多少人排在 i 前面）
    """
    total = 0.0
    for r_j in other_ratings:
        # P(j beats i) = 1 / (1 + 10^((rating_i - rating_j) / D))
        # Clamp exponent to avoid OverflowError with 10^x for large x
        exponent = (rating_i - r_j) / D
        if exponent > 700:
            # 10^700 ≈ inf → P ≈ 0 (i much stronger than j)
            total += 0.0
        elif exponent < -700:
            # 10^(-700) ≈ 0 → P ≈ 1 (j much stronger than i)
            total += 1.0
        else:
            total += 1.0 / (1.0 + 10.0 ** exponent)
    return total


def compute_effective_k(
    K_base: float,
    rounds_played: int,
    K_provisional: float,
    provisional_rounds: int
) -> float:
    """计算有效 K 值（新矿工 provisional period）。

    Args:
        K_base: 基础 K 值
        rounds_played: 已参与轮数
        K_provisional: 新矿工 K 值
        provisional_rounds: provisional 期轮数

    Returns:
        有效 K 值
    """
    if rounds_played < provisional_rounds:
        # 线性衰减从 K_provisional 到 K_base
        t = rounds_played / provisional_rounds
        return K_provisional * (1 - t) + K_base * t
    return K_base


def compute_seniority_factor(
    age_i: int,
    age_j: int,
    alpha: float,
    i_beat_j: Optional[bool]
) -> float:
    """计算 seniority bonus 的 K 缩放因子。

    使用 2×sigmoid 使同龄时 factor=1.0（不影响 K），范围 (0, 2)。
    老模型赢新模型 → factor > 1（加更多分）
    老模型输新模型 → factor < 1（扣更少分）

    Args:
        age_i: 矿工 i 的模型年龄（current_block - submit_block）
        age_j: 矿工 j 的模型年龄
        alpha: 灵敏度参数
        i_beat_j: i 是否排名高于 j

    Returns:
        K 的缩放因子 (0, 2)，1.0 = 无影响
    """
    if alpha == 0 or i_beat_j is None:
        return 1.0
    age_diff = age_j - age_i  # 正 = j 比 i 老
    if i_beat_j:
        # i 赢了 j：如果 j 比 i 老（age_diff > 0），i 加分少
        exponent = alpha * age_diff
    else:
        # i 输给 j：如果 j 比 i 新（age_diff < 0），i 扣分少
        exponent = -alpha * age_diff
    # 防止 math.exp 溢出（exp(709) 是 float64 上限）
    exponent = max(min(exponent, 700.0), -700.0)
    return 2.0 / (1.0 + math.exp(exponent))


def update_ratings(
    round_ranks: Dict[int, int],
    current_ratings: Dict[int, float],
    current_rounds: Dict[int, int],
    model_ages: Dict[int, int],
    D: float,
    K_base: float,
    K_provisional: float,
    provisional_rounds: int,
    alpha: float,
) -> Dict[int, Tuple[float, float, int]]:
    """执行一轮 ELO 更新。

    Args:
        round_ranks: {uid: rank} 本轮排名（1 = 最高）
        current_ratings: {uid: rating} 当前积分
        current_rounds: {uid: rounds_played} 当前轮数
        model_ages: {uid: age_in_blocks} 模型年龄
        D: 积分差尺度
        K_base: 基础更新步长
        K_provisional: 新矿工更新步长
        provisional_rounds: provisional 期轮数
        alpha: seniority 灵敏度

    Returns:
        {uid: (new_rating, rating_change, new_rounds_played)}
    """
    uids = list(round_ranks.keys())
    results = {}

    for uid_i in uids:
        rank_i = round_ranks[uid_i]
        rating_i = current_ratings.get(uid_i, 1200.0)
        rounds_i = current_rounds.get(uid_i, 0)
        age_i = model_ages.get(uid_i, 0)

        K_eff = compute_effective_k(K_base, rounds_i, K_provisional, provisional_rounds)

        # 计算 rating 变化
        total_change = 0.0

        for uid_j in uids:
            if uid_j == uid_i:
                continue

            rank_j = round_ranks[uid_j]
            rating_j = current_ratings.get(uid_j, 1200.0)
            age_j = model_ages.get(uid_j, 0)

            # 期望结果（clamp exponent to avoid OverflowError）
            exp = (rating_i - rating_j) / D
            if exp > 700:
                p_j_beats_i = 0.0
            elif exp < -700:
                p_j_beats_i = 1.0
            else:
                p_j_beats_i = 1.0 / (1.0 + 10.0 ** exp)

            # 实际结果（处理平局：tied rank → actual=0.5）
            if rank_i < rank_j:
                actual = 0.0  # i 赢了 j
                i_beat_j = True
            elif rank_i == rank_j:
                actual = 0.5  # 平局
                i_beat_j = None
            else:
                actual = 1.0  # i 输给 j
                i_beat_j = False
            expected = p_j_beats_i

            # seniority 调整 K（平局时 factor=1.0，不施加年龄优势）
            if i_beat_j is None:
                sen_factor = 1.0
            else:
                sen_factor = compute_seniority_factor(age_i, age_j, alpha, i_beat_j)

            total_change += K_eff * sen_factor * (expected - actual)

        # Normalize by number of opponents (Codeforces convention):
        # use mean pairwise result instead of sum, so total change
        # is bounded by K regardless of field size.
        n_opponents = len(uids) - 1
        if n_opponents > 1:
            total_change /= n_opponents

        new_rating = max(rating_i + total_change, 0.0)  # 积分下限为 0
        rating_change = new_rating - rating_i  # 实际变化（考虑 floor 截断）
        new_rounds = rounds_i + 1

        results[uid_i] = (new_rating, rating_change, new_rounds)

    return results
