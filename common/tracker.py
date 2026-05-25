from copy import deepcopy
from dataclasses import dataclass
import logging

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def depth_first_search(start_point, search_points):
    """Greedy nearest-neighbor ordering of search_points starting from start_point.

    Returns an array of ints: the order in which search_points are visited when
    repeatedly picking the closest unvisited point.
    """
    start_point = np.asarray(start_point, dtype=float)
    search_points = np.asarray(search_points, dtype=float)

    if start_point.shape != (3,):
        raise ValueError(f"start_point must have shape (3,), got {start_point.shape}")
    if search_points.ndim != 2 or search_points.shape[1] != 3:
        raise ValueError(
            f"search_points must have shape (N, 3), got {search_points.shape}"
        )

    num_points = search_points.shape[0]
    if num_points == 0:
        return np.array([], dtype=int)

    remaining = np.ones(num_points, dtype=bool)
    order = np.empty(num_points, dtype=int)
    current_point = start_point

    for i in range(num_points):
        remaining_idxs = np.flatnonzero(remaining)
        diffs = search_points[remaining_idxs] - current_point
        sq_distances = np.einsum("ij,ij->i", diffs, diffs)
        next_idx = int(remaining_idxs[np.argmin(sq_distances)])
        order[i] = next_idx
        remaining[next_idx] = False
        current_point = search_points[next_idx]

    return order


def remove_unlabeled_near_labeled(
    candidate_idxs: list[int],
    frame_data: dict,
    proximity_threshold_m: float = 0.02,
) -> list[int]:
    """Drop candidate idxs whose unlabeled marker is too close to any labeled marker.

    Inputs:
        candidate_idxs: indices into frame_data["unlabeled_markers"] to filter.
        frame_data: per-frame mocap dict with keys "unlabeled_markers" (list of (3,)
            lists in mm) and "labeled_markers" (dict[str, list of (3,) lists or None,
            in mm]).
        proximity_threshold_m: drop candidates within this many meters of any labeled
            marker.

    Returns: filtered candidate_idxs (input is not mutated).
    """
    labeled_positions = []
    for _, markers in frame_data["labeled_markers"].items():
        for marker in markers:
            if marker is not None:
                labeled_positions.append(np.array(marker, dtype=float) / 1000.0)
    if not labeled_positions:
        return list(candidate_idxs)
    labeled_positions = np.vstack(labeled_positions)
    raw_positions = np.array(frame_data["unlabeled_markers"]) / 1000.0
    too_close = np.any(
        cdist(labeled_positions, raw_positions) < proximity_threshold_m, axis=0
    )
    return [i for i in candidate_idxs if not too_close[i]]


def order_initial_frame(
    raw_marker_positions: np.ndarray,
    candidate_idxs: list[int],
    seed_position: np.ndarray,
    num_rope_markers: int,
) -> list[int]:
    """Order candidate markers in one frame by DFS from a seed position.

    Inputs:
        raw_marker_positions: (N, 3) array of all marker positions in the frame (meters).
        candidate_idxs: indices into raw_marker_positions of plausible rope candidates.
        seed_position: (3,) starting point for the DFS (typically handle-tip in world frame).
        num_rope_markers: rope length in markers; result is truncated to this many entries.

    Returns: list[int] of length num_rope_markers, ordered along the rope.
    """
    candidate_positions = np.asarray(raw_marker_positions)[candidate_idxs]
    marker_order = depth_first_search(seed_position, candidate_positions)[:num_rope_markers]
    return np.array(candidate_idxs)[marker_order].tolist()


@dataclass
class TrackBranch:
    overrides: dict
    close_pairs_prev: set
    close_ghost_pairs_prev: set
    cost: float


@dataclass
class TrackingResult:
    candidate_idxs_per_frame: list[list[int]]
    ordered_frames: list[bool]
    pruned_orderings_per_frame: list[list[list[int]]]
    success: bool
    ambiguous: bool


def track_markers(
    raw_marker_positions_per_frame: list[np.ndarray],
    candidate_idxs_per_frame: list[list[int]],
    start_idx: int,
    end_idx: int,
    num_rope_markers: int,
    marker_spacing_distance: float,
    proximity_threshold_m: float = 0.030,
    pruning_tolerance: float = 1.15,
    max_branches: int = 32,
) -> TrackingResult:
    """Multi-hypothesis tracking of ordered rope markers across frames.

    Inputs:
        raw_marker_positions_per_frame[t]: (N_t, 3) np.ndarray of all marker positions
            at frame t in meters.
        candidate_idxs_per_frame[t]: list of indices into raw_marker_positions_per_frame[t]
            that are plausible rope-marker candidates. Frames `start_idx` and `start_idx+1`
            must already be ordered (length == num_rope_markers, element k corresponds to
            the k-th rope marker along the rope).
        start_idx, end_idx: tracking range; ordering is produced for [start_idx, end_idx).
        num_rope_markers: number of rope markers tracked per frame.
        marker_spacing_distance: nominal rope-marker spacing in meters; used for pruning.
        proximity_threshold_m: distance below which two markers spawn a branch.
        pruning_tolerance: branches whose consecutive ordered spacing exceeds
            marker_spacing_distance * pruning_tolerance are pruned.
        max_branches: hard cap on simultaneous hypotheses.

    Returns a TrackingResult. On `success=False` the caller should not commit the result.
    """
    candidate_idxs_per_frame = deepcopy(candidate_idxs_per_frame)
    num_frames = len(candidate_idxs_per_frame)
    ordered_frames = [False] * num_frames
    ordered_frames[start_idx] = True
    ordered_frames[start_idx + 1] = True
    pruned_orderings_per_frame: list[list[list[int]]] = [[] for _ in range(num_frames)]

    branches = [
        TrackBranch(
            overrides={},
            close_pairs_prev=set(),
            close_ghost_pairs_prev=set(),
            cost=0.0,
        )
    ]

    for t in range(start_idx + 2, end_idx):
        raw_idxs_t = candidate_idxs_per_frame[t]
        raw_pos_t = raw_marker_positions_per_frame[t]
        new_branches = []

        for b in branches:
            idxs_tm1 = b.overrides.get(t - 1, candidate_idxs_per_frame[t - 1])
            idxs_tm2 = b.overrides.get(t - 2, candidate_idxs_per_frame[t - 2])
            last_markers = raw_marker_positions_per_frame[t - 1][idxs_tm1]
            last_last_markers = raw_marker_positions_per_frame[t - 2][idxs_tm2]
            if (
                last_markers.shape[0] != num_rope_markers
                or last_last_markers.shape[0] != num_rope_markers
            ):
                logging.info(f"t={t}: pruned branch (lost track in prior frame)")
                continue

            guess = last_markers + (last_markers - last_last_markers)
            cand_pos = raw_pos_t[raw_idxs_t]
            if cand_pos.shape[0] < num_rope_markers:
                logging.info(
                    f"t={t}: pruned branch (only {cand_pos.shape[0]} candidates)"
                )
                continue

            row_idx, col_idx = linear_sum_assignment(cdist(cand_pos, guess))
            selected_local = row_idx[np.argsort(col_idx)][:num_rope_markers]
            base_ordering = [raw_idxs_t[i] for i in selected_local]
            sel_pos = cand_pos[selected_local]
            selected_set = set(selected_local.tolist())
            ghost_locals = [
                i for i in range(len(raw_idxs_t)) if i not in selected_set
            ]

            def order_cost(ordering):
                return float(
                    np.linalg.norm(raw_pos_t[ordering] - guess, axis=1).sum()
                )

            tt_now = set()
            for i in range(num_rope_markers):
                for j in range(i + 1, num_rope_markers):
                    if np.linalg.norm(sel_pos[i] - sel_pos[j]) < proximity_threshold_m:
                        tt_now.add(frozenset((i, j)))

            tg_now = set()
            for k in range(num_rope_markers):
                for g_local in ghost_locals:
                    if np.linalg.norm(sel_pos[k] - cand_pos[g_local]) < proximity_threshold_m:
                        tg_now.add((k, raw_idxs_t[g_local]))

            new_branches.append(TrackBranch(
                overrides={**b.overrides, t: base_ordering},
                close_pairs_prev=tt_now,
                close_ghost_pairs_prev=tg_now,
                cost=b.cost + order_cost(base_ordering),
            ))

            for pair in tt_now - b.close_pairs_prev:
                i, j = tuple(pair)
                alt = list(base_ordering)
                alt[i], alt[j] = alt[j], alt[i]
                new_branches.append(TrackBranch(
                    overrides={**b.overrides, t: alt},
                    close_pairs_prev=tt_now,
                    close_ghost_pairs_prev=tg_now,
                    cost=b.cost + order_cost(alt),
                ))
                logging.info(
                    f"t={t}: branched on tracked-tracked pair (slots {i},{j})"
                )

            for (k, ghost_global) in tg_now - b.close_ghost_pairs_prev:
                alt = list(base_ordering)
                alt[k] = ghost_global
                new_branches.append(TrackBranch(
                    overrides={**b.overrides, t: alt},
                    close_pairs_prev=tt_now,
                    close_ghost_pairs_prev=tg_now,
                    cost=b.cost + order_cost(alt),
                ))
                logging.info(
                    f"t={t}: branched on tracked-ghost pair (slot {k}, ghost idx {ghost_global})"
                )

        survivors = []
        spacing_limit = marker_spacing_distance * pruning_tolerance
        for b in new_branches:
            ordering = b.overrides[t]
            pos = raw_pos_t[ordering]
            spacings = np.linalg.norm(np.diff(pos, axis=0), axis=1)
            if spacings.size == 0 or np.all(spacings <= spacing_limit):
                survivors.append(b)
            else:
                pruned_orderings_per_frame[t].append(list(ordering))
                violating = np.where(spacings > spacing_limit)[0]
                violations = ", ".join(
                    f"slots {i}-{i+1} (markers {ordering[i]},{ordering[i+1]})={spacings[i]:.4f}"
                    for i in violating
                )
                logging.info(
                    f"t={t}: pruned branch (max consecutive spacing "
                    f"{spacings.max():.4f} > {spacing_limit:.4f}); violations: {violations}"
                )

        seen = {}
        for b in survivors:
            seen.setdefault(tuple(b.overrides[t]), b)
        survivors = list(seen.values())

        if len(survivors) > max_branches:
            logging.warning(
                f"Branch cap hit at t={t}; truncating {len(survivors)}->{max_branches}"
            )
            survivors = survivors[:max_branches]
        if not survivors:
            logging.error(f"All branches pruned at t={t}; cannot complete tracking")
            return TrackingResult(
                candidate_idxs_per_frame=candidate_idxs_per_frame,
                ordered_frames=ordered_frames,
                pruned_orderings_per_frame=pruned_orderings_per_frame,
                success=False,
                ambiguous=False,
            )
        branches = survivors

    if not branches:
        logging.error("No surviving tracking branches")
        return TrackingResult(
            candidate_idxs_per_frame=candidate_idxs_per_frame,
            ordered_frames=ordered_frames,
            pruned_orderings_per_frame=pruned_orderings_per_frame,
            success=False,
            ambiguous=False,
        )
    ambiguous = len(branches) > 1
    if ambiguous:
        logging.error(
            f"Tracking ended in branched state ({len(branches)} surviving); "
            f"picking lowest-cost branch (costs: "
            f"{sorted(b.cost for b in branches)})"
        )
    chosen = min(branches, key=lambda b: b.cost)
    for t, ordering in chosen.overrides.items():
        candidate_idxs_per_frame[t] = list(ordering)
        ordered_frames[t] = True

    return TrackingResult(
        candidate_idxs_per_frame=candidate_idxs_per_frame,
        ordered_frames=ordered_frames,
        pruned_orderings_per_frame=pruned_orderings_per_frame,
        success=True,
        ambiguous=ambiguous,
    )
