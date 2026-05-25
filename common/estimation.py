from pydrake.all import (
    PiecewisePolynomial,
    BezierCurve,
    BsplineTrajectory,
    BezierCurve_,
    Expression,
    MathematicalProgram,
    OsqpSolver,
    QuadraticConstraint,
    SnoptSolver,
    Quaternion,
    RigidTransform,
    CameraInfo
)
import cv2
import itertools
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True)
class CalibrationBoardDetection:
    image_path: Path
    object_points: np.ndarray
    image_points: np.ndarray
    overlay: np.ndarray | None
    num_corners: int
    num_markers: int | None = None
    board_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class CalibrationBoardCandidate:
    label: str
    board_type: str
    size: tuple[int, int]
    payload: Any


def parse_grid(value: Any) -> tuple[int, int]:
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+)\s*[xX,]\s*(\d+)\s*", value)
        if match is None:
            raise ValueError(f"Could not parse grid value '{value}'. Expected like 9x12.")
        return int(match.group(1)), int(match.group(2))

    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])

    raise ValueError(f"Could not parse grid value '{value}'. Expected like 9x12.")


def aruco_dictionary_id(name: str, required_markers: int | None = None) -> int:
    normalized = name.strip().upper()
    normalized = normalized.replace("CV2.ARUCO.", "")
    normalized = normalized.replace("ARUCO_", "")
    normalized = normalized.replace("ARUCO-", "")
    if not normalized.startswith("DICT_"):
        normalized = f"DICT_{normalized}"

    if hasattr(cv2.aruco, normalized):
        return int(getattr(cv2.aruco, normalized))

    match = re.fullmatch(r"DICT_(4X4|5X5|6X6|7X7)", normalized)
    if match is not None:
        family = match.group(1)
        if required_markers is None:
            required_markers = 50
        for count in (50, 100, 250, 1000):
            if required_markers <= count:
                return int(getattr(cv2.aruco, f"DICT_{family}_{count}"))

    valid = sorted(name for name in dir(cv2.aruco) if name.startswith("DICT_"))
    raise ValueError(
        f"Unsupported ArUco dictionary '{name}'. Use one of OpenCV's names, "
        f"for example DICT_5X5_100. Available names include: {', '.join(valid[:8])}, ..."
    )


def make_charuco_board_candidates(config: dict[str, Any]) -> list[CalibrationBoardCandidate]:
    rows, cols = parse_grid(config["grid"])
    square_length = float(config["checker_size"] if "checker_size" in config else config["square_size"])
    marker_length = float(config["marker_size"])

    required_markers = int(np.ceil(rows * cols / 2.0))
    dictionary = cv2.aruco.getPredefinedDictionary(
        aruco_dictionary_id(str(config["dictionary"]), required_markers=required_markers)
    )

    candidates = []
    for label, size in (
        ("grid interpreted as rows x cols", (cols, rows)),
        ("grid interpreted as OpenCV x,y", (rows, cols)),
    ):
        board = cv2.aruco.CharucoBoard(size, square_length, marker_length, dictionary)
        candidates.append(
            CalibrationBoardCandidate(
                label=label,
                board_type="charuco",
                size=size,
                payload=cv2.aruco.CharucoDetector(board),
            )
        )

    return candidates


def make_checkerboard_candidates(config: dict[str, Any]) -> list[CalibrationBoardCandidate]:
    square_length = float(config["checker_size"] if "checker_size" in config else config["square_size"])
    grid_specs = []

    if "inner_corners" in config or "corners" in config:
        rows, cols = parse_grid(config.get("inner_corners", config.get("corners")))
        grid_specs.append(("inner corners", rows, cols))
    elif "squares" in config:
        rows, cols = parse_grid(config["squares"])
        grid_specs.append(("squares", rows - 1, cols - 1))
    elif "grid" in config:
        rows, cols = parse_grid(config["grid"])
        grid_specs.append(("inner corners", rows, cols))
        if rows > 1 and cols > 1:
            grid_specs.append(("squares", rows - 1, cols - 1))
    else:
        raise ValueError("Checkerboard config needs inner_corners, corners, squares, or grid.")

    candidates = []
    for convention, rows, cols in grid_specs:
        if rows <= 0 or cols <= 0:
            continue
        for label, pattern_size in (
            (f"grid interpreted as {convention}, rows x cols", (cols, rows)),
            (f"grid interpreted as {convention}, OpenCV x,y", (rows, cols)),
        ):
            object_points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
            object_points[:, :2] = (
                np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]]
                .T.reshape(-1, 2)
                .astype(np.float32)
            )
            object_points *= square_length
            candidates.append(
                CalibrationBoardCandidate(
                    label=label,
                    board_type="checkerboard",
                    size=pattern_size,
                    payload=object_points,
                )
            )

    if not candidates:
        raise ValueError(
            "Checkerboard config did not produce any positive-size corner patterns."
        )

    return candidates


def make_calibration_board_candidates(config: dict[str, Any]) -> list[CalibrationBoardCandidate]:
    board_type = str(config.get("board_type", "")).strip().lower().replace("_", "")
    if board_type == "charuco":
        return make_charuco_board_candidates(config)
    if board_type in ("checkerboard", "chessboard"):
        return make_checkerboard_candidates(config)
    raise ValueError(
        f"Unsupported board_type '{config.get('board_type')}'. "
        "Expected ChArUco or checkerboard."
    )


def load_grayscale_image(
    path: Path, expected_size: tuple[int, int] | None
) -> tuple[np.ndarray, tuple[int, int]]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image {path}")

    image_size = (int(image.shape[1]), int(image.shape[0]))
    if expected_size is not None and image_size != expected_size:
        raise ValueError(
            f"Image {path} has size {image_size}, expected {expected_size}. "
            "Use one calibration trial with consistent resolution."
        )
    return image, image_size


def detect_charuco_board(
    candidate: CalibrationBoardCandidate,
    image_path: Path,
    gray: np.ndarray,
    *,
    min_corners: int,
    make_overlay: bool,
) -> CalibrationBoardDetection | None:
    detector = candidate.payload
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    num_markers = 0 if marker_ids is None else int(len(marker_ids))
    if charuco_ids is None or charuco_corners is None:
        return None
    if len(charuco_ids) < min_corners:
        return None

    board = detector.getBoard()
    if board.checkCharucoCornersCollinear(charuco_ids):
        return None

    ids = charuco_ids.reshape(-1).astype(int)
    object_points = board.getChessboardCorners()[ids].astype(np.float32)
    image_points = charuco_corners.reshape(-1, 2).astype(np.float32)

    overlay = None
    if make_overlay:
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if marker_corners is not None and marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(overlay, marker_corners, marker_ids)
        cv2.aruco.drawDetectedCornersCharuco(overlay, charuco_corners, charuco_ids)

    return CalibrationBoardDetection(
        image_path=image_path,
        object_points=object_points,
        image_points=image_points,
        overlay=overlay,
        num_corners=int(len(charuco_ids)),
        num_markers=num_markers,
        board_size=candidate.size,
    )


def detect_checkerboard(
    candidate: CalibrationBoardCandidate,
    image_path: Path,
    gray: np.ndarray,
    *,
    min_corners: int,
    make_overlay: bool,
) -> CalibrationBoardDetection | None:
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    found, corners = cv2.findChessboardCornersSB(gray, candidate.size, flags)
    if not found or corners is None or len(corners) < min_corners:
        return None

    image_points = corners.reshape(-1, 2).astype(np.float32)
    object_points = candidate.payload.astype(np.float32)

    overlay = None
    if make_overlay:
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(overlay, candidate.size, corners, found)

    return CalibrationBoardDetection(
        image_path=image_path,
        object_points=object_points,
        image_points=image_points,
        overlay=overlay,
        num_corners=int(len(corners)),
        board_size=candidate.size,
    )


def detect_calibration_board(
    candidate: CalibrationBoardCandidate,
    image_path: Path,
    gray: np.ndarray,
    *,
    min_corners: int,
    make_overlay: bool,
) -> CalibrationBoardDetection | None:
    if candidate.board_type == "charuco":
        return detect_charuco_board(
            candidate,
            image_path,
            gray,
            min_corners=min_corners,
            make_overlay=make_overlay,
        )
    if candidate.board_type == "checkerboard":
        return detect_checkerboard(
            candidate,
            image_path,
            gray,
            min_corners=min_corners,
            make_overlay=make_overlay,
        )
    raise AssertionError(f"Unknown board type {candidate.board_type}")


def select_calibration_board_candidate(
    candidates: list[CalibrationBoardCandidate],
    image_paths: list[Path],
    *,
    min_corners: int,
    max_selection_images: int,
) -> CalibrationBoardCandidate:
    if len(image_paths) <= max_selection_images:
        sample_paths = image_paths
    else:
        indices = np.linspace(0, len(image_paths) - 1, max_selection_images, dtype=int)
        sample_paths = [image_paths[int(index)] for index in np.unique(indices)]
    best_candidate = None
    best_score = (-1, -1)
    image_size = None

    for candidate in candidates:
        detected_images = 0
        total_corners = 0
        for image_path in sample_paths:
            gray, image_size = load_grayscale_image(image_path, image_size)
            detection = detect_calibration_board(
                candidate,
                image_path,
                gray,
                min_corners=min_corners,
                make_overlay=False,
            )
            if detection is not None:
                detected_images += 1
                total_corners += detection.num_corners

        score = (detected_images, total_corners)
        if score > best_score:
            best_candidate = candidate
            best_score = score

    if best_candidate is None or best_score[0] == 0:
        raise RuntimeError("No calibration board was detected in the sample images.")

    return best_candidate


def put_status_text(overlay: np.ndarray, text: str) -> np.ndarray:
    output = overlay.copy()
    cv2.putText(
        output,
        text,
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return output


def show_overlay(window_name: str, overlay: np.ndarray, delay_ms: int) -> bool:
    cv2.imshow(window_name, overlay)
    key = cv2.waitKey(delay_ms) & 0xFF
    return key not in (ord("q"), 27)


def collect_calibration_board_detections(
    candidate: CalibrationBoardCandidate,
    image_paths: list[Path],
    *,
    min_corners: int,
    visualize: bool,
    visualization_delay_ms: int,
) -> tuple[list[CalibrationBoardDetection], list[dict[str, Any]], tuple[int, int]]:
    detections = []
    rejected = []
    image_size = None
    keep_showing = visualize
    window_name = "calibration board detections"

    for image_path in image_paths:
        gray, image_size = load_grayscale_image(image_path, image_size)
        detection = detect_calibration_board(
            candidate,
            image_path,
            gray,
            min_corners=min_corners,
            make_overlay=visualize,
        )

        if detection is None:
            rejected.append({"image": image_path.name, "reason": "board not detected"})
            if keep_showing:
                overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                overlay = put_status_text(overlay, f"{image_path.name}: no detection")
                keep_showing = show_overlay(window_name, overlay, visualization_delay_ms)
            continue

        detections.append(detection)
        if keep_showing and detection.overlay is not None:
            text = f"{image_path.name}: {detection.num_corners} corners"
            keep_showing = show_overlay(
                window_name,
                put_status_text(detection.overlay, text),
                visualization_delay_ms,
            )

    if visualize:
        cv2.destroyAllWindows()

    if image_size is None:
        raise RuntimeError("No images were loaded.")

    return detections, rejected, image_size


def config_to_camerainfo(config):
    assert not ("fov_y" in config and "fx" in config)

    if "fov_y" in config:
        return CameraInfo(config["width"], config["height"], config["fov_y"])
    elif "fx" in config:
        return CameraInfo(
            config["width"],
            config["height"],
            config["fx"],
            config["fy"],
            config["cx"],
            config["cy"],
        )
    else:
        return None


def project_world_points_to_image(
    points_W: np.ndarray, K: np.ndarray, X_WC: RigidTransform
) -> np.ndarray:
    X_CW_matrix = X_WC.inverse().GetAsMatrix4()
    points_W_hom = np.column_stack((points_W, np.ones(points_W.shape[0])))
    points_C = (X_CW_matrix @ points_W_hom.T).T[:, :3]
    pixels_hom = (K @ points_C.T).T
    return pixels_hom[:, :2] / pixels_hom[:, 2:3]


def binary_mask_to_rgb(mask_frame: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask_frame)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    rgb[mask > 0] = np.array([255, 255, 255], dtype=np.uint8)
    return rgb


def thin_binary_image(binary_image: np.ndarray) -> np.ndarray:
    return cv2.ximgproc.thinning(binary_image)


@dataclass
class MocapRopeTrackPoints:
    track_points: np.ndarray # (N, 3) 3D points along the rope
    arc_length_dists: list[float] # (N,) Distance along the rope for each value in track_points


def _as_binary_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3:
        image = np.any(image > 0, axis=2)
    return (image > 0).astype(np.uint8) * 255


def _image_point_to_row_col(
    point: np.ndarray, image_shape: tuple[int, int]
) -> tuple[int, int]:
    point = np.asarray(point, dtype=float).reshape(-1)
    if point.size < 2:
        raise ValueError("Image point must contain at least x and y coordinates.")

    col = int(round(point[0]))
    row = int(round(point[1]))
    height, width = image_shape
    return (
        int(np.clip(row, 0, height - 1)),
        int(np.clip(col, 0, width - 1)),
    )


def _snap_to_foreground(
    point: tuple[int, int], foreground_points: np.ndarray
) -> tuple[int, int]:
    if foreground_points.size == 0:
        raise ValueError("Cannot trace paths through an empty binary image.")

    deltas = foreground_points - np.array(point)
    closest_index = int(np.argmin(np.sum(deltas * deltas, axis=1)))
    return tuple(int(value) for value in foreground_points[closest_index])


def _foreground_neighbors(
    point: tuple[int, int], foreground: np.ndarray
) -> list[tuple[int, int]]:
    row, col = point
    height, width = foreground.shape
    neighbors = []
    for drow in (-1, 0, 1):
        for dcol in (-1, 0, 1):
            if drow == 0 and dcol == 0:
                continue
            neighbor_row = row + drow
            neighbor_col = col + dcol
            if not (0 <= neighbor_row < height and 0 <= neighbor_col < width):
                continue
            if foreground[neighbor_row, neighbor_col]:
                if (
                    drow != 0
                    and dcol != 0
                    and (
                        foreground[row + drow, col]
                        or foreground[row, col + dcol]
                    )
                ):
                    continue
                neighbors.append((neighbor_row, neighbor_col))
    return neighbors


def _connected_component(
    start: tuple[int, int], foreground: np.ndarray
) -> set[tuple[int, int]]:
    component = {start}
    stack = [start]
    while stack:
        point = stack.pop()
        for neighbor in _foreground_neighbors(point, foreground):
            if neighbor in component:
                continue
            component.add(neighbor)
            stack.append(neighbor)
    return component


def _foreground_graph(
    component: set[tuple[int, int]], foreground: np.ndarray
) -> tuple[dict[tuple[int, int], list[tuple[tuple[int, int], int]]], int]:
    adjacency = {point: [] for point in component}
    edge_count = 0
    for point in component:
        for neighbor in _foreground_neighbors(point, foreground):
            if neighbor not in component or neighbor <= point:
                continue
            adjacency[point].append((neighbor, edge_count))
            adjacency[neighbor].append((point, edge_count))
            edge_count += 1
    return adjacency, edge_count


def _has_full_edge_trail(
    adjacency: dict[tuple[int, int], list[tuple[tuple[int, int], int]]],
    start: tuple[int, int],
    end: tuple[int, int],
) -> bool:
    odd_nodes = {point for point, edges in adjacency.items() if len(edges) % 2 == 1}
    if start == end:
        return len(odd_nodes) == 0
    return odd_nodes == {start, end}


def _select_full_trace_endpoints(
    adjacency: dict[tuple[int, int], list[tuple[tuple[int, int], int]]],
    start_hint: tuple[int, int],
    end_hint: tuple[int, int],
    snapped_start: tuple[int, int],
    snapped_end: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    endpoint_candidates = [
        point for point, edges in adjacency.items() if len(edges) == 1
    ]
    if len(endpoint_candidates) >= 2:
        best_endpoints = None
        best_cost = np.inf
        for endpoint_a in endpoint_candidates:
            for endpoint_b in endpoint_candidates:
                if endpoint_a == endpoint_b:
                    continue
                cost = (
                    np.sum((np.array(endpoint_a) - np.array(start_hint)) ** 2)
                    + np.sum((np.array(endpoint_b) - np.array(end_hint)) ** 2)
                )
                if cost < best_cost:
                    best_cost = cost
                    best_endpoints = (endpoint_a, endpoint_b)
        return best_endpoints

    odd_nodes = [point for point, edges in adjacency.items() if len(edges) % 2 == 1]
    if len(odd_nodes) == 2:
        endpoint_a, endpoint_b = odd_nodes
        forward_cost = (
            np.sum((np.array(endpoint_a) - np.array(start_hint)) ** 2)
            + np.sum((np.array(endpoint_b) - np.array(end_hint)) ** 2)
        )
        reverse_cost = (
            np.sum((np.array(endpoint_b) - np.array(start_hint)) ** 2)
            + np.sum((np.array(endpoint_a) - np.array(end_hint)) ** 2)
        )
        if forward_cost <= reverse_cost:
            return endpoint_a, endpoint_b
        return endpoint_b, endpoint_a

    if len(odd_nodes) == 0 and snapped_start == snapped_end:
        return snapped_start, snapped_start
    if snapped_start != snapped_end:
        return snapped_start, snapped_end

    return None


def _shortest_graph_path(
    adjacency: dict[tuple[int, int], list[tuple[tuple[int, int], int]]],
    start: tuple[int, int],
    end: tuple[int, int],
    blocked_edges: set[tuple[tuple[int, int], tuple[int, int]]] | None = None,
) -> list[tuple[int, int]] | None:
    if blocked_edges is None:
        blocked_edges = set()
    parents = {start: None}
    queue = [start]
    queue_index = 0
    while queue_index < len(queue):
        point = queue[queue_index]
        queue_index += 1
        if point == end:
            break
        for neighbor, _edge_index in adjacency[point]:
            edge = tuple(sorted((point, neighbor)))
            if edge in blocked_edges:
                continue
            if neighbor in parents:
                continue
            parents[neighbor] = point
            queue.append(neighbor)

    if end not in parents:
        return None

    path = [end]
    while path[-1] != start:
        path.append(parents[path[-1]])
    path.reverse()
    return path


def _pair_path_options(
    adjacency: dict[tuple[int, int], list[tuple[tuple[int, int], int]]],
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[list[tuple[int, int]]]:
    shortest_path = _shortest_graph_path(adjacency, start, end)
    if shortest_path is None:
        return []

    options = [shortest_path]
    seen_options = {tuple(shortest_path)}
    shortest_edges = {
        tuple(sorted((point_a, point_b)))
        for point_a, point_b in zip(shortest_path[:-1], shortest_path[1:])
    }
    alternate_path = _shortest_graph_path(
        adjacency, start, end, blocked_edges=shortest_edges
    )
    if alternate_path is not None and tuple(alternate_path) not in seen_options:
        options.append(alternate_path)
    return options


def _pairing_path_candidates(
    adjacency: dict[tuple[int, int], list[tuple[tuple[int, int], int]]],
    nodes: list[tuple[int, int]],
    *,
    max_candidates: int = 64,
) -> list[list[list[tuple[int, int]]]] | None:
    nodes = sorted(nodes)
    if len(nodes) == 0:
        return []
    if len(nodes) % 2 != 0:
        return None

    pair_path_options = {}
    for i, node_a in enumerate(nodes):
        for node_b in nodes[i + 1 :]:
            options = _pair_path_options(adjacency, node_a, node_b)
            if not options:
                return None
            pair_path_options[(node_a, node_b)] = options
            pair_path_options[(node_b, node_a)] = [
                list(reversed(path)) for path in options
            ]

    if len(nodes) > 14:
        remaining = set(nodes)
        pairings = []
        while remaining:
            node_a = min(remaining)
            remaining.remove(node_a)
            node_b = min(
                remaining,
                key=lambda candidate: len(pair_path_options[(node_a, candidate)][0]),
            )
            remaining.remove(node_b)
            pairings.append(pair_path_options[(node_a, node_b)][0])
        return [pairings]

    memo = {}

    def solve(remaining: tuple[tuple[int, int], ...]):
        if not remaining:
            return [(0, [])]
        if remaining in memo:
            return memo[remaining]

        node_a = remaining[0]
        candidates = []
        for i in range(1, len(remaining)):
            node_b = remaining[i]
            rest = remaining[1:i] + remaining[i + 1 :]
            for path in pair_path_options[(node_a, node_b)]:
                path_cost = len(path) - 1
                for rest_cost, rest_pairings in solve(rest):
                    candidates.append(
                        (path_cost + rest_cost, [path] + rest_pairings)
                    )

        candidates.sort(key=lambda candidate: candidate[0])
        candidates = candidates[:max_candidates]
        memo[remaining] = candidates
        return candidates

    return [pairings for _cost, pairings in solve(tuple(nodes))]


def _add_multigraph_edge(
    multigraph: dict[tuple[int, int], list[tuple[tuple[int, int], int]]],
    point_a: tuple[int, int],
    point_b: tuple[int, int],
    edge_index: int,
) -> None:
    multigraph[point_a].append((point_b, edge_index))
    multigraph[point_b].append((point_a, edge_index))


def _euler_walk(
    multigraph: dict[tuple[int, int], list[tuple[tuple[int, int], int]]],
    start: tuple[int, int],
    edge_count: int,
) -> list[tuple[int, int]] | None:
    local_graph = {point: edges.copy() for point, edges in multigraph.items()}
    used_edges = set()
    stack = [start]
    walk = []

    while stack:
        point = stack[-1]
        while local_graph[point] and local_graph[point][-1][1] in used_edges:
            local_graph[point].pop()
        if not local_graph[point]:
            walk.append(stack.pop())
            continue

        neighbor, edge_index = local_graph[point].pop()
        used_edges.add(edge_index)
        stack.append(neighbor)

    if len(used_edges) != edge_count:
        return None
    walk.reverse()
    return walk


def _multigraph_order_variants(
    multigraph: dict[tuple[int, int], list[tuple[tuple[int, int], int]]],
    branch_nodes: list[tuple[int, int]] | None = None,
    *,
    max_permuted_variants: int = 1024,
) -> list[dict[tuple[int, int], list[tuple[tuple[int, int], int]]]]:
    variants = [
        {point: edges.copy() for point, edges in multigraph.items()},
        {point: list(reversed(edges)) for point, edges in multigraph.items()},
    ]
    if branch_nodes is None:
        return variants

    branch_nodes = [
        point for point in branch_nodes if point in multigraph and len(multigraph[point]) > 2
    ]
    if len(branch_nodes) > 2:
        return variants

    permutations_by_node = [
        list(itertools.permutations(multigraph[point])) for point in branch_nodes
    ]
    for permutations in itertools.product(*permutations_by_node):
        variant = {point: edges.copy() for point, edges in multigraph.items()}
        for point, edge_order in zip(branch_nodes, permutations):
            variant[point] = list(edge_order)
        variants.append(variant)
        if len(variants) >= max_permuted_variants:
            break

    return variants


def _edge_covering_walks(
    adjacency: dict[tuple[int, int], list[tuple[tuple[int, int], int]]],
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[np.ndarray]:
    odd_nodes = {point for point, edges in adjacency.items() if len(edges) % 2 == 1}
    parity_corrections = set(odd_nodes)
    for endpoint in (start, end):
        if endpoint in parity_corrections:
            parity_corrections.remove(endpoint)
        else:
            parity_corrections.add(endpoint)

    duplicate_path_candidates = _pairing_path_candidates(
        adjacency, list(parity_corrections)
    )
    if duplicate_path_candidates is None:
        return []

    paths = []
    seen_paths = set()
    branch_nodes = [point for point, edges in adjacency.items() if len(edges) > 2]
    for duplicate_paths in duplicate_path_candidates:
        multigraph = {point: [] for point in adjacency}
        next_edge_index = 0
        for point, edges in adjacency.items():
            for neighbor, _edge_index in edges:
                if neighbor <= point:
                    continue
                _add_multigraph_edge(multigraph, point, neighbor, next_edge_index)
                next_edge_index += 1

        for duplicate_path in duplicate_paths:
            for point_a, point_b in zip(duplicate_path[:-1], duplicate_path[1:]):
                _add_multigraph_edge(multigraph, point_a, point_b, next_edge_index)
                next_edge_index += 1

        for multigraph_variant in _multigraph_order_variants(
            multigraph, branch_nodes
        ):
            walk = _euler_walk(multigraph_variant, start, next_edge_index)
            if walk is None or walk[-1] != end:
                continue
            path = _path_nodes_to_image_points(walk)
            path_key = tuple(tuple(point) for point in path)
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)
            paths.append(path)

    return paths


def _path_nodes_to_image_points(path: list[tuple[int, int]]) -> np.ndarray:
    return np.array([(col, row) for row, col in path], dtype=int)


def _enumerate_full_edge_trails(
    adjacency: dict[tuple[int, int], list[tuple[tuple[int, int], int]]],
    start: tuple[int, int],
    end: tuple[int, int],
    edge_count: int,
) -> list[np.ndarray]:
    full_edge_mask = (1 << edge_count) - 1
    paths = []
    stack = [(start, [start], 0)]

    while stack:
        point, path, used_edge_mask = stack.pop()
        if used_edge_mask == full_edge_mask:
            if point == end:
                paths.append(_path_nodes_to_image_points(path))
            continue

        for neighbor, edge_index in adjacency[point]:
            edge_mask = 1 << edge_index
            if used_edge_mask & edge_mask:
                continue
            stack.append((neighbor, path + [neighbor], used_edge_mask | edge_mask))

    unique_paths = []
    seen_paths = set()
    for path in paths:
        path_key = tuple(tuple(point) for point in path)
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        unique_paths.append(path)
    return unique_paths


def trace_paths(binary_image, start_point, end_point) -> list[np.ndarray]:
    """
    binary_image
    start_point: point in image space where the tracing should start
    end_point: point in image space where the tracing should end

    output:
    a list of all paths starting near the start_point, going along the full
    thinned image component, and ending near the end_point. Points identify the
    trace component and orientation; the actual traversal endpoints are the
    component's graph endpoints when the component has them.
    Each path is returned as an (N, 2) array of image coordinates in (x, y) order.
    """
    binary_image = _as_binary_uint8(binary_image)
    thinned = thin_binary_image(binary_image)
    foreground = thinned > 0
    foreground_points = np.argwhere(foreground)
    start_hint = _image_point_to_row_col(start_point, foreground.shape)
    end_hint = _image_point_to_row_col(end_point, foreground.shape)
    snapped_start = _snap_to_foreground(start_hint, foreground_points)
    snapped_end = _snap_to_foreground(end_hint, foreground_points)

    component = _connected_component(snapped_start, foreground)
    if snapped_end not in component:
        return []
    if len(component) == 1:
        row, col = snapped_start
        return [np.array([[col, row]], dtype=int)]

    adjacency, edge_count = _foreground_graph(component, foreground)
    if edge_count == 0:
        return []
    endpoints = _select_full_trace_endpoints(
        adjacency, start_hint, end_hint, snapped_start, snapped_end
    )
    if endpoints is None:
        return []
    start, end = endpoints
    if _has_full_edge_trail(adjacency, start, end):
        paths = _enumerate_full_edge_trails(adjacency, start, end, edge_count)
    else:
        paths = _edge_covering_walks(adjacency, start, end)
    paths.sort(key=len)
    return paths


def _marker_points_xy(marker_points: np.ndarray) -> np.ndarray:
    marker_points = np.asarray(marker_points, dtype=float)
    if marker_points.size == 0:
        return np.empty((0, 2), dtype=float)
    if marker_points.ndim == 1:
        marker_points = marker_points.reshape(1, -1)
    if marker_points.shape[-1] < 2:
        raise ValueError("Marker points must contain at least x and y coordinates.")
    return marker_points.reshape(-1, marker_points.shape[-1])[:, :2]


def _path_progress_for_points(
    path: np.ndarray, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    path = np.asarray(path, dtype=float)
    points = np.asarray(points, dtype=float)
    if len(path) == 0:
        raise ValueError("Cannot order marker points along an empty path.")
    if len(path) == 1:
        distances = np.linalg.norm(points - path[0], axis=1)
        return np.zeros(len(points)), distances

    segment_starts = path[:-1]
    segment_vectors = path[1:] - path[:-1]
    segment_lengths_sq = np.sum(segment_vectors * segment_vectors, axis=1)
    segment_lengths = np.sqrt(segment_lengths_sq)
    valid_segments = segment_lengths_sq > 1e-12
    if not np.any(valid_segments):
        distances = np.linalg.norm(points - path[0], axis=1)
        return np.zeros(len(points)), distances

    segment_starts = segment_starts[valid_segments]
    segment_vectors = segment_vectors[valid_segments]
    segment_lengths_sq = segment_lengths_sq[valid_segments]
    segment_lengths = segment_lengths[valid_segments]
    segment_arc_starts = np.r_[0.0, np.cumsum(segment_lengths)[:-1]]

    point_offsets = points[:, None, :] - segment_starts[None, :, :]
    projection_fractions = (
        np.sum(point_offsets * segment_vectors[None, :, :], axis=2)
        / segment_lengths_sq[None, :]
    )
    projection_fractions = np.clip(projection_fractions, 0.0, 1.0)
    projections = (
        segment_starts[None, :, :]
        + projection_fractions[:, :, None] * segment_vectors[None, :, :]
    )
    distance_sq = np.sum((points[:, None, :] - projections) ** 2, axis=2)
    nearest_segments = np.argmin(distance_sq, axis=1)
    progress = (
        segment_arc_starts[nearest_segments]
        + projection_fractions[np.arange(len(points)), nearest_segments]
        * segment_lengths[nearest_segments]
    )
    distances = np.sqrt(distance_sq[np.arange(len(points)), nearest_segments])
    return progress, distances


def generate_point_orderings(
    binary_image, start_point, end_point, marker_points
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Return unique marker index orderings and one corresponding traced path.

    For each path returned by trace_paths, each marker point is projected to the
    nearest position on that path and sorted by arc-length position from start_point
    toward end_point. The returned tuples are (ordering, traced_path), where
    ordering indexes marker_points and traced_path is an (N, 2) image-space path.
    """
    marker_points_xy = _marker_points_xy(marker_points)
    if len(marker_points_xy) == 0:
        return []

    ordering_paths = []
    seen_orderings = set()
    for path in trace_paths(binary_image, start_point, end_point):
        ordering = order_marker_points_along_path(path, marker_points_xy)
        ordering_key = tuple(int(index) for index in ordering)
        if ordering_key in seen_orderings:
            continue
        seen_orderings.add(ordering_key)
        ordering_paths.append((ordering.astype(int), path))

    return ordering_paths


def order_marker_points_along_path(path: np.ndarray, marker_points) -> np.ndarray:
    marker_points_xy = _marker_points_xy(marker_points)
    if len(marker_points_xy) == 0:
        return np.array([], dtype=int)

    progress, distances = _path_progress_for_points(path, marker_points_xy)
    return np.lexsort(
        (np.arange(len(marker_points_xy)), distances, progress)
    ).astype(int)
