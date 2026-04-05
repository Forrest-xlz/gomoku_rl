from __future__ import annotations

import ctypes
import enum
import json
import logging
import random
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import hydra
import numpy as np
import pyautogui
from omegaconf import DictConfig, OmegaConf
from PIL import ImageGrab
from PyQt5.QtCore import Qt, QRect, QTimer
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from gomoku_rl import CONFIG_PATH
from gomoku_rl.mcts_infer import MCTSManager

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05

IS_WINDOWS = sys.platform.startswith("win")
VK_ESCAPE = 0x1B


class Piece(enum.Enum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2


EMPTY = Piece.EMPTY.value
BLACK = Piece.BLACK.value
WHITE = Piece.WHITE.value


class RunMode(enum.Enum):
    MANUAL = "manual"
    AUTO_INFER = "auto_infer"
    AUTO_PLAY = "auto_play"


@dataclass
class FrameJob:
    generation: int
    roi: Optional[list[int]]
    corners_in_roi: Optional[np.ndarray]
    board_size: int
    cell: int
    margin: int
    black_diff_thresh: float
    white_diff_thresh: float
    my_color: Piece
    mode: RunMode
    game_active: bool
    last_board_hash: Optional[str]
    last_suggested_hash: Optional[str]
    request_manual_infer: bool
    stop_event: threading.Event


@dataclass
class FrameResult:
    generation: int
    ok: bool
    error: str = ""
    warped: Optional[np.ndarray] = None
    board: Optional[np.ndarray] = None
    debug_info: Optional[list[dict]] = None
    perspective_matrix: Optional[np.ndarray] = None
    current_turn: Optional[Piece] = None
    board_hash_value: Optional[str] = None
    board_changed: bool = False
    move: Optional[tuple[int, int]] = None
    model_msg: str = ""
    status_msg: str = ""
    clicked: bool = False
    clicked_screen_pos: Optional[tuple[int, int]] = None
    reference_board_after_click: Optional[np.ndarray] = None
    predicted: bool = False
    suppress_preview_update: bool = False
    click_requested: bool = False
    click_move: Optional[tuple[int, int]] = None


@dataclass
class InferJob:
    generation: int
    board: np.ndarray
    latest_move: Optional[tuple[int, int]]
    current_turn: Optional[Piece]
    board_hash_value: str
    perspective_matrix: np.ndarray
    roi: list[int]
    cell: int
    margin: int
    my_color: Piece
    mode: RunMode
    game_active: bool
    stop_event: threading.Event


@dataclass
class InferResult:
    generation: int
    ok: bool
    error: str = ""
    board_hash_value: Optional[str] = None
    move: Optional[tuple[int, int]] = None
    model_msg: str = ""
    status_msg: str = ""
    predicted: bool = False
    click_requested: bool = False
    clicked_screen_pos: Optional[tuple[int, int]] = None
    reference_board_after_click: Optional[np.ndarray] = None


class MonitorOverlay(QWidget):
    """Top-level transparent overlay to mark the monitored ROI."""

    def __init__(self):
        super().__init__(None)
        self.message_lines: list[str] = ["正在监控此区域", "请勿遮挡"]
        self.board_rect = QRect()
        self.label_rect = QRect()
        self._last_layout_key = None
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.BypassWindowManagerHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.hide()

    def update_overlay(self, roi: Optional[list[int]], message_lines: list[str], visible: bool) -> None:
        if not visible or roi is None:
            self._last_layout_key = None
            self.hide()
            return

        lines = list(message_lines)
        x, y, w, h = roi
        line_count = max(len(lines), 1)
        label_height = 12 + line_count * 18 + 10
        label_width = min(max(w - 16, 240), 420)
        desired_top_padding = label_height + 12
        overlay_top = max(0, y - desired_top_padding)
        board_top = y - overlay_top
        label_top = max(4, board_top - label_height - 4)
        geom = (x, overlay_top, w, h + board_top)
        board_rect = QRect(0, board_top, w, h)
        label_rect = QRect(8, label_top, label_width, label_height)
        layout_key = (geom, board_rect.getRect(), label_rect.getRect(), tuple(lines))

        self.message_lines = lines
        if self._last_layout_key != layout_key:
            self._last_layout_key = layout_key
            self.setGeometry(*geom)
            self.board_rect = board_rect
            self.label_rect = label_rect
            if not self.isVisible():
                self.show()
            self.raise_()
            self.update()
        elif not self.isVisible():
            self.show()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        border_pen = QPen(QColor(0, 255, 170, 220), 2)
        painter.setPen(border_pen)
        painter.drawRect(self.board_rect.adjusted(1, 1, -2, -2))

        painter.fillRect(self.label_rect, QColor(0, 0, 0, 115))
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Microsoft YaHei", 9))
        painter.drawText(
            self.label_rect.adjusted(7, 5, -7, -5),
            Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap,
            "\n".join(self.message_lines),
        )


def grab_screen_bgr() -> np.ndarray:
    try:
        img = ImageGrab.grab(all_screens=True)
    except TypeError:
        img = ImageGrab.grab()
    img = np.array(img)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def crop_roi_from_screen(roi: list[int]) -> Optional[np.ndarray]:
    screen = grab_screen_bgr()
    x, y, w, h = roi
    roi_img = screen[y : y + h, x : x + w].copy()
    if roi_img.size == 0:
        return None
    return roi_img


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.array(pts, dtype=np.float32)
    pts = pts[np.argsort(pts[:, 1])]
    top = pts[:2]
    bottom = pts[2:]
    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]
    tl, tr = top
    bl, br = bottom
    return np.array([tl, tr, bl, br], dtype=np.float32)


def warp_board(
    img: np.ndarray,
    corners: np.ndarray,
    board_size: int,
    cell: int,
    margin: int,
) -> tuple[np.ndarray, np.ndarray]:
    out_size = int(2 * margin + (board_size - 1) * cell)
    dst = np.array(
        [
            [margin, margin],
            [margin + (board_size - 1) * cell, margin],
            [margin, margin + (board_size - 1) * cell],
            [margin + (board_size - 1) * cell, margin + (board_size - 1) * cell],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(img, matrix, (out_size, out_size))
    return warped, matrix


def build_local_masks(cell: int) -> tuple[int, np.ndarray, np.ndarray]:
    outer_r = int(round(cell * 0.60))
    yy, xx = np.ogrid[-outer_r : outer_r + 1, -outer_r : outer_r + 1]
    dist = np.sqrt(xx * xx + yy * yy)

    center_mask = dist <= cell * 0.28
    ring_mask = (dist >= cell * 0.45) & (dist <= cell * 0.60)
    return outer_r, center_mask, ring_mask


def classify_intersection(
    gray: np.ndarray,
    x: int,
    y: int,
    outer_r: int,
    center_mask: np.ndarray,
    ring_mask: np.ndarray,
    black_thresh: float,
    white_thresh: float,
) -> tuple[int, float, float, float]:
    patch = gray[y - outer_r : y + outer_r + 1, x - outer_r : x + outer_r + 1]
    if patch.shape != center_mask.shape:
        return EMPTY, 0.0, 0.0, 0.0

    center_mean = float(patch[center_mask].mean())
    ring_mean = float(patch[ring_mask].mean())
    diff = center_mean - ring_mean

    if diff < black_thresh:
        state = BLACK
    elif diff > white_thresh:
        state = WHITE
    else:
        state = EMPTY

    return state, diff, center_mean, ring_mean


def detect_board_state(
    warped: np.ndarray,
    board_size: int,
    cell: int,
    margin: int,
    black_thresh: float,
    white_thresh: float,
) -> tuple[np.ndarray, list[dict]]:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    board = np.zeros((board_size, board_size), dtype=np.int32)
    debug_info: list[dict] = []

    outer_r, center_mask, ring_mask = build_local_masks(cell)

    for row in range(board_size):
        for col in range(board_size):
            x = int(round(margin + col * cell))
            y = int(round(margin + row * cell))
            state, diff, center_mean, ring_mean = classify_intersection(
                gray=gray,
                x=x,
                y=y,
                outer_r=outer_r,
                center_mask=center_mask,
                ring_mask=ring_mask,
                black_thresh=black_thresh,
                white_thresh=white_thresh,
            )
            board[row, col] = state
            debug_info.append(
                {
                    "row": row,
                    "col": col,
                    "x": x,
                    "y": y,
                    "state": int(state),
                    "diff": float(diff),
                    "center_mean": float(center_mean),
                    "ring_mean": float(ring_mean),
                }
            )

    return board, debug_info


def draw_result(
    warped: np.ndarray,
    board: np.ndarray,
    board_size: int,
    cell: int,
    margin: int,
    latest_move: Optional[tuple[int, int]] = None,
    suggested_move: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    vis = warped.copy()

    for row in range(board_size):
        for col in range(board_size):
            x = int(round(margin + col * cell))
            y = int(round(margin + row * cell))

            cv2.drawMarker(
                vis,
                (x, y),
                (0, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=7,
                thickness=1,
            )

            state = int(board[row, col])
            if state == BLACK:
                color = (255, 0, 0)
                text = "B"
            elif state == WHITE:
                color = (0, 0, 255)
                text = "W"
            else:
                continue

            cv2.circle(vis, (x, y), int(cell * 0.22), color, 2)
            cv2.putText(
                vis,
                text,
                (x - 8, y + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
                cv2.LINE_AA,
            )

    if latest_move is not None:
        row, col = latest_move
        x = int(round(margin + col * cell))
        y = int(round(margin + row * cell))
        cv2.drawMarker(
            vis,
            (x, y),
            (0, 255, 255),
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=18,
            thickness=2,
        )
        cv2.putText(
            vis,
            "LAST",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if suggested_move is not None:
        row, col = suggested_move
        x = int(round(margin + col * cell))
        y = int(round(margin + row * cell))
        cv2.circle(vis, (x, y), int(cell * 0.32), (255, 255, 0), 2)
        cv2.drawMarker(
            vis,
            (x, y),
            (255, 255, 0),
            markerType=cv2.MARKER_STAR,
            markerSize=20,
            thickness=2,
        )
        cv2.putText(
            vis,
            "AI",
            (x + 8, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return vis


def infer_current_player(board: np.ndarray) -> Optional[Piece]:
    black_count = int((board == BLACK).sum())
    white_count = int((board == WHITE).sum())
    if black_count == white_count:
        return Piece.BLACK
    if black_count == white_count + 1:
        return Piece.WHITE
    return None


def opponent_piece_value(player: Piece) -> int:
    return WHITE if player == Piece.BLACK else BLACK


def self_piece_value(player: Piece) -> int:
    return BLACK if player == Piece.BLACK else WHITE


def derive_opponent_last_move(
    previous_board: Optional[np.ndarray],
    current_board: np.ndarray,
    opponent_value: int,
) -> Optional[tuple[int, int]]:
    if previous_board is None or previous_board.shape != current_board.shape:
        return None

    prev_opponent = previous_board == opponent_value
    curr_opponent = current_board == opponent_value

    removed = np.argwhere(prev_opponent & (~curr_opponent))
    added = np.argwhere((~prev_opponent) & curr_opponent)

    if len(removed) != 0 or len(added) != 1:
        return None

    row, col = map(int, added[0])
    return row, col


def board_hash(board: np.ndarray) -> str:
    return board.tobytes().hex()


def board_to_screen(
    row: int,
    col: int,
    *,
    roi: list[int],
    perspective_matrix: np.ndarray,
    cell: int,
    margin: int,
) -> tuple[int, int]:
    x = margin + col * cell
    y = margin + row * cell
    inv_matrix = np.linalg.inv(perspective_matrix)
    point = np.array([[[x, y]]], dtype=np.float32)
    roi_point = cv2.perspectiveTransform(point, inv_matrix)[0][0]
    rx, ry = int(roi_point[0]), int(roi_point[1])
    screen_x = roi[0] + rx
    screen_y = roi[1] + ry
    return screen_x, screen_y


def run_click_job(
    *,
    stop_event: threading.Event,
    screen_x: int,
    screen_y: int,
    row: int,
    col: int,
) -> tuple[bool, str]:
    try:
        if stop_event.is_set():
            return False, "对局已停止，已取消自动点击。"
        pyautogui.moveTo(screen_x, screen_y, duration=random.uniform(0.15, 0.35))
        if stop_event.is_set():
            return False, "对局已停止，已取消自动点击。"
        pyautogui.click()
        time.sleep(1.0 / 3.0)
        if stop_event.is_set():
            return False, "对局已停止，已取消自动点击。"
        pyautogui.click()
        return True, f"自动操作：已在第 {row + 1} 行第 {col + 1} 列执行双击。"
    except Exception as exc:
        logging.exception("run_click_job failed")
        return False, f"自动点击失败：{exc}"


def run_frame_job(job: FrameJob, model_manager: MCTSManager) -> FrameResult:
    result = FrameResult(generation=job.generation, ok=False)

    try:
        if job.roi is None or job.corners_in_roi is None:
            result.error = "请先设置棋盘区域并标记四角交点。"
            return result

        roi_img = crop_roi_from_screen(job.roi)
        if roi_img is None:
            result.error = "当前 ROI 超出屏幕范围，请重新设置。"
            return result

        warped, matrix = warp_board(
            img=roi_img,
            corners=job.corners_in_roi,
            board_size=job.board_size,
            cell=job.cell,
            margin=job.margin,
        )

        board, debug_info = detect_board_state(
            warped=warped,
            board_size=job.board_size,
            cell=job.cell,
            margin=job.margin,
            black_thresh=job.black_diff_thresh,
            white_thresh=job.white_diff_thresh,
        )

        current_hash = board_hash(board)
        board_changed = current_hash != job.last_board_hash
        current_turn = infer_current_player(board)

        result.ok = True
        result.warped = warped
        result.board = board
        result.debug_info = debug_info
        result.perspective_matrix = matrix
        result.current_turn = current_turn
        result.board_hash_value = current_hash
        result.board_changed = board_changed
        return result
    except Exception as exc:
        logging.exception("run_frame_job failed")
        result.error = f"后台任务失败：{exc}"
        return result


def run_infer_job(job: InferJob, model_manager: MCTSManager) -> InferResult:
    result = InferResult(generation=job.generation, ok=False, board_hash_value=job.board_hash_value)
    try:
        if job.stop_event.is_set() or not job.game_active:
            result.ok = True
            return result

        if job.current_turn is None:
            result.ok = True
            return result

        if job.current_turn != job.my_color:
            result.ok = True
            return result

        move, msg, _current_player = model_manager.predict(job.board, job.latest_move)
        result.ok = True
        result.predicted = True
        result.move = move
        result.model_msg = msg if move is not None else "无可用建议"

        if move is not None and job.mode == RunMode.AUTO_PLAY:
            if job.stop_event.is_set():
                result.status_msg = "对局已停止，已取消自动点击。"
                return result
            row, col = move
            screen_x, screen_y = board_to_screen(
                row,
                col,
                roi=job.roi,
                perspective_matrix=job.perspective_matrix,
                cell=job.cell,
                margin=job.margin,
            )
            reference_board = job.board.copy()
            reference_board[row, col] = self_piece_value(job.my_color)
            result.reference_board_after_click = reference_board
            result.click_requested = True
            result.clicked_screen_pos = (screen_x, screen_y)
            result.status_msg = f"自动操作：准备在第 {row + 1} 行第 {col + 1} 列执行双击。"
        elif move is not None:
            row, col = move
            result.status_msg = f"建议第 {row + 1} 行第 {col + 1} 列。"
        else:
            result.status_msg = "已完成推理，但未得到可用建议。"
        return result
    except Exception as exc:
        logging.exception("run_infer_job failed")
        result.error = f"推理失败：{exc}"
        return result


class ScreenMonitorWidget(QWidget):
    def __init__(self, cfg: DictConfig, model_manager: MCTSManager):
        super().__init__()
        self.cfg = cfg
        self.model_manager = model_manager

        self.roi: Optional[list[int]] = None
        self.corners_in_roi: Optional[np.ndarray] = None
        self.latest_move: Optional[tuple[int, int]] = None
        self.latest_move_board_hash: Optional[str] = None
        self.latest_warped: Optional[np.ndarray] = None
        self.latest_board: Optional[np.ndarray] = None
        self.lastmove_reference_board: Optional[np.ndarray] = None
        self.preview_pixmap: Optional[QPixmap] = None
        self.suggested_move: Optional[tuple[int, int]] = None
        self.last_turn: Optional[Piece] = None
        self.last_board_hash: Optional[str] = None
        self.last_suggested_hash: Optional[str] = None
        self.perspective_matrix: Optional[np.ndarray] = None

        self.mode: RunMode = RunMode.AUTO_PLAY
        self.game_active: bool = False
        self.detect_busy: bool = False
        self.infer_busy: bool = False
        self.pending_manual_request: bool = False
        self.pending_fast_refresh: bool = False
        self.run_generation: int = 0
        self.current_future: Optional[Future] = None
        self.infer_future: Optional[Future] = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gomoku_detect")
        self.infer_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gomoku_infer")
        self.click_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gomoku_click")
        self.click_future: Optional[Future] = None
        self.last_infer_request_hash: Optional[str] = None
        self.lastmove_reference_hash: Optional[str] = None
        self.stop_event = threading.Event()
        self.stop_event.set()
        self.pending_click_reference_board: Optional[np.ndarray] = None
        self.pending_click_move: Optional[tuple[int, int]] = None

        self.overlay = MonitorOverlay()
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.on_poll_tick)
        self.future_timer = QTimer(self)
        self.future_timer.timeout.connect(self.consume_background_results)
        self.future_timer.start(30)

        self.global_key_timer = QTimer(self)
        self.global_key_timer.timeout.connect(self.poll_global_keys)
        self.global_key_timer.start(30)
        self._esc_was_down = False

        self._build_ui()
        self.load_runtime_config()
        self.apply_mode(RunMode.AUTO_PLAY)
        self.poll_timer.start(int(self.cfg.get("poll_interval_ms", 120)))

    @property
    def runtime_config_path(self) -> Path:
        return Path(self.cfg.runtime_config_path)

    @property
    def my_color(self) -> Piece:
        data = self.color_combo.currentData()
        return data if data is not None else Piece.BLACK

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(4)

        title = QLabel("屏幕五子棋监控")
        title.setFont(QFont("Arial", 11, QFont.Bold))
        root_layout.addWidget(title)

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)

        self.btn_manual = QPushButton("手动推理")
        self.btn_manual.setCheckable(True)
        self.btn_manual.clicked.connect(lambda: self.apply_mode(RunMode.MANUAL))
        self.btn_manual.setMinimumHeight(30)
        self.btn_manual.setStyleSheet(
            "QPushButton {font-size: 12px; font-weight: 600;}"
            "QPushButton:checked {background: #6c757d; color: white;}"
        )
        self.mode_group.addButton(self.btn_manual)
        root_layout.addWidget(self.btn_manual)

        self.btn_auto_infer = QPushButton("自动推理")
        self.btn_auto_infer.setCheckable(True)
        self.btn_auto_infer.clicked.connect(lambda: self.apply_mode(RunMode.AUTO_INFER))
        self.btn_auto_infer.setMinimumHeight(30)
        self.btn_auto_infer.setStyleSheet(
            "QPushButton {font-size: 12px; font-weight: 600;}"
            "QPushButton:checked {background: #2471a3; color: white;}"
        )
        self.mode_group.addButton(self.btn_auto_infer)
        root_layout.addWidget(self.btn_auto_infer)

        self.btn_auto_play = QPushButton("自动操作")
        self.btn_auto_play.setCheckable(True)
        self.btn_auto_play.clicked.connect(lambda: self.apply_mode(RunMode.AUTO_PLAY))
        self.btn_auto_play.setMinimumHeight(30)
        self.btn_auto_play.setStyleSheet(
            "QPushButton {font-size: 12px; font-weight: 600;}"
            "QPushButton:checked {background: #c0392b; color: white;}"
        )
        self.mode_group.addButton(self.btn_auto_play)
        root_layout.addWidget(self.btn_auto_play)

        self.manual_action_box = QWidget()
        manual_layout = QVBoxLayout(self.manual_action_box)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        manual_layout.setSpacing(4)
        manual_label = QLabel("手动模式操作")
        manual_label.setFont(QFont("Arial", 9))
        manual_layout.addWidget(manual_label)
        self.btn_manual_infer = QPushButton("立即推理")
        self.btn_manual_infer.clicked.connect(self.request_manual_infer)
        self.btn_manual_infer.setMinimumHeight(28)
        self.btn_manual_infer.setStyleSheet("QPushButton {font-size: 11px; font-weight: 600; background: #ecf0f1;}")
        manual_layout.addWidget(self.btn_manual_infer)
        root_layout.addWidget(self.manual_action_box)

        self.btn_start_game = QPushButton("开始对局")
        self.btn_start_game.setMinimumHeight(30)
        self.btn_start_game.setStyleSheet(
            "QPushButton {font-size: 12px; font-weight: 600; background: #1e8449; color: white;}"
        )
        self.btn_start_game.clicked.connect(self.start_game)
        root_layout.addWidget(self.btn_start_game)

        self.btn_end_game = QPushButton("结束对局")
        self.btn_end_game.setMinimumHeight(30)
        self.btn_end_game.setStyleSheet(
            "QPushButton {font-size: 12px; font-weight: 600; background: #7f8c8d; color: white;}"
        )
        self.btn_end_game.clicked.connect(self.end_game)
        root_layout.addWidget(self.btn_end_game)

        self.btn_select_roi = QPushButton("选择棋盘区域")
        self.btn_select_roi.clicked.connect(self.select_roi)
        self.btn_select_roi.setMinimumHeight(28)
        root_layout.addWidget(self.btn_select_roi)

        self.btn_select_corners = QPushButton("标记四角交点")
        self.btn_select_corners.clicked.connect(self.select_corners)
        self.btn_select_corners.setMinimumHeight(28)
        root_layout.addWidget(self.btn_select_corners)

        color_label = QLabel("我方：")
        color_label.setFont(QFont("Arial", 9))
        root_layout.addWidget(color_label)

        self.color_combo = QComboBox()
        self.color_combo.addItem("黑方", Piece.BLACK)
        self.color_combo.addItem("白方", Piece.WHITE)
        self.color_combo.currentIndexChanged.connect(self.on_color_changed)
        self.color_combo.setMinimumHeight(26)
        root_layout.addWidget(self.color_combo)

        self.phase_label = QLabel("对局状态：未开始")
        self.phase_label.setMinimumHeight(22)
        root_layout.addWidget(self.phase_label)

        self.turn_label = QLabel("当前轮次：未知")
        self.turn_label.setMinimumHeight(22)
        root_layout.addWidget(self.turn_label)

        self.last_move_label = QLabel("最近一手：None")
        self.last_move_label.setMinimumHeight(22)
        root_layout.addWidget(self.last_move_label)

        self.status_label = QLabel("状态：等待设置")
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(34)
        root_layout.addWidget(self.status_label)

        self.model_label = QLabel("模型建议：-")
        self.model_label.setWordWrap(True)
        self.model_label.setMinimumHeight(40)
        root_layout.addWidget(self.model_label)

        preview_title = QLabel("识别预览")
        preview_title.setFont(QFont("Arial", 10))
        root_layout.addWidget(preview_title)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(220, 220)
        self.preview_label.setStyleSheet("background: #202020;")
        root_layout.addWidget(self.preview_label, stretch=1)

    def runtime_payload(self) -> dict:
        return {
            "roi": self.roi,
            "corners_in_roi": self.corners_in_roi.tolist() if self.corners_in_roi is not None else None,
            "my_color": "black" if self.my_color == Piece.BLACK else "white",
        }

    def save_runtime_config(self) -> None:
        self.runtime_config_path.write_text(
            json.dumps(self.runtime_payload(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.set_status(f"运行配置已保存到 {self.runtime_config_path.resolve()}")

    def load_runtime_config(self) -> None:
        self.color_combo.setCurrentIndex(0)
        if not self.runtime_config_path.exists():
            self.set_status("未找到运行配置，请先选择棋盘区域。")
            self.refresh_controls()
            self.update_overlay()
            return

        payload = json.loads(self.runtime_config_path.read_text(encoding="utf-8"))
        self.roi = payload.get("roi")
        corners = payload.get("corners_in_roi")
        self.corners_in_roi = np.array(corners, dtype=np.float32) if corners else None

        if payload.get("my_color") in {"black", "white"}:
            self.color_combo.setCurrentIndex(0 if payload["my_color"] == "black" else 1)

        parts: list[str] = []
        if self.roi:
            parts.append(f"ROI={self.roi}")
        if self.corners_in_roi is not None:
            parts.append("已加载四角交点")
        parts.append(f"我方={'黑方' if self.my_color == Piece.BLACK else '白方'}")
        self.set_status("；".join(parts) if parts else "运行配置为空")
        self.refresh_controls()
        self.update_overlay()

    def closeEvent(self, event) -> None:
        self.poll_timer.stop()
        self.future_timer.stop()
        self.global_key_timer.stop()
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        try:
            self.infer_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        try:
            self.click_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        self.overlay.close()
        super().closeEvent(event)

    def current_mode_text(self) -> str:
        if self.mode == RunMode.MANUAL:
            return "手动推理"
        if self.mode == RunMode.AUTO_INFER:
            return "自动推理"
        return "自动操作"

    def refresh_controls(self) -> None:
        self.manual_action_box.setVisible(self.mode == RunMode.MANUAL)
        self.btn_manual_infer.setEnabled(self.mode == RunMode.MANUAL and self.game_active)
        self.btn_start_game.setEnabled(not self.game_active)
        self.btn_end_game.setEnabled(self.game_active)

    def set_window_topmost(self, enabled: bool) -> None:
        top = self.window()
        if top is None:
            return
        top.setWindowFlag(Qt.WindowStaysOnTopHint, enabled)
        top.show()
        if enabled:
            top.raise_()

    def apply_mode(self, mode: RunMode) -> None:
        self.mode = mode
        if mode == RunMode.MANUAL:
            self.btn_manual.setChecked(True)
        elif mode == RunMode.AUTO_INFER:
            self.btn_auto_infer.setChecked(True)
        else:
            self.btn_auto_play.setChecked(True)
        self.refresh_controls()
        self.set_status(f"已切换到{self.current_mode_text()}模式")
        if self.mode == RunMode.MANUAL and self.suggested_move is None:
            self.model_label.setText("模型建议：-")
        self.update_overlay()

    def poll_global_keys(self) -> None:
        if not IS_WINDOWS:
            return
        try:
            state = ctypes.windll.user32.GetAsyncKeyState(VK_ESCAPE)
            is_down = bool(state & 0x8000)
        except Exception:
            return

        if is_down and not self._esc_was_down:
            self.handle_escape()
        self._esc_was_down = is_down

    def handle_escape(self) -> None:
        if self.game_active:
            self.end_game(set_status_text="Esc：已结束当前对局。")

    def set_status(self, text: str) -> None:
        self.status_label.setText(f"状态：{text}")

    def update_phase_label(self) -> None:
        self.phase_label.setText(f"对局状态：{'进行中' if self.game_active else '未开始 / 已结束'}")

    def update_overlay(self) -> None:
        turn_text = '未知' if self.last_turn is None else ('黑方' if self.last_turn == Piece.BLACK else '白方')
        if self.latest_move is None:
            latest_text = '最近一手：-'
        else:
            r, c = self.latest_move
            latest_text = f"最近一手：({r + 1}, {c + 1})"
        if self.suggested_move is None:
            suggest_text = '建议：-'
        else:
            r, c = self.suggested_move
            suggest_text = f"建议：({r + 1}, {c + 1})"

        lines = [
            '正在监控此区域',
            '请勿遮挡',
            f"模式：{self.current_mode_text()}",
            f"对局：{'进行中' if self.game_active else '未开始/已结束'}",
            f"我方：{'黑方' if self.my_color == Piece.BLACK else '白方'}",
            f"当前轮次：{turn_text}",
            latest_text,
            suggest_text,
        ]
        self.overlay.update_overlay(self.roi, lines, self.roi is not None)

    def reset_game_runtime(self) -> None:
        self.latest_move = None
        self.latest_move_board_hash = None
        self.lastmove_reference_board = None
        self.lastmove_reference_hash = None
        self.suggested_move = None
        self.last_turn = None
        self.last_board_hash = None
        self.last_suggested_hash = None
        self.perspective_matrix = None
        self.pending_fast_refresh = False
        self.pending_manual_request = False
        self.pending_click_reference_board = None
        self.pending_click_move = None
        if hasattr(self, "last_move_label"):
            self.last_move_label.setText("最近一手：None")
        self.run_generation += 1
        if hasattr(self.model_manager, "reset_search_tree"):
            try:
                self.model_manager.reset_search_tree()
            except Exception:
                logging.exception("reset_search_tree failed")

    def start_game(self) -> None:
        self.stop_event.set()
        self.stop_event = threading.Event()
        self.game_active = True
        self.reset_game_runtime()
        self.set_window_topmost(True)
        self.update_phase_label()
        self.refresh_controls()
        self.last_move_label.setText("最近一手：None")
        self.model_label.setText("模型建议：-")
        if self.mode == RunMode.MANUAL:
            self.set_status("对局已开始。手动模式等待你手动触发推理。")
        else:
            self.set_status("对局已开始。窗口保持置顶，Esc 可直接结束对局。")
            self.submit_frame_job(request_manual_infer=False)
        self.update_overlay()

    def end_game(self, set_status_text: Optional[str] = None) -> None:
        self.stop_event.set()
        self.game_active = False
        self.reset_game_runtime()
        self.set_window_topmost(True)
        self.update_phase_label()
        self.refresh_controls()
        self.model_label.setText("模型建议：-")
        self.turn_label.setText("当前轮次：未知")
        self.last_move_label.setText("最近一手：None")
        self.suggested_move = None
        self.last_turn = None
        self.render_preview()
        self.set_status(set_status_text or "对局已结束。窗口保持置顶。")
        self.update_overlay()

    def on_color_changed(self, _index: int) -> None:
        self.save_runtime_config()
        self.update_overlay()

    def select_roi(self) -> None:
        self.set_status("正在截屏，请框选整个棋盘区域。")
        self.set_window_topmost(False)
        screen = grab_screen_bgr()
        cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
        roi = cv2.selectROI("Select ROI", screen, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select ROI")

        x, y, w, h = roi
        if w == 0 or h == 0:
            self.set_status("没有选中 ROI。")
            self.set_window_topmost(True)
            return

        self.roi = [int(x), int(y), int(w), int(h)]
        self.save_runtime_config()
        self.update_overlay()
        self.set_window_topmost(True)

    def select_corners(self) -> None:
        if self.roi is None:
            QMessageBox.warning(self, "提示", "请先选择棋盘区域。")
            return

        self.set_window_topmost(False)
        roi_img = crop_roi_from_screen(self.roi)
        if roi_img is None:
            QMessageBox.critical(self, "错误", "当前 ROI 超出屏幕范围，请重新选择。")
            self.set_window_topmost(True)
            return

        clone = roi_img.copy()
        points: list[tuple[int, int]] = []

        def mouse_callback(event, px, py, flags, param):
            nonlocal clone, points
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append((px, py))
                cv2.circle(clone, (px, py), 5, (0, 0, 255), -1)
                cv2.putText(
                    clone,
                    str(len(points)),
                    (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        self.set_status("请点击四个角上的最外侧交点，Enter 确认，r 重置。")
        cv2.namedWindow("Pick 4 Corners", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Pick 4 Corners", mouse_callback)

        while True:
            cv2.imshow("Pick 4 Corners", clone)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("r"):
                clone = roi_img.copy()
                points = []
            elif key == 13 and len(points) == 4:
                break
            elif key == 27:
                cv2.destroyWindow("Pick 4 Corners")
                self.set_status("已取消四角交点设置。")
                self.set_window_topmost(True)
                return

        cv2.destroyWindow("Pick 4 Corners")
        self.corners_in_roi = order_points(np.array(points, dtype=np.float32))
        self.save_runtime_config()
        self.update_overlay()
        self.set_window_topmost(True)

    def current_vis(self) -> np.ndarray:
        if self.latest_warped is None or self.latest_board is None:
            return np.zeros((300, 300, 3), dtype=np.uint8)
        return draw_result(
            warped=self.latest_warped,
            board=self.latest_board,
            board_size=self.cfg.board_size,
            cell=self.cfg.cell,
            margin=self.cfg.margin,
            latest_move=self.latest_move,
            suggested_move=self.suggested_move,
        )

    def render_preview(self) -> None:
        if self.latest_warped is None or self.latest_board is None:
            self.preview_label.clear()
            return
        self.update_preview(self.current_vis())

    def update_preview(self, bgr_img: np.ndarray) -> None:
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        pixmap = pixmap.scaled(
            self.preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview_pixmap = pixmap
        self.preview_label.setPixmap(self.preview_pixmap)

    def request_manual_infer(self) -> None:
        if self.mode != RunMode.MANUAL:
            self.set_status("“立即推理”仅在手动推理模式下可用。")
            return
        if not self.game_active:
            self.set_status("请先点击“开始对局”。")
            return
        self.pending_manual_request = True
        self.submit_frame_job(request_manual_infer=True)

    def on_poll_tick(self) -> None:
        self.submit_frame_job(request_manual_infer=False)

    def submit_frame_job(self, request_manual_infer: bool) -> None:
        if self.detect_busy:
            if request_manual_infer:
                self.pending_manual_request = True
            return
        if self.roi is None or self.corners_in_roi is None:
            return

        self.detect_busy = True
        if request_manual_infer:
            self.pending_manual_request = False

        job = FrameJob(
            generation=self.run_generation,
            roi=None if self.roi is None else list(self.roi),
            corners_in_roi=None if self.corners_in_roi is None else np.array(self.corners_in_roi, dtype=np.float32),
            board_size=int(self.cfg.board_size),
            cell=int(self.cfg.cell),
            margin=int(self.cfg.margin),
            black_diff_thresh=float(self.cfg.black_diff_thresh),
            white_diff_thresh=float(self.cfg.white_diff_thresh),
            my_color=self.my_color,
            mode=self.mode,
            game_active=self.game_active,
            last_board_hash=self.last_board_hash,
            last_suggested_hash=self.last_suggested_hash,
            request_manual_infer=request_manual_infer,
            stop_event=self.stop_event,
        )
        self.current_future = self.executor.submit(run_frame_job, job, self.model_manager)

    def schedule_fast_refresh(self, delay_ms: int = 40) -> None:
        if self.pending_fast_refresh:
            return
        self.pending_fast_refresh = True

        def _kick_once() -> None:
            if self.game_active and self.mode == RunMode.AUTO_PLAY:
                self.submit_frame_job(request_manual_infer=False)

        def _done() -> None:
            self.pending_fast_refresh = False

        for delay in (delay_ms, delay_ms + 50, delay_ms + 110, delay_ms + 180):
            QTimer.singleShot(delay, _kick_once)
        QTimer.singleShot(delay_ms + 240, _done)

    def start_click_task(
        self,
        move: tuple[int, int],
        screen_pos: tuple[int, int],
        reference_board_after_click: Optional[np.ndarray],
    ) -> None:
        if self.click_future is not None and not self.click_future.done():
            return
        if not self.game_active or self.stop_event.is_set():
            return
        row, col = move
        screen_x, screen_y = screen_pos
        self.pending_click_reference_board = None if reference_board_after_click is None else reference_board_after_click.copy()
        self.pending_click_move = move
        self.click_future = self.click_executor.submit(
            run_click_job,
            stop_event=self.stop_event,
            screen_x=screen_x,
            screen_y=screen_y,
            row=row,
            col=col,
        )

    def consume_click_result(self) -> None:
        if self.click_future is None or not self.click_future.done():
            return
        future = self.click_future
        self.click_future = None
        try:
            clicked, msg = future.result()
        except Exception as exc:
            logging.exception("consume_click_result failed")
            self.set_status(f"自动点击线程异常：{exc}")
            return

        self.set_status(msg)
        if clicked and self.game_active and self.mode == RunMode.AUTO_PLAY:
            if self.pending_click_reference_board is not None:
                self.lastmove_reference_board = self.pending_click_reference_board.copy()
                self.lastmove_reference_hash = board_hash(self.pending_click_reference_board)
                self.latest_move = None
                self.latest_move_board_hash = None
                self.last_move_label.setText("最近一手：None")
            self.schedule_fast_refresh(30)
        self.pending_click_reference_board = None
        self.pending_click_move = None


    def maybe_submit_infer_job(self, request_manual_infer: bool = False) -> None:
        if self.infer_busy:
            if request_manual_infer:
                self.pending_manual_request = True
            return
        if not self.game_active:
            return
        if self.latest_board is None or self.latest_warped is None or self.perspective_matrix is None or self.last_board_hash is None:
            return
        if self.roi is None:
            return

        if request_manual_infer:
            should_infer = True
        elif self.mode in {RunMode.AUTO_INFER, RunMode.AUTO_PLAY}:
            should_infer = self.last_turn == self.my_color and self.last_infer_request_hash != self.last_board_hash
        else:
            should_infer = False

        if not should_infer:
            return

        self.infer_busy = True
        self.pending_manual_request = False
        self.last_infer_request_hash = self.last_board_hash
        job = InferJob(
            generation=self.run_generation,
            board=self.latest_board.copy(),
            latest_move=self.latest_move,
            current_turn=self.last_turn,
            board_hash_value=self.last_board_hash,
            perspective_matrix=self.perspective_matrix.copy(),
            roi=list(self.roi),
            cell=int(self.cfg.cell),
            margin=int(self.cfg.margin),
            my_color=self.my_color,
            mode=self.mode,
            game_active=self.game_active,
            stop_event=self.stop_event,
        )
        self.infer_future = self.infer_executor.submit(run_infer_job, job, self.model_manager)

    def consume_worker_result(self) -> None:
        if self.current_future is None or not self.current_future.done():
            return

        future = self.current_future
        self.current_future = None
        self.detect_busy = False

        try:
            result: FrameResult = future.result()
        except Exception as exc:
            logging.exception("consume_worker_result failed")
            self.set_status(f"后台线程异常：{exc}")
            if self.pending_manual_request:
                self.maybe_submit_infer_job(request_manual_infer=True)
            return

        if result.generation != self.run_generation:
            if self.pending_manual_request:
                self.maybe_submit_infer_job(request_manual_infer=True)
            return

        if not result.ok:
            if result.error:
                self.set_status(result.error)
            if self.pending_manual_request:
                self.maybe_submit_infer_job(request_manual_infer=True)
            return

        self.last_turn = result.current_turn
        self.perspective_matrix = result.perspective_matrix
        self.latest_warped = result.warped
        self.latest_board = result.board
        self.last_board_hash = result.board_hash_value

        if result.board is not None and result.board_hash_value is not None:
            if result.current_turn == self.my_color:
                expected_stone = opponent_piece_value(self.my_color)
                candidate = derive_opponent_last_move(
                    self.lastmove_reference_board,
                    result.board,
                    expected_stone,
                )
                self.latest_move = candidate
                self.latest_move_board_hash = result.board_hash_value
            else:
                self.latest_move = None
                self.latest_move_board_hash = result.board_hash_value
                self.lastmove_reference_board = result.board.copy()
                self.lastmove_reference_hash = result.board_hash_value

        if self.latest_move is None:
            self.last_move_label.setText("最近一手：None")
        else:
            r, c = self.latest_move
            self.last_move_label.setText(f"最近一手：第 {r + 1} 行第 {c + 1} 列")

        if result.current_turn is None:
            self.turn_label.setText("当前轮次：无法判断")
        else:
            self.turn_label.setText(f"当前轮次：{'黑方' if result.current_turn == Piece.BLACK else '白方'}")

        if result.board_changed:
            self.suggested_move = None
            self.last_suggested_hash = None
            if self.mode != RunMode.MANUAL:
                self.last_infer_request_hash = None

        self.render_preview()
        self.update_phase_label()
        self.refresh_controls()
        self.update_overlay()

        if self.pending_manual_request:
            self.maybe_submit_infer_job(request_manual_infer=True)
        else:
            self.maybe_submit_infer_job(request_manual_infer=False)

    def consume_infer_result(self) -> None:
        if self.infer_future is None or not self.infer_future.done():
            return

        future = self.infer_future
        self.infer_future = None
        self.infer_busy = False

        try:
            result: InferResult = future.result()
        except Exception as exc:
            logging.exception("consume_infer_result failed")
            self.set_status(f"推理线程异常：{exc}")
            return

        if result.generation != self.run_generation:
            return
        if not result.ok:
            if result.error:
                self.set_status(result.error)
            return

        if result.predicted:
            self.suggested_move = result.move
            self.last_suggested_hash = result.board_hash_value
            self.model_label.setText(f"模型建议：{result.model_msg or '-'}")
            self.render_preview()
            self.update_overlay()

        if result.status_msg:
            self.set_status(result.status_msg)

        if result.click_requested and result.move is not None and result.clicked_screen_pos is not None:
            self.start_click_task(result.move, result.clicked_screen_pos, result.reference_board_after_click)

    def consume_background_results(self) -> None:
        self.consume_click_result()
        self.consume_worker_result()
        self.consume_infer_result()


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="screen_monitor_demo")
def main(cfg: DictConfig) -> None:
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)

    model_manager = MCTSManager(cfg)
    try:
        model_manager.load_from_cfg()
    except Exception as exc:
        logging.warning("Failed to load model from cfg: %s", exc)

    app = QApplication(sys.argv)
    widget = ScreenMonitorWidget(cfg=cfg, model_manager=model_manager)

    window = QMainWindow()
    window.setWindowTitle("screen_monitor_v12")
    window.resize(410, 760)
    window.setFixedWidth(410)
    window.setWindowFlag(Qt.WindowStaysOnTopHint, True)
    window.setCentralWidget(widget)
    status_bar = QStatusBar(window)
    window.setStatusBar(status_bar)

    
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
