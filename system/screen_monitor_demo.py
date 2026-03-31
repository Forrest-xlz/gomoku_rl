import enum
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageGrab
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QApplication,
    QPushButton,
    QPlainTextEdit,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)
from tensordict import TensorDict
from torchrl.data.tensor_specs import (
    BinaryDiscreteTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)

from gomoku_rl import CONFIG_PATH
from gomoku_rl.policy import Policy, get_policy


class Piece(enum.Enum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2


EMPTY = Piece.EMPTY.value
BLACK = Piece.BLACK.value
WHITE = Piece.WHITE.value


def grab_screen_bgr() -> np.ndarray:
    """Grab the full desktop as a BGR image."""
    try:
        img = ImageGrab.grab(all_screens=True)
    except TypeError:
        img = ImageGrab.grab()
    img = np.array(img)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def order_points(pts: np.ndarray) -> np.ndarray:
    """Return corners ordered as [top-left, top-right, bottom-left, bottom-right]."""
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
    """Perspective-transform the board ROI into a normalized square view."""
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
                markerSize=8,
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
                0.5,
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

    return vis


def infer_current_player(board: np.ndarray) -> Optional[Piece]:
    black_count = int((board == BLACK).sum())
    white_count = int((board == WHITE).sum())
    if black_count == white_count:
        return Piece.BLACK
    if black_count == white_count + 1:
        return Piece.WHITE
    return None


def make_model(cfg: DictConfig) -> Policy:
    board_size = cfg.board_size
    action_spec = DiscreteTensorSpec(
        board_size * board_size,
        shape=[1],
        device=cfg.device,
    )
    observation_spec = CompositeSpec(
        {
            "observation": UnboundedContinuousTensorSpec(
                device=cfg.device,
                shape=[2, 3, board_size, board_size],
            ),
            "action_mask": BinaryDiscreteTensorSpec(
                n=board_size * board_size,
                device=cfg.device,
                shape=[2, board_size * board_size],
                dtype=torch.bool,
            ),
        },
        shape=[2],
        device=cfg.device,
    )
    return get_policy(
        name=cfg.algo.name,
        cfg=cfg.algo,
        action_spec=action_spec,
        observation_spec=observation_spec,
        device=cfg.device,
    )


def load_policy(cfg: DictConfig, checkpoint_path: str) -> Policy:
    model = make_model(cfg)
    state_dict = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_model_input(
    board: np.ndarray,
    current_player: Piece,
    latest_move: Optional[tuple[int, int]],
    device: str,
) -> TensorDict:
    board_tensor = torch.tensor(board, dtype=torch.long, device=device)
    signed_board = torch.zeros_like(board_tensor)
    signed_board = torch.where(board_tensor == BLACK, torch.ones_like(signed_board), signed_board)
    signed_board = torch.where(board_tensor == WHITE, -torch.ones_like(signed_board), signed_board)

    piece_value = 1 if current_player == Piece.BLACK else -1
    layer_current = (signed_board == piece_value).float()
    layer_opponent = (signed_board == -piece_value).float()
    layer_last_move = torch.zeros_like(layer_current)
    if latest_move is not None:
        row, col = latest_move
        layer_last_move[row, col] = 1.0

    observation = torch.stack([layer_current, layer_opponent, layer_last_move], dim=0).unsqueeze(0)
    action_mask = (board_tensor == EMPTY).flatten().unsqueeze(0)

    return TensorDict(
        {
            "observation": observation,
            "action_mask": action_mask,
        },
        batch_size=1,
    )


class ModelManager:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.single_model: Optional[Policy] = None
        self.black_model: Optional[Policy] = None
        self.white_model: Optional[Policy] = None
        self.single_checkpoint: Optional[str] = None
        self.black_checkpoint: Optional[str] = None
        self.white_checkpoint: Optional[str] = None

    def load_from_cfg(self) -> None:
        if self.cfg.get("checkpoint"):
            self.load_single(self.cfg.checkpoint)
        if self.cfg.get("black_checkpoint"):
            self.load_black(self.cfg.black_checkpoint)
        if self.cfg.get("white_checkpoint"):
            self.load_white(self.cfg.white_checkpoint)

    def has_model(self) -> bool:
        return self.single_model is not None or (self.black_model is not None and self.white_model is not None)

    def load_single(self, checkpoint_path: str) -> None:
        self.single_model = load_policy(self.cfg, checkpoint_path)
        self.single_checkpoint = checkpoint_path
        logging.info("Loaded single checkpoint: %s", checkpoint_path)

    def load_black(self, checkpoint_path: str) -> None:
        self.black_model = load_policy(self.cfg, checkpoint_path)
        self.black_checkpoint = checkpoint_path
        logging.info("Loaded black checkpoint: %s", checkpoint_path)

    def load_white(self, checkpoint_path: str) -> None:
        self.white_model = load_policy(self.cfg, checkpoint_path)
        self.white_checkpoint = checkpoint_path
        logging.info("Loaded white checkpoint: %s", checkpoint_path)

    def _select_model(self, current_player: Piece) -> Optional[Policy]:
        if self.black_model is not None and self.white_model is not None:
            return self.black_model if current_player == Piece.BLACK else self.white_model
        return self.single_model

    def predict(
        self,
        board: np.ndarray,
        latest_move: Optional[tuple[int, int]],
    ) -> tuple[Optional[tuple[int, int]], str]:
        current_player = infer_current_player(board)
        if current_player is None:
            return None, "当前棋盘黑白子数量不合法，无法判断轮到谁。"

        model = self._select_model(current_player)
        if model is None:
            return None, "当前未加载模型，只完成棋盘识别。"

        if latest_move is None and self.cfg.get("require_last_move", False):
            return None, "模型需要 last_move 通道；请先标记最近一手。"

        td = build_model_input(
            board=board,
            current_player=current_player,
            latest_move=latest_move,
            device=self.cfg.device,
        )
        with torch.no_grad():
            out = model(td).cpu()
        action = int(out["action"].item())
        row = action // self.cfg.board_size
        col = action % self.cfg.board_size

        if latest_move is None:
            extra = "（未提供最近一手，last_move 通道置零）"
        else:
            extra = ""
        return (row, col), f"轮到{current_player.name.lower()}落子，建议第 {row + 1} 行第 {col + 1} 列{extra}"


class ScreenMonitorWidget(QWidget):
    def __init__(self, cfg: DictConfig, model_manager: ModelManager):
        super().__init__()
        self.cfg = cfg
        self.model_manager = model_manager

        self.roi: Optional[list[int]] = None
        self.corners_in_roi: Optional[np.ndarray] = None
        self.latest_move: Optional[tuple[int, int]] = None
        self.latest_warped: Optional[np.ndarray] = None
        self.latest_board: Optional[np.ndarray] = None

        self.preview_pixmap: Optional[QPixmap] = None

        self._build_ui()
        self.load_runtime_config()

    def _build_ui(self) -> None:
        root_layout = QHBoxLayout(self)

        left = QVBoxLayout()
        right = QVBoxLayout()
        root_layout.addLayout(left, stretch=0)
        root_layout.addLayout(right, stretch=1)

        title = QLabel("屏幕五子棋监控")
        title.setFont(QFont("Arial", 14))
        left.addWidget(title)

        self.btn_select_roi = QPushButton("1. 选择棋盘区域")
        self.btn_select_roi.clicked.connect(self.select_roi)
        left.addWidget(self.btn_select_roi)

        self.btn_select_corners = QPushButton("2. 标记四角交点")
        self.btn_select_corners.clicked.connect(self.select_corners)
        left.addWidget(self.btn_select_corners)

        self.btn_select_last_move = QPushButton("3. 标记最近一手（可选）")
        self.btn_select_last_move.clicked.connect(self.select_last_move)
        left.addWidget(self.btn_select_last_move)

        self.btn_infer = QPushButton("4. 推理当前画面")
        self.btn_infer.clicked.connect(self.infer_current)
        left.addWidget(self.btn_infer)

        self.status_label = QLabel("状态：等待设置")
        self.status_label.setWordWrap(True)
        left.addWidget(self.status_label)

        self.model_label = QLabel("模型建议：尚未推理")
        self.model_label.setWordWrap(True)
        left.addWidget(self.model_label)

        board_title = QLabel("棋盘矩阵 (0空,1黑,2白)")
        board_title.setFont(QFont("Arial", 11))
        left.addWidget(board_title)

        self.board_text = QPlainTextEdit()
        self.board_text.setReadOnly(True)
        self.board_text.setMinimumWidth(280)
        self.board_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left.addWidget(self.board_text, stretch=1)

        preview_title = QLabel("识别预览")
        preview_title.setFont(QFont("Arial", 13))
        right.addWidget(preview_title)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(640, 640)
        self.preview_label.setStyleSheet("background: #202020;")
        right.addWidget(self.preview_label, stretch=1)

    def set_status(self, text: str) -> None:
        self.status_label.setText(f"状态：{text}")

    @property
    def runtime_config_path(self) -> Path:
        return Path(self.cfg.runtime_config_path)

    def save_runtime_config(self) -> None:
        payload = {
            "roi": self.roi,
            "corners_in_roi": self.corners_in_roi.tolist() if self.corners_in_roi is not None else None,
            "latest_move": list(self.latest_move) if self.latest_move is not None else None,
        }
        self.runtime_config_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.set_status(f"运行配置已保存到 {self.runtime_config_path.resolve()}")

    def load_runtime_config(self) -> None:
        if not self.runtime_config_path.exists():
            self.set_status("未找到运行配置，请先选择棋盘区域。")
            return

        payload = json.loads(self.runtime_config_path.read_text(encoding="utf-8"))
        self.roi = payload.get("roi")
        corners = payload.get("corners_in_roi")
        self.corners_in_roi = np.array(corners, dtype=np.float32) if corners else None
        latest_move = payload.get("latest_move")
        self.latest_move = tuple(latest_move) if latest_move is not None else None

        parts: list[str] = []
        if self.roi:
            parts.append(f"ROI={self.roi}")
        if self.corners_in_roi is not None:
            parts.append("已加载四角交点")
        if self.latest_move is not None:
            parts.append(f"最近一手={self.latest_move}")
        self.set_status("；".join(parts) if parts else "运行配置为空")

    def capture_roi_image(self) -> Optional[np.ndarray]:
        if self.roi is None:
            return None
        screen = grab_screen_bgr()
        x, y, w, h = self.roi
        roi_img = screen[y : y + h, x : x + w].copy()
        if roi_img.size == 0:
            return None
        return roi_img

    def select_roi(self) -> None:
        self.set_status("正在截屏，请框选整个棋盘区域。")
        screen = grab_screen_bgr()
        cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
        roi = cv2.selectROI("Select ROI", screen, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select ROI")

        x, y, w, h = roi
        if w == 0 or h == 0:
            self.set_status("没有选中 ROI。")
            return

        self.roi = [int(x), int(y), int(w), int(h)]
        self.save_runtime_config()

    def select_corners(self) -> None:
        if self.roi is None:
            QMessageBox.warning(self, "提示", "请先选择棋盘区域。")
            return

        roi_img = self.capture_roi_image()
        if roi_img is None:
            QMessageBox.critical(self, "错误", "当前 ROI 超出屏幕范围，请重新选择。")
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
                return

        cv2.destroyWindow("Pick 4 Corners")
        self.corners_in_roi = order_points(np.array(points, dtype=np.float32))
        self.save_runtime_config()

    def select_last_move(self) -> None:
        if self.roi is None or self.corners_in_roi is None:
            QMessageBox.warning(self, "提示", "请先设置 ROI 和四角交点。")
            return

        result = self._infer_position(update_preview=False)
        if result is None:
            return
        warped, board, _debug_info = result

        vis = draw_result(
            warped=warped,
            board=board,
            board_size=self.cfg.board_size,
            cell=self.cfg.cell,
            margin=self.cfg.margin,
            latest_move=self.latest_move,
        )

        selection: dict[str, Optional[tuple[int, int]]] = {"move": None}

        def mouse_callback(event, px, py, flags, param):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            col = int(round((px - self.cfg.margin) / self.cfg.cell))
            row = int(round((py - self.cfg.margin) / self.cfg.cell))
            if not (0 <= row < self.cfg.board_size and 0 <= col < self.cfg.board_size):
                return
            if int(board[row, col]) == EMPTY:
                return
            selection["move"] = (row, col)

        cv2.namedWindow("Pick Last Move", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Pick Last Move", mouse_callback)
        self.set_status("请点击最近一手所在交点，Enter 确认，Esc 取消。")

        while True:
            show = vis.copy()
            if selection["move"] is not None:
                show = draw_result(
                    warped=warped,
                    board=board,
                    board_size=self.cfg.board_size,
                    cell=self.cfg.cell,
                    margin=self.cfg.margin,
                    latest_move=selection["move"],
                )
            cv2.imshow("Pick Last Move", show)
            key = cv2.waitKey(20) & 0xFF
            if key == 13 and selection["move"] is not None:
                break
            if key == 27:
                cv2.destroyWindow("Pick Last Move")
                self.set_status("已取消最近一手设置。")
                return

        cv2.destroyWindow("Pick Last Move")
        self.latest_move = selection["move"]
        self.save_runtime_config()

    def _infer_position(
        self,
        update_preview: bool = True,
    ) -> Optional[tuple[np.ndarray, np.ndarray, list[dict]]]:
        if self.roi is None or self.corners_in_roi is None:
            QMessageBox.warning(self, "提示", "请先设置棋盘区域并标记四角交点。")
            return None

        roi_img = self.capture_roi_image()
        if roi_img is None:
            QMessageBox.critical(self, "错误", "当前 ROI 超出屏幕范围，请重新设置。")
            return None

        warped, _ = warp_board(
            img=roi_img,
            corners=self.corners_in_roi,
            board_size=self.cfg.board_size,
            cell=self.cfg.cell,
            margin=self.cfg.margin,
        )
        board, debug_info = detect_board_state(
            warped=warped,
            board_size=self.cfg.board_size,
            cell=self.cfg.cell,
            margin=self.cfg.margin,
            black_thresh=self.cfg.black_diff_thresh,
            white_thresh=self.cfg.white_diff_thresh,
        )
        self.latest_warped = warped
        self.latest_board = board

        if update_preview:
            self.update_preview(
                draw_result(
                    warped=warped,
                    board=board,
                    board_size=self.cfg.board_size,
                    cell=self.cfg.cell,
                    margin=self.cfg.margin,
                    latest_move=self.latest_move,
                )
            )
        return warped, board, debug_info

    def infer_current(self) -> None:
        result = self._infer_position(update_preview=True)
        if result is None:
            return

        warped, board, debug_info = result
        self.board_text.setPlainText(
            "\n".join(
                " ".join(str(int(v)) for v in board[row])
                for row in range(board.shape[0])
            )
        )

        move, msg = self.model_manager.predict(board, self.latest_move)
        self.model_label.setText(f"模型建议：{msg}")

        if move is not None:
            row, col = move
            self.update_preview(
                draw_result(
                    warped=warped,
                    board=board,
                    board_size=self.cfg.board_size,
                    cell=self.cfg.cell,
                    margin=self.cfg.margin,
                    latest_move=self.latest_move,
                )
            )

        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "last_warped.png"), warped)
        cv2.imwrite(
            str(out_dir / "last_result.png"),
            draw_result(
                warped=warped,
                board=board,
                board_size=self.cfg.board_size,
                cell=self.cfg.cell,
                margin=self.cfg.margin,
                latest_move=self.latest_move,
            ),
        )
        (out_dir / "last_board_state.json").write_text(
            json.dumps(
                {
                    "roi": self.roi,
                    "corners_in_roi": self.corners_in_roi.tolist(),
                    "latest_move": list(self.latest_move) if self.latest_move is not None else None,
                    "board": board.tolist(),
                    "debug_info": debug_info,
                    "prediction": list(move) if move is not None else None,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        black_cnt = int((board == BLACK).sum())
        white_cnt = int((board == WHITE).sum())
        self.set_status(f"推理完成：黑棋 {black_cnt}，白棋 {white_cnt}")

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


@hydra.main(version_base=None, config_path=f"{CONFIG_PATH}", config_name="screen_monitor_demo")
def main(cfg: DictConfig) -> None:
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)

    model_manager = ModelManager(cfg)
    try:
        model_manager.load_from_cfg()
    except Exception as exc:
        logging.warning("Failed to load model from cfg: %s", exc)

    app = QApplication(sys.argv)
    widget = ScreenMonitorWidget(cfg=cfg, model_manager=model_manager)

    window = QMainWindow()
    window.setWindowTitle("screen_monitor_demo")
    window.resize(1280, 860)
    window.setCentralWidget(widget)
    status_bar = QStatusBar(window)
    window.setStatusBar(status_bar)

    menu = window.menuBar().addMenu("&Menu")

    def load_single_checkpoint() -> None:
        path = QFileDialog.getOpenFileName(window, "Open Checkpoint", filter="*.pt")[0]
        if not path:
            return
        try:
            model_manager.load_single(path)
            status_bar.showMessage(f"Single checkpoint: {path}")
        except Exception as exc:
            QMessageBox.critical(window, "错误", f"加载模型失败：{exc}")

    def load_black_checkpoint() -> None:
        path = QFileDialog.getOpenFileName(window, "Open Black Checkpoint", filter="*.pt")[0]
        if not path:
            return
        try:
            model_manager.load_black(path)
            status_bar.showMessage(f"Black checkpoint: {path}")
        except Exception as exc:
            QMessageBox.critical(window, "错误", f"加载黑棋模型失败：{exc}")

    def load_white_checkpoint() -> None:
        path = QFileDialog.getOpenFileName(window, "Open White Checkpoint", filter="*.pt")[0]
        if not path:
            return
        try:
            model_manager.load_white(path)
            status_bar.showMessage(f"White checkpoint: {path}")
        except Exception as exc:
            QMessageBox.critical(window, "错误", f"加载白棋模型失败：{exc}")

    open_single_action = QAction("&Open Single Checkpoint", window)
    open_single_action.triggered.connect(load_single_checkpoint)
    open_single_action.setShortcut(QKeySequence.Open)
    menu.addAction(open_single_action)

    open_black_action = QAction("Open &Black Checkpoint", window)
    open_black_action.triggered.connect(load_black_checkpoint)
    menu.addAction(open_black_action)

    open_white_action = QAction("Open &White Checkpoint", window)
    open_white_action.triggered.connect(load_white_checkpoint)
    menu.addAction(open_white_action)

    menu.addSeparator()

    save_runtime_action = QAction("&Save Runtime Config", window)
    save_runtime_action.triggered.connect(widget.save_runtime_config)
    menu.addAction(save_runtime_action)

    load_runtime_action = QAction("&Load Runtime Config", window)
    load_runtime_action.triggered.connect(widget.load_runtime_config)
    menu.addAction(load_runtime_action)

    infer_action = QAction("&Infer", window)
    infer_action.triggered.connect(widget.infer_current)
    infer_action.setShortcut("Ctrl+R")
    menu.addAction(infer_action)

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
