"""Visual debug overlay renderer for manga panel detection and reading order."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from manga2kindle.cv.io import read_image_bgr, write_image
from manga2kindle.models import DetectorKind, PanelBox

BOX_COLOR_BGR = (0, 200, 0)
ARROW_COLOR_BGR = (0, 0, 255)
TEXT_COLOR_BGR = (255, 255, 255)
BANNER_BG_BGR = (30, 30, 30)
BANNER_HEIGHT = 40

BACKEND_COLORS_BGR: dict[DetectorKind, tuple[int, int, int]] = {
    DetectorKind.CONTOUR: (0, 200, 0),
    DetectorKind.ONNX_YOLO: (255, 160, 0),
    DetectorKind.MAGI: (200, 0, 200),
    DetectorKind.AUTO: (0, 165, 255),
}


class PanelPreviewRenderer:
    """Renders debug overlays with bounding boxes, ordinal badges, and reading order arrows."""

    def __init__(
        self,
        box_color: tuple[int, int, int] = BOX_COLOR_BGR,
        arrow_color: tuple[int, int, int] = ARROW_COLOR_BGR,
        box_thickness: int = 3,
        arrow_thickness: int = 2,
    ) -> None:
        self.box_color = box_color
        self.arrow_color = arrow_color
        self.box_thickness = box_thickness
        self.arrow_thickness = arrow_thickness

    def _draw_overlays(
        self,
        canvas: np.ndarray,
        panels: list[PanelBox],
        *,
        box_color: tuple[int, int, int],
        arrow_color: tuple[int, int, int],
    ) -> np.ndarray:
        """Draw bounding boxes, [ordinal] badges, and reading-order arrows onto canvas."""
        if not panels:
            return canvas

        height, width = canvas.shape[:2]

        for idx, panel in enumerate(panels, start=1):
            x1, y1 = panel.x, panel.y
            x2 = min(width - 1, panel.x + panel.w)
            y2 = min(height - 1, panel.y + panel.h)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, self.box_thickness)

            ordinal = panel.ordinal if panel.ordinal > 0 else idx
            badge_text = f"[{ordinal}]"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(badge_text, font, font_scale, font_thickness)

            pad = 4
            badge_x1 = max(0, min(x1, width - 1))
            badge_y1 = max(0, min(y1, height - 1))
            badge_x2 = max(0, min(x1 + text_w + 2 * pad, width - 1))
            badge_y2 = max(0, min(y1 + text_h + 2 * pad, height - 1))

            cv2.rectangle(
                canvas,
                (badge_x1, badge_y1),
                (badge_x2, badge_y2),
                box_color,
                -1,
            )
            cv2.putText(
                canvas,
                badge_text,
                (badge_x1 + pad, badge_y1 + text_h + pad),
                font,
                font_scale,
                TEXT_COLOR_BGR,
                font_thickness,
                lineType=cv2.LINE_AA,
            )

        for i in range(len(panels) - 1):
            curr_box = panels[i]
            next_box = panels[i + 1]
            start_pt = (curr_box.x + curr_box.w // 2, curr_box.y + curr_box.h // 2)
            end_pt = (next_box.x + next_box.w // 2, next_box.y + next_box.h // 2)

            cv2.arrowedLine(
                canvas,
                start_pt,
                end_pt,
                arrow_color,
                self.arrow_thickness,
                tipLength=0.05,
            )

        return canvas

    def render(
        self,
        image_path: Path,
        panels: list[PanelBox],
        output_path: Path,
    ) -> Path:
        """Render panel bounding boxes, badges, and reading order arrows onto an image."""
        image = read_image_bgr(image_path)
        canvas = self._draw_overlays(
            image.copy(),
            panels,
            box_color=self.box_color,
            arrow_color=self.arrow_color,
        )
        return write_image(output_path, canvas)

    def render_comparison(
        self,
        image_path: Path,
        results: dict[DetectorKind, list[PanelBox]],
        output_path: Path,
    ) -> Path:
        """Render side-by-side comparison tiles for multiple detector backends."""
        if not results:
            raise ValueError("results must contain at least one backend.")

        image = read_image_bgr(image_path)
        tiles: list[np.ndarray] = []
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2

        for kind, panels in results.items():
            box_color = BACKEND_COLORS_BGR.get(kind, self.box_color)
            body = self._draw_overlays(
                image.copy(),
                panels,
                box_color=box_color,
                arrow_color=self.arrow_color,
            )
            tile_width = body.shape[1]
            banner = np.full((BANNER_HEIGHT, tile_width, 3), BANNER_BG_BGR, dtype=np.uint8)
            cv2.rectangle(banner, (0, BANNER_HEIGHT - 4), (tile_width - 1, BANNER_HEIGHT - 1), box_color, -1)

            label = f"{kind.value.upper()} ({len(panels)} panels)"
            cv2.putText(
                banner,
                label,
                (12, 26),
                font,
                font_scale,
                TEXT_COLOR_BGR,
                font_thickness,
                lineType=cv2.LINE_AA,
            )
            tiles.append(np.vstack([banner, body]))

        combined = np.hstack(tiles)
        return write_image(output_path, combined)
