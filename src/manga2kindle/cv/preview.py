"""Visual debug overlay renderer for manga panel detection and reading order."""

from __future__ import annotations

from pathlib import Path

import cv2

from manga2kindle.cv.io import read_image_bgr, write_image
from manga2kindle.models import PanelBox

BOX_COLOR_BGR = (0, 200, 0)
ARROW_COLOR_BGR = (0, 0, 255)
TEXT_COLOR_BGR = (255, 255, 255)


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

    def render(
        self,
        image_path: Path,
        panels: list[PanelBox],
        output_path: Path,
    ) -> Path:
        """Render panel bounding boxes, badges, and reading order arrows onto an image."""
        image = read_image_bgr(image_path)
        canvas = image.copy()

        if not panels:
            return write_image(output_path, canvas)

        height, width = canvas.shape[:2]

        for idx, panel in enumerate(panels, start=1):
            x1, y1 = panel.x, panel.y
            x2 = min(width - 1, panel.x + panel.w)
            y2 = min(height - 1, panel.y + panel.h)

            # Draw green bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), self.box_color, self.box_thickness)

            # Draw ordinal badge: e.g. [1], [2]
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
                self.box_color,
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

        # Draw red reading order arrows from center of panel i to i+1
        for i in range(len(panels) - 1):
            curr_box = panels[i]
            next_box = panels[i + 1]
            start_pt = (curr_box.x + curr_box.w // 2, curr_box.y + curr_box.h // 2)
            end_pt = (next_box.x + next_box.w // 2, next_box.y + next_box.h // 2)

            cv2.arrowedLine(
                canvas,
                start_pt,
                end_pt,
                self.arrow_color,
                self.arrow_thickness,
                tipLength=0.05,
            )

        return write_image(output_path, canvas)
