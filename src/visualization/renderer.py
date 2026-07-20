import cv2
import numpy as np

class VisualizationRenderer:
    def __init__(self, opacity=0.25):
        """
        Initializes the renderer.
        opacity: translucent overlay blending factor (0.0 to 1.0).
        """
        self.opacity = opacity

        # Risk colors in BGR (OpenCV uses BGR)
        self.colors = {
            "GREEN": (46, 204, 113),   # Modern Emerald Green
            "YELLOW": (241, 196, 15),  # Sunflower Yellow
            "ORANGE": (230, 126, 34),  # Carrot Orange
            "RED": (231, 76, 60),      # Alizarin Red
            "STATIC": (149, 165, 166)  # Asbestos Grey
        }

    def render_overlay(self, frame, grids, grid_metrics, active_alerts_count=0):
        """
        Draws overlapping grids, flow arrows, risk map and legend on the frame.
        frame: BGR image frame.
        grids: list of grid cells.
        grid_metrics: aggregated metrics per grid (including risk_level, mean_dx, mean_dy, etc.).
        """
        overlay = frame.copy()
        output_frame = frame.copy()

        # 1. Draw Translucent Grid Risk Map
        for g in grids:
            grid_id = g["grid_id"]
            if grid_id not in grid_metrics:
                continue

            metrics = grid_metrics[grid_id]
            risk_lvl = metrics.get("risk_level", "GREEN")
            color = self.colors.get(risk_lvl, self.colors["GREEN"])

            x1, y1, x2, y2 = g["x1"], g["y1"], g["x2"], g["y2"]

            # Fill the cell on the overlay image
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        # Blend the overlay with the original frame for transparency
        cv2.addWeighted(overlay, self.opacity, output_frame, 1.0 - self.opacity, 0, output_frame)

        # 2. Draw Grid Borders and Flow Arrows
        for g in grids:
            grid_id = g["grid_id"]
            if grid_id not in grid_metrics:
                continue

            metrics = grid_metrics[grid_id]
            risk_lvl = metrics.get("risk_level", "GREEN")
            color = self.colors.get(risk_lvl, self.colors["GREEN"])

            x1, y1, x2, y2 = g["x1"], g["y1"], g["x2"], g["y2"]

            # Draw thin border
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 1)

            # Center of grid cell
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Draw dominant flow arrow
            mean_dx = metrics.get("mean_dx", 0.0)
            mean_dy = metrics.get("mean_dy", 0.0)
            mag = metrics.get("motion_magnitude", 0.0)

            if mag > 0.5:
                # Scale the arrow for visualization (arrow length proportional to magnitude)
                scale = 6.0
                end_x = int(cx + mean_dx * scale)
                end_y = int(cy + mean_dy * scale)

                # Clip coordinates to prevent drawing out of bounds
                end_x = max(min(end_x, x2), x1)
                end_y = max(min(end_y, y2), y1)

                # Draw the arrowed line
                # High-contrast arrow color (white with black border, or just black for red, etc.)
                arrow_color = (255, 255, 255) if risk_lvl != "YELLOW" else (0, 0, 0)
                cv2.arrowedLine(output_frame, (cx, cy), (end_x, end_y), arrow_color, 2, tipLength=0.3)

            # Draw count or Grid ID text (clean and small)
            count_val = metrics.get("crowd_count", 0)
            if count_val > 0:
                text = f"{int(count_val)}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.35
                thickness = 1
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = x1 + 5
                text_y = y1 + text_size[1] + 5
                # Background block for text legibility
                cv2.rectangle(output_frame, (text_x - 2, text_y - text_size[1] - 2),
                              (text_x + text_size[0] + 2, text_y + 2), (0, 0, 0), -1)
                cv2.putText(output_frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # 3. Draw Legend HUD on the bottom-left corner
        self._draw_hud(output_frame, active_alerts_count)

        return output_frame

    def _draw_hud(self, frame, active_alerts_count):
        """
        Draws a professional legend HUD on the frame.
        """
        h, w = frame.shape[:2]
        hud_w, hud_h = 240, 160
        margin = 15

        # Position at bottom-left corner
        bx1 = margin
        by1 = h - hud_h - margin
        bx2 = bx1 + hud_w
        by2 = h - margin

        # Draw translucent background panel
        hud_panel = frame[by1:by2, bx1:bx2].copy()
        cv2.rectangle(hud_panel, (0, 0), (hud_w, hud_h), (20, 20, 20), -1)
        cv2.addWeighted(hud_panel, 0.8, frame[by1:by2, bx1:bx2], 0.2, 0, frame[by1:by2, bx1:bx2])

        # Draw borders around HUD panel
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (80, 80, 80), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Draw Title
        cv2.putText(frame, "SUDHARSHAN FLOW INTELLIGENCE", (bx1 + 10, by1 + 20), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (bx1 + 10, by1 + 27), (bx2 - 10, by1 + 27), (100, 100, 100), 1)

        # Draw Risk Colors legend
        risks = [
            ("Green: Normal", self.colors["GREEN"]),
            ("Yellow: Slowing", self.colors["YELLOW"]),
            ("Orange: Congestion", self.colors["ORANGE"]),
            ("Red: Danger", self.colors["RED"])
        ]

        y_offset = by1 + 45
        for text, bgr_color in risks:
            # Draw color square
            cv2.rectangle(frame, (bx1 + 15, y_offset - 8), (bx1 + 25, y_offset), bgr_color, -1)
            cv2.rectangle(frame, (bx1 + 15, y_offset - 8), (bx1 + 25, y_offset), (255, 255, 255), 1)
            cv2.putText(frame, text, (bx1 + 35, y_offset - 1), font, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
            y_offset += 16

        # Draw Alerts status
        alert_color = self.colors["RED"] if active_alerts_count > 0 else self.colors["GREEN"]
        cv2.putText(frame, f"Active Alerts: {active_alerts_count}", (bx1 + 15, y_offset + 10), font, 0.38, alert_color, 1, cv2.LINE_AA)

        # Draw System Status Indicator
        cv2.putText(frame, "System: OFFLINE OPERATIONAL", (bx1 + 15, y_offset + 26), font, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
