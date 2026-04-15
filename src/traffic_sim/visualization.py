from __future__ import annotations

import pygame

from .config import SimulationConfig
from .models import CellType, LightPhase


BACKGROUND = (245, 242, 232)
ROAD = (94, 103, 110)
INTERSECTION = (62, 70, 76)
VEHICLE = (196, 62, 62)
GRID_LINE = (220, 216, 205)
TEXT = (35, 38, 41)
GREEN = (51, 145, 92)
RED = (182, 59, 59)


class TrafficVisualizer:
    """Render the simulation grid and summary metrics with pygame."""

    def __init__(self, config: SimulationConfig) -> None:
        """Create the pygame window, clock, and font resources."""

        pygame.init()
        self.config = config
        self.surface = pygame.display.set_mode(
            (config.grid_width * config.cell_size, config.grid_height * config.cell_size + 104)
        )
        pygame.display.set_caption("Traffic Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Menlo", 18)
        self.small_font = pygame.font.SysFont("Menlo", 14)

    def draw(self, simulation, metrics) -> bool:
        """Draw one frame and return False when the user closes the window."""

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        self.surface.fill(BACKGROUND)
        self._draw_grid(simulation)
        self._draw_status_bar(simulation, metrics)
        self._draw_legend(simulation)
        pygame.display.flip()
        self.clock.tick(self.config.fps)
        return True

    def _draw_grid(self, simulation) -> None:
        """Render roads, intersections, lights, and vehicles onto the main surface."""

        light_phase = simulation.current_light_phase() if hasattr(simulation, "current_light_phase") else None
        occupancy = simulation.occupancy_cpu() if hasattr(simulation, "occupancy_cpu") else simulation.occupancy
        grid = simulation.grid_cpu() if hasattr(simulation, "grid_cpu") else simulation.grid

        for y in range(self.config.grid_height):
            for x in range(self.config.grid_width):
                rect = pygame.Rect(
                    x * self.config.cell_size,
                    y * self.config.cell_size,
                    self.config.cell_size,
                    self.config.cell_size,
                )
                cell = int(grid[y, x])
                color = BACKGROUND
                if cell in (CellType.ROAD_HORIZONTAL, CellType.ROAD_VERTICAL):
                    color = ROAD
                elif cell == CellType.INTERSECTION:
                    color = INTERSECTION

                pygame.draw.rect(self.surface, color, rect)
                pygame.draw.rect(self.surface, GRID_LINE, rect, 1)

                if cell == CellType.INTERSECTION and light_phase is not None:
                    light_color = GREEN if light_phase == LightPhase.HORIZONTAL_GREEN else RED
                    pygame.draw.circle(self.surface, light_color, rect.center, max(2, self.config.cell_size // 4))

                if occupancy[y, x] != -1:
                    inset = max(2, self.config.cell_size // 6)
                    pygame.draw.rect(self.surface, VEHICLE, rect.inflate(-inset, -inset))

    def _draw_status_bar(self, simulation, metrics) -> None:
        """Render a compact text summary of the latest simulation metrics."""

        base_y = self.config.grid_height * self.config.cell_size
        status_rect = pygame.Rect(0, base_y, self.surface.get_width(), 48)
        pygame.draw.rect(self.surface, (232, 228, 216), status_rect)

        text = (
            f"step={metrics.steps}  "
            f"active={metrics.active_vehicles}  "
            f"completed={metrics.completed_vehicles}  "
            f"avg_queue={metrics.average_queue_length:.3f}"
        )
        rendered = self.font.render(text, True, TEXT)
        self.surface.blit(rendered, (10, base_y + 14))

    def _draw_legend(self, simulation) -> None:
        """Render a legend explaining the visible map elements and signals."""

        base_y = self.config.grid_height * self.config.cell_size + 48
        legend_rect = pygame.Rect(0, base_y, self.surface.get_width(), 56)
        pygame.draw.rect(self.surface, (241, 237, 227), legend_rect)

        phase = simulation.current_light_phase() if hasattr(simulation, "current_light_phase") else None
        phase_label = "horizontal traffic moves" if phase == LightPhase.HORIZONTAL_GREEN else "vertical traffic moves"
        entries = [
            (ROAD, "road"),
            (INTERSECTION, "intersection"),
            (VEHICLE, "vehicle"),
            (GREEN, "green light"),
            (RED, "red light"),
        ]

        x = 10
        y = base_y + 10
        for color, label in entries:
            swatch = pygame.Rect(x, y, 16, 16)
            pygame.draw.rect(self.surface, color, swatch)
            pygame.draw.rect(self.surface, TEXT, swatch, 1)
            rendered = self.small_font.render(label, True, TEXT)
            self.surface.blit(rendered, (x + 24, y - 1))
            x += 120

        info = self.small_font.render(f"Current green phase: {phase_label}", True, TEXT)
        self.surface.blit(info, (10, base_y + 32))

    def close(self) -> None:
        """Release pygame resources and close the visualization window."""

        pygame.quit()
