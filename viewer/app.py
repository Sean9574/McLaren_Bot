"""
viewer/app.py — Interactive 3D panorama viewer for fall-risk assessment.

Places the user "inside" a sphere textured with the room's 360 panorama
(click-drag to look around, scroll to zoom — like Google Street View).
Fall-risk hazards detected by the analysis pipeline are shown as colored
markers floating in the scene; clicking one opens a detail panel with the
HOME FAST category, risk description, and suggested fix.

Data sources (from session.outputs_dir):
  - panorama.jpg   : equirectangular 360 image (required)
  - hazards.json   : list of hazard dicts (optional — demo hazards used
                     if missing, so the viewer is testable before the
                     segmentation/analysis stages are built)

hazards.json format (written by analysis/home_fast.py):
[
  {
    "label": "Loose throw rug",
    "severity": "high",          # high|medium|low|safe|normal
    "lon_deg": 12.5,              # horizontal angle, -180..180
    "lat_deg": -15.0,             # vertical angle,   -90..90
    "category": "floors",         # HOME FAST domain
    "description": "Unsecured rug near walkway — trip hazard.",
    "recommendation": "Remove or secure with non-slip backing."
  },
  ...
]
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering

# ── Geometry helpers ────────────────────────────────────────────────────
#
# The stitched panorama is NOT a standard 360x180 equirectangular image —
# OpenCV's stitcher output is a wide, short crop (e.g. ~9173x1797, a ~5:1
# aspect ratio) covering only the camera's actual pan/tilt sweep.
#
# Mapping that directly onto a full sphere (which assumes 360deg horizontal
# x 180deg vertical) would vertically stretch the image ~2.5x, causing
# severe distortion.
#
# Instead we build a curved "screen" — a partial cylinder — sized to the
# REAL field of view: horizontal extent = the camera's pan sweep (from
# config), vertical extent DERIVED from the image aspect ratio (assumes
# ~equal degrees/pixel in both axes, true for OpenCV's warper output).

def make_panorama_cylinder(image_w: int, image_h: int, hfov_deg: float,
                           radius: float = 10.0,
                           h_segments: int = 120,
                           v_segments: int = 40) -> tuple:
    """
    Build a curved screen matching the panorama's real angular extent.
    Returns (mesh, vfov_deg).
    """
    vfov_deg = hfov_deg * (image_h / image_w)
    hfov = math.radians(hfov_deg)
    vfov = math.radians(vfov_deg)

    # Reversed theta order fixes left/right mirroring: image column 0
    # (left edge) must map to the viewer's left side. Given Open3D's
    # look_at(forward=+Z, up=+Y), the camera's "right" is world -X, so
    # image-left (u=0) needs world +X -> theta=+hfov/2 at u=0.
    thetas = np.linspace(hfov / 2, -hfov / 2, h_segments + 1)
    lats   = np.linspace(vfov / 2, -vfov / 2, v_segments + 1)  # top -> bottom

    verts = np.zeros(((h_segments + 1) * (v_segments + 1), 3))
    uvs_grid = np.zeros(((h_segments + 1) * (v_segments + 1), 2))
    cols = h_segments + 1
    for j, lat in enumerate(lats):
        for i, th in enumerate(thetas):
            idx = j * cols + i
            x = radius * math.sin(th)
            z = radius * math.cos(th)
            y = radius * math.tan(lat)
            verts[idx] = (x, y, z)
            u = i / h_segments
            # Flip v: image row 0 (top) -> v=1, because GL texture origin
            # is bottom-left but image data is stored top-to-bottom.
            v = 1.0 - j / v_segments
            uvs_grid[idx] = (u, v)

    tri_list, uv_list = [], []
    for j in range(v_segments):
        for i in range(h_segments):
            a = j * cols + i
            b = a + 1
            c = a + cols
            d = c + 1
            # Inward-facing winding (visible from inside the curve)
            for tri in ((a, c, b), (b, c, d)):
                tri_list.append(tri)
                for vi in tri:
                    uv_list.append(uvs_grid[vi])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(tri_list))
    mesh.triangle_uvs = o3d.utility.Vector2dVector(np.array(uv_list))
    mesh.compute_vertex_normals()
    return mesh, vfov_deg


def angle_to_point(lon_deg: float, lat_deg: float,
                   radius: float = 9.5) -> np.ndarray:
    """
    Convert (lon, lat) in degrees to a 3D point on the curved screen.
    lon=0,lat=0 is straight ahead. lon>0 = toward the LEFT of the panorama
    image (matches the reversed-theta mapping in make_panorama_cylinder).
    """
    th = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    x = radius * math.sin(th)
    z = radius * math.cos(th)
    y = radius * math.tan(lat)
    return np.array([x, y, z])


# ── Demo data (used if hazards.json doesn't exist yet) ─────────────────

DEMO_HAZARDS = [
    {
        "label": "Loose floor cable",
        "severity": "high",
        "lon_deg": -40.0, "lat_deg": -20.0,
        "category": "floors",
        "description": "An unsecured cable crosses a walkway — significant "
                       "trip hazard, especially when entering/exiting the room.",
        "recommendation": "Route cable along the wall and secure with cable "
                          "covers or clips.",
    },
    {
        "label": "Storage shelf — items at height",
        "severity": "medium",
        "lon_deg": 70.0, "lat_deg": 5.0,
        "category": "storage",
        "description": "Frequently-used items stored above shoulder height "
                       "require reaching/climbing, increasing fall risk.",
        "recommendation": "Relocate commonly used items to between waist "
                          "and shoulder height.",
    },
    {
        "label": "Office chair with wheels",
        "severity": "medium",
        "lon_deg": 150.0, "lat_deg": -25.0,
        "category": "furniture",
        "description": "Wheeled chairs can roll unexpectedly, providing "
                       "unstable support if used to brace or stand.",
        "recommendation": "Use only stable, fixed seating in walkways.",
    },
    {
        "label": "Clear pathway",
        "severity": "safe",
        "lon_deg": 0.0, "lat_deg": -30.0,
        "category": "mobility_paths",
        "description": "This pathway is clear and well-lit — meets minimum "
                       "clearance for mobility aids.",
        "recommendation": "No action needed — maintain clear access.",
    },
]


SEVERITY_COLORS = {
    "high":   [1.0, 0.18, 0.18],
    "medium": [1.0, 0.65, 0.0],
    "low":    [1.0, 1.0, 0.0],
    "safe":   [0.2, 0.9, 0.4],
    "normal": [0.7, 0.7, 0.7],
}


# ── Main application ────────────────────────────────────────────────────

class RoomViewerApp:
    """360 panorama viewer with clickable fall-risk hazard markers."""

    def __init__(self, session, config: dict):
        self.session = session
        self.config = config
        self.viewer_cfg = config.get("viewer", {})
        self.colors = {**SEVERITY_COLORS,
                       **self.viewer_cfg.get("colors", {})}

        self.hazards = self._load_hazards()
        self.marker_radius = 9.5
        self.sphere_radius = 10.0

        self.app = gui.Application.instance
        self.app.initialize()

        title = self.viewer_cfg.get("window_title",
                                     "McLaren Fall Risk Assessment")
        self.window = self.app.create_window(title, 1280, 800)

        self._build_ui()
        self._populate_scene()

    # -- Data --------------------------------------------------------------

    def _load_hazards(self) -> List[dict]:
        hazards_path = self.session.outputs_dir / "hazards.json"
        if hazards_path.exists():
            try:
                data = json.loads(hazards_path.read_text())
                print(f"[Viewer] Loaded {len(data)} hazards from "
                      f"{hazards_path.name}")
                return data
            except Exception as e:
                print(f"[Viewer] Failed to load hazards.json: {e}")
        print("[Viewer] No hazards.json found — using demo hazards "
              "(run the 'analysis' stage to generate real data)")
        return DEMO_HAZARDS

    # -- UI layout -----------------------------------------------------------

    def _build_ui(self):
        em = self.window.theme.font_size
        w = self.window

        # 3D scene widget (takes most of the window)
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(w.renderer)
        bg = self.viewer_cfg.get("background_color", [0.05, 0.05, 0.07])
        self.scene_widget.scene.set_background([*bg, 1.0])
        self.scene_widget.scene.set_lighting(
            rendering.Open3DScene.LightingProfile.NO_SHADOWS, [0, 0, 0])

        # Side info panel
        self.panel = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
        self.panel.background_color = gui.Color(0.12, 0.12, 0.14)

        self.title_label = gui.Label("McLaren Fall Risk Assessment")
        self.title_label.text_color = gui.Color(1, 1, 1)
        self.panel.add_child(self.title_label)
        self.panel.add_fixed(0.5 * em)

        self.hint_label = gui.Label(
            "Drag to look around. Click a marker for details.")
        self.hint_label.text_color = gui.Color(0.7, 0.7, 0.7)
        self.panel.add_child(self.hint_label)
        self.panel.add_fixed(1.0 * em)

        # Full panorama thumbnail (so the whole stitched image is visible
        # at a glance, alongside the immersive 3D view)
        pano_path = self.session.outputs_dir / "panorama.jpg"
        if pano_path.exists():
            self.panel.add_child(gui.Label("Full panorama:"))
            import cv2
            full = cv2.imread(str(pano_path))
            full_rgb = cv2.cvtColor(full, cv2.COLOR_BGR2RGB)
            thumb_w = 320
            scale = thumb_w / full_rgb.shape[1]
            thumb_h = max(1, int(full_rgb.shape[0] * scale))
            thumb = cv2.resize(full_rgb, (thumb_w, thumb_h))
            thumb_img = o3d.geometry.Image(np.ascontiguousarray(thumb))
            self.pano_thumb = gui.ImageWidget(thumb_img)
            self.panel.add_child(self.pano_thumb)
            self.panel.add_fixed(1.0 * em)

        # Legend
        self.panel.add_child(gui.Label("Legend:"))
        for sev, color in self.colors.items():
            row = gui.Horiz(0.3 * em)
            swatch = gui.Label("⬤")
            swatch.text_color = gui.Color(*color)
            row.add_child(swatch)
            row.add_child(gui.Label(sev.capitalize()))
            self.panel.add_child(row)
        self.panel.add_fixed(1.0 * em)

        # Hazard detail section (populated on click)
        self.detail_title = gui.Label("")
        self.detail_title.text_color = gui.Color(1, 1, 1)
        self.panel.add_child(self.detail_title)

        self.detail_severity = gui.Label("")
        self.panel.add_child(self.detail_severity)

        self.detail_category = gui.Label("")
        self.detail_category.text_color = gui.Color(0.7, 0.7, 0.7)
        self.panel.add_child(self.detail_category)
        self.panel.add_fixed(0.5 * em)

        self.detail_desc = gui.Label("")
        self.detail_desc.text_color = gui.Color(0.85, 0.85, 0.85)
        self.panel.add_child(self.detail_desc)
        self.panel.add_fixed(0.5 * em)

        self.detail_rec_header = gui.Label("")
        self.detail_rec_header.text_color = gui.Color(0.4, 0.9, 0.6)
        self.panel.add_child(self.detail_rec_header)

        self.detail_rec = gui.Label("")
        self.detail_rec.text_color = gui.Color(0.85, 0.85, 0.85)
        self.panel.add_child(self.detail_rec)

        # Hazard list (clickable)
        self.panel.add_fixed(1.0 * em)
        self.panel.add_child(gui.Label(f"All hazards ({len(self.hazards)}):"))
        self.hazard_list = gui.ListView()
        labels = [f"[{h['severity'].upper()}] {h['label']}"
                 for h in self.hazards]
        self.hazard_list.set_items(labels)
        self.hazard_list.set_on_selection_changed(self._on_list_select)
        self.panel.add_child(self.hazard_list)

        # Layout: scene on left, panel on right (fixed width)
        w.add_child(self.scene_widget)
        w.add_child(self.panel)

        self.scene_widget.set_on_mouse(self._on_mouse)
        w.set_on_layout(self._on_layout)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        panel_width = 22 * self.window.theme.font_size
        self.scene_widget.frame = gui.Rect(r.x, r.y,
                                           r.width - panel_width, r.height)
        self.panel.frame = gui.Rect(r.get_right() - panel_width, r.y,
                                    panel_width, r.height)

    # -- Scene content --------------------------------------------------------

    def _populate_scene(self):
        scene = self.scene_widget.scene

        # Panorama curved screen, sized to the camera's real FOV
        pano_path = self.session.outputs_dir / "panorama.jpg"
        cam_cfg = self.config.get("camera", {})
        hfov_deg = (cam_cfg.get("pan_max_deg", 170)
                    - cam_cfg.get("pan_min_deg", -170))

        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"

        if pano_path.exists():
            img = o3d.io.read_image(str(pano_path))
            arr = np.asarray(img)
            ih, iw = arr.shape[0], arr.shape[1]
            mat.albedo_img = img
            print(f"[Viewer] Loaded panorama: {pano_path} ({iw}x{ih})")
        else:
            iw, ih = 9173, 1797  # fallback aspect if missing
            print(f"[Viewer] WARNING: {pano_path} not found — "
                  f"screen will be untextured")

        screen, vfov_deg = make_panorama_cylinder(
            iw, ih, hfov_deg=hfov_deg, radius=self.sphere_radius)
        print(f"[Viewer] Panorama FOV: {hfov_deg:.0f} deg H x "
              f"{vfov_deg:.1f} deg V")
        self.hfov_deg = hfov_deg
        self.vfov_deg = vfov_deg

        scene.add_geometry("panorama_screen", screen, mat)

        # Hazard markers
        self._marker_names = []
        self._marker_positions = []
        marker_size = self.viewer_cfg.get("point_size", 2.0) * 0.15
        for i, hz in enumerate(self.hazards):
            pos = angle_to_point(hz.get("lon_deg", 0), hz.get("lat_deg", 0),
                                 radius=self.marker_radius)
            color = self.colors.get(hz.get("severity", "normal"),
                                    self.colors["normal"])

            marker = o3d.geometry.TriangleMesh.create_sphere(
                radius=marker_size, resolution=12)
            marker.translate(pos)
            marker.paint_uniform_color(color)
            marker.compute_vertex_normals()

            mmat = rendering.MaterialRecord()
            mmat.shader = "defaultLit"
            mmat.base_color = [*color, 1.0]

            name = f"hazard_{i}"
            scene.add_geometry(name, marker, mmat)
            self._marker_names.append(name)
            self._marker_positions.append(pos)

        # Camera: positioned at center, looking forward (theta=0).
        # Initial vertical FOV shows most of the screen's vertical extent.
        bounds = screen.get_axis_aligned_bounding_box()
        init_fov = min(90.0, max(30.0, vfov_deg + 10))
        self.scene_widget.setup_camera(init_fov, bounds, [0, 0, 0])
        self.scene_widget.scene.camera.look_at(
            [0, 0, 1], [0, 0, 0], [0, 1, 0])

    # -- Interaction -----------------------------------------------------------

    def _on_mouse(self, event):
        if (event.type == gui.MouseEvent.Type.BUTTON_DOWN
                and event.is_button_down(gui.MouseButton.LEFT)
                and event.is_modifier_down(gui.KeyModifier.CTRL) == False):
            # Try picking on plain left-click (not drag-to-look, which
            # Open3D's default camera controller handles itself when no
            # geometry is hit — so we only intercept if a marker is hit).
            self._try_pick(event.x, event.y)
            # Don't consume the event — let the camera controller also
            # process drags normally.
        return gui.SceneWidget.EventCallbackResult.IGNORED

    def _try_pick(self, mx: int, my: int):
        """Unproject the click into a ray and test against marker spheres."""
        if not self._marker_positions:
            return

        view = self.scene_widget.scene.camera
        frame = self.scene_widget.frame

        # Unproject near and far points at this pixel to build a ray
        near = view.unproject(mx, my, 0.0, frame.width, frame.height)
        far  = view.unproject(mx, my, 1.0, frame.width, frame.height)
        ray_o = np.array(near)
        ray_d = np.array(far) - ray_o
        norm = np.linalg.norm(ray_d)
        if norm < 1e-9:
            return
        ray_d /= norm

        marker_size = self.viewer_cfg.get("point_size", 2.0) * 0.15
        best_i, best_t = None, 1e9
        for i, pos in enumerate(self._marker_positions):
            # Ray-sphere intersection
            oc = ray_o - pos
            b = np.dot(oc, ray_d)
            c = np.dot(oc, oc) - marker_size * marker_size
            disc = b*b - c
            if disc < 0:
                continue
            t = -b - math.sqrt(disc)
            if 0 < t < best_t:
                best_t = t
                best_i = i

        if best_i is not None:
            self._show_hazard(best_i)
            self.hazard_list.selected_index = best_i

    def _on_list_select(self, new_val, is_dbl_click):
        idx = self.hazard_list.selected_index
        if 0 <= idx < len(self.hazards):
            self._show_hazard(idx)

    def _show_hazard(self, idx: int):
        hz = self.hazards[idx]
        self.detail_title.text = hz.get("label", "")
        sev = hz.get("severity", "normal")
        self.detail_severity.text = f"Severity: {sev.upper()}"
        self.detail_category.text = (
            f"HOME FAST category: {hz.get('category', '—')}")
        self.detail_desc.text = hz.get("description", "")
        self.detail_rec_header.text = "Recommendation:"
        self.detail_rec.text = hz.get("recommendation", "—")
        self.window.set_needs_layout()

    # -- Run --------------------------------------------------------------

    def run(self):
        self.app.run()


def launch_viewer(session, config: dict):
    """Pipeline entry point — called by `run_scan.py view --session NAME`."""
    pano_path = session.outputs_dir / "panorama.jpg"
    if not pano_path.exists():
        print(f"[Viewer] ERROR: no panorama found at {pano_path}")
        print("[Viewer] Run the stitch stage first:")
        print(f"  python run_scan.py process --session "
              f"{session.manifest.session_name} --stage stitch")
        return

    print("[Viewer] Launching 3D room viewer...")
    app = RoomViewerApp(session, config)
    app.run()