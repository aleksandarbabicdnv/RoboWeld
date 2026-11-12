"""Isaac Sim side of the surface challenges and of Path A: author the sheet, put a template image
on it, and photograph it with a straight-down camera.

Only USD (pxr) and Replicator calls live here; the geometry comes from src/paper_surface.py and
the image mapping from src/template_strokes.py, both pure numpy and unit tested offline. Import
this module only inside a running Isaac Sim (SimulationApp started, or a Kit script).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from paper_surface import Surface, surface_mesh
from template_strokes import CameraMapping

SHEET_PRIM = "/World/ChallengeSheet"
TEMPLATE_PRIM = "/World/TemplateOverlay"
TEMPLATE_CAMERA_PRIM = "/World/TemplateCamera"
HORIZONTAL_APERTURE_MM = 20.955  # USD camera default; RTX fits the horizontal aperture


def _mesh(stage, path: str, points: np.ndarray, triangles: np.ndarray, color, uv: Optional[np.ndarray] = None):
    from pxr import Gf, Sdf, UsdGeom, Vt

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in points]))
    mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(triangles)))
    mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray([int(i) for i in np.asarray(triangles).ravel()]))
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
    if uv is not None:
        primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                         UsdGeom.Tokens.vertex)
        primvar.Set(Vt.Vec2fArray([Gf.Vec2f(float(a), float(b)) for a, b in uv]))
    return mesh


def author_sheet(stage, surface: Surface, size_xy: Sequence[float], *, collision: bool, margin_m: float = 0.02,
                 resolution_m: float = 0.004, path: str = SHEET_PRIM):
    """The challenge sheet as a triangle mesh (white), with an optional static triangle collider."""
    from pxr import UsdPhysics

    points, triangles, _ = surface_mesh(surface, size_xy, resolution_m, margin_m)
    mesh = _mesh(stage, path, points, triangles, (0.97, 0.97, 0.95))
    if collision:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set("none")
    return mesh


def apply_template(stage, surface: Surface, size_xy: Sequence[float], image_path: str, *,
                   lift_m: float = 0.0002, resolution_m: float = 0.002, path: str = TEMPLATE_PRIM):
    """Lay a template image over the drawing area, following the surface, lifted `lift_m` along the
    normal. The image's top edge is the +v edge of the drawing area (template_strokes convention)."""
    from pxr import Sdf, UsdShade

    points, triangles, uv = surface_mesh(surface, size_xy, resolution_m, 0.0)
    normals = surface.normals(uv)
    sx, sy = float(size_xy[0]), float(size_xy[1])
    st = np.c_[(uv[:, 0] + sx / 2.0) / sx, (uv[:, 1] + sy / 2.0) / sy]  # USD st: (0, 0) bottom-left
    mesh = _mesh(stage, path, points + normals * lift_m, triangles, (1.0, 1.0, 1.0), uv=st)

    material = UsdShade.Material.Define(stage, path + "_Material")
    surface_shader = UsdShade.Shader.Define(stage, path + "_Material/Surface")
    surface_shader.CreateIdAttr("UsdPreviewSurface")
    surface_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    surface_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    reader = UsdShade.Shader.Define(stage, path + "_Material/StReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    texture = UsdShade.Shader.Define(stage, path + "_Material/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(Path(image_path).resolve()))
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    surface_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(surface_shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def template_camera(stage, surface: Surface, size_xy: Sequence[float], *, height_above_sheet_m: float,
                    view_margin_m: float, border_m: float = 0.002,
                    path: str = TEMPLATE_CAMERA_PRIM) -> CameraMapping:
    """Straight-down camera over the drawing area; returns the pixel -> paper mapping that matches it."""
    from pxr import Gf, UsdGeom

    centre = surface.point([0.0, 0.0])
    eye = (float(surface.center_xy[0]), float(surface.center_xy[1]), float(centre[2] + height_above_sheet_m))
    view = float(max(size_xy)) + 2.0 * float(view_margin_m)
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr().Set(float(HORIZONTAL_APERTURE_MM * height_above_sheet_m / view))
    camera.CreateHorizontalApertureAttr().Set(float(HORIZONTAL_APERTURE_MM))
    camera.CreateVerticalApertureAttr().Set(float(HORIZONTAL_APERTURE_MM))  # square images
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
    look = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(eye[0], eye[1], float(centre[2])), Gf.Vec3d(0.0, 1.0, 0.0))
    xform = UsdGeom.Xformable(camera)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(look.GetInverse())
    return CameraMapping(eye, view, float(height_above_sheet_m), surface, tuple(size_xy), border_m=border_m)


def capture_rgb(camera_path: str, resolution: Tuple[int, int], update: Callable[[], None],
                warmup_frames: int = 90) -> np.ndarray:
    """Render `camera_path` at `resolution` (width, height) and return an (H, W, 3) uint8 image.

    `update` advances the app one frame (simulation_app.update in a standalone script). Textures
    stream in asynchronously, so the image is taken after `warmup_frames` frames, and only once
    two consecutive frames agree.
    """
    import omni.replicator.core as rep

    render_product = rep.create.render_product(camera_path, tuple(int(v) for v in resolution))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach([render_product])
    previous, image = None, None
    try:
        for frame in range(int(warmup_frames) + 60):
            update()
            data = annotator.get_data()
            if data is None or not getattr(data, "size", 0):
                continue
            current = np.asarray(data)[..., :3].copy()
            if frame >= warmup_frames and previous is not None and np.array_equal(current, previous):
                image = current
                break
            previous = current
        if image is None:
            image = previous
    finally:
        try:
            annotator.detach([render_product])
            render_product.destroy()
        except Exception:
            pass
    if image is None:
        raise RuntimeError(f"No frame rendered from {camera_path}")
    return image


def ensure_light(stage, path: str = "/World/TemplateLight", intensity: float = 1000.0) -> None:
    """An even dome light, unless the stage already has a light (a new stage has none, and an
    unlit sheet photographs black)."""
    from pxr import UsdLux

    for prim in stage.Traverse():
        if prim.HasAPI(UsdLux.LightAPI) or "Light" in prim.GetTypeName():
            return
    UsdLux.DomeLight.Define(stage, path).CreateIntensityAttr().Set(float(intensity))


def set_visible(stage, path: str, visible: bool) -> None:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(path)
    if prim and prim.IsValid():
        imageable = UsdGeom.Imageable(prim)
        imageable.MakeVisible() if visible else imageable.MakeInvisible()


def curve(stage, path: str, strokes: Sequence[np.ndarray], color, width: float) -> None:
    """3D polylines (e.g. the desired path on a tilted sheet) as linear BasisCurves."""
    from pxr import Gf, UsdGeom

    strokes = [np.asarray(s, dtype=np.float64) for s in strokes if len(s) >= 2]
    if not strokes:
        return
    curves = UsdGeom.BasisCurves.Define(stage, path)
    curves.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
    curves.CreateCurveVertexCountsAttr().Set([len(s) for s in strokes])
    curves.CreatePointsAttr().Set([Gf.Vec3f(*map(float, p)) for s in strokes for p in s])
    curves.CreateWidthsAttr().Set([float(width)])
    curves.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    curves.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
