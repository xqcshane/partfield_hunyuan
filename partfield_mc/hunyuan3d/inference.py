from __future__ import annotations

from pathlib import Path
import os


def _maybe_local_model(hunyuan_repo: str | Path | None, subfolder: str, fallback_repo: str) -> tuple[str, str, str | None]:
    if hunyuan_repo:
        repo = Path(hunyuan_repo).expanduser().resolve()
        candidate = repo / "models"
        if (candidate / subfolder).exists():
            return str(candidate), subfolder, "fp16"
    return fallback_repo, subfolder, "fp16"


def generate_mesh(
    image: str | Path | None,
    output: str | Path,
    hunyuan_repo: str | Path | None = None,
    images: dict[str, str | Path] | None = None,
    texture: bool = False,
    multiview: bool = False,
) -> Path:
    """Generate a GLB mesh using official Hunyuan3D Python APIs.

    Supports:
    - single-image geometry
    - single-image textured mesh
    - true multi-view geometry (front/left/back/right image dict)
    - true multi-view textured mesh (geometry from multi-view, texture from front view)
    """
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if hunyuan_repo:
        repo = Path(hunyuan_repo).expanduser().resolve()
        os.environ.setdefault("PYTHONPATH", str(repo))

    try:
        import torch
        from PIL import Image
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dgen.rembg import BackgroundRemover
        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
        except Exception:
            Hunyuan3DPaintPipeline = None
    except ImportError as e:
        raise RuntimeError(
            "Cannot import Hunyuan3D. Activate the Hunyuan3D environment or install Hunyuan3D-2 dependencies."
        ) from e

    if multiview and not images:
        raise ValueError("Multi-view generation requires --image-front/--image-left/--image-back or other multi-view inputs")
    if not multiview and not image:
        raise ValueError("Single-image generation requires --image")

    def _load_image(pathlike: str | Path):
        path = Path(pathlike).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        img = Image.open(path).convert("RGBA")
        # Example scripts run background removal for RGB images. Here convert directly to RGBA
        # and fall back to rembg only if alpha is fully opaque.
        if all(extrema == (255, 255) for extrema in [img.getchannel('A').getextrema()]):
            try:
                rembg = BackgroundRemover()
                img = rembg(img)
            except Exception:
                pass
        return img

    if multiview:
        prepared = {k: _load_image(v) for k, v in (images or {}).items()}
        model_path, subfolder, variant = _maybe_local_model(hunyuan_repo, 'hunyuan3d-dit-v2-mv', 'tencent/Hunyuan3D-2mv')
        print(f"[Hunyuan3D] Loading multi-view model from {model_path}")
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path, subfolder=subfolder, variant=variant)
        print("[Hunyuan3D] Generating multi-view mesh...")
        mesh = pipeline(
            image=prepared,
            num_inference_steps=50,
            octree_resolution=380,
            num_chunks=20000,
            generator=torch.manual_seed(12345),
            output_type='trimesh',
        )[0]
        if texture:
            if Hunyuan3DPaintPipeline is None:
                raise RuntimeError("Texture generation requested but Hunyuan3DPaintPipeline is unavailable")
            paint_model, _, _ = _maybe_local_model(hunyuan_repo, 'hunyuan3d-paint-v2-0', 'tencent/Hunyuan3D-2')
            print(f"[Hunyuan3D] Loading texture model from {paint_model}")
            tex_pipeline = Hunyuan3DPaintPipeline.from_pretrained(paint_model)
            front = prepared.get('front') or next(iter(prepared.values()))
            print("[Hunyuan3D] Baking texture from reference image...")
            mesh = tex_pipeline(mesh, image=front)
    else:
        img = _load_image(image)
        model_path, subfolder, variant = _maybe_local_model(hunyuan_repo, 'hunyuan3d-dit-v2-0', 'tencent/Hunyuan3D-2')
        print(f"[Hunyuan3D] Loading model from {model_path}")
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path, subfolder=subfolder, variant=variant)
        print("[Hunyuan3D] Generating mesh...")
        mesh = pipeline(image=img, output_type='trimesh')[0]
        if texture:
            if Hunyuan3DPaintPipeline is None:
                raise RuntimeError("Texture generation requested but Hunyuan3DPaintPipeline is unavailable")
            paint_model, _, _ = _maybe_local_model(hunyuan_repo, 'hunyuan3d-paint-v2-0', 'tencent/Hunyuan3D-2')
            print(f"[Hunyuan3D] Loading texture model from {paint_model}")
            tex_pipeline = Hunyuan3DPaintPipeline.from_pretrained(paint_model)
            print("[Hunyuan3D] Baking texture from reference image...")
            mesh = tex_pipeline(mesh, image=img)

    mesh.export(str(output))
    if not output.exists():
        raise RuntimeError(f"Hunyuan3D finished but GLB was not created: {output}")
    print(f"[Hunyuan3D] Saved: {output}")
    return output
