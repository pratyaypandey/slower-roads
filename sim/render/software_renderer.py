"""Pure numpy/PIL software renderer: sim state + road -> RGB chase-cam frame.

Sidesteps the native `gl` module entirely — needs only numpy + pillow, so it
runs anywhere Python does and gives us pixel training data today. It mirrors the
chase camera in sim/src/world.js (behind + above the car, looking ahead), and
draws the road as a filled ground polygon with a dashed centerline plus a simple
sky gradient. Deliberately low-fidelity: the world model distills this, so clean
flat shading beats detail.

Not a Three.js clone — a minimal rasterizer sufficient for a drivable view.
"""

import numpy as np

# Camera intrinsics/extrinsics chosen to match world.js: fov ~72deg, chase cam
# 8m back and 3.4m up, looking ~10m ahead of the car.
CAM_BACK = 8.0
CAM_HEIGHT = 3.4
CAM_LOOK_AHEAD = 10.0
FOV_Y = np.radians(72.0)


def _camera(car):
    x, z, heading = car["x"], car["z"], car["heading"]
    eye = np.array([
        x - np.sin(heading) * CAM_BACK,
        CAM_HEIGHT,
        z - np.cos(heading) * CAM_BACK,
    ])
    target = np.array([
        x + np.sin(heading) * CAM_LOOK_AHEAD,
        0.8,
        z + np.cos(heading) * CAM_LOOK_AHEAD,
    ])
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return eye, right, up, fwd


def _project(pts_world, eye, right, up, fwd, w, h):
    # World -> camera space, then perspective divide to pixel coords.
    rel = pts_world - eye
    cx = rel @ right
    cy = rel @ up
    cz = rel @ fwd  # depth along view direction
    f = (h / 2) / np.tan(FOV_Y / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        sx = w / 2 + f * cx / cz
        sy = h / 2 - f * cy / cz
    return sx, sy, cz


def _fill_poly(img, xs, ys, color):
    # Scanline fill of a convex quad given its 4 pixel corners.
    h, w = img.shape[:2]
    ymin = max(0, int(np.floor(ys.min())))
    ymax = min(h - 1, int(np.ceil(ys.max())))
    n = len(xs)
    for y in range(ymin, ymax + 1):
        xints = []
        for i in range(n):
            j = (i + 1) % n
            y0, y1 = ys[i], ys[j]
            if (y0 <= y < y1) or (y1 <= y < y0):
                t = (y - y0) / (y1 - y0)
                xints.append(xs[i] + t * (xs[j] - xs[i]))
        if len(xints) >= 2:
            x0, x1 = int(max(0, min(xints))), int(min(w - 1, max(xints)))
            img[y, x0:x1 + 1] = color


def render_frame(car, road, size=64):
    w = h = size
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Sky gradient (top) over ground (bottom); horizon ~mid-frame.
    for y in range(h):
        t = y / h
        if t < 0.5:
            img[y, :] = (135 - int(40 * t * 2), 183 - int(30 * t * 2), 224)
        else:
            img[y, :] = (63, 95, 46)  # ground green

    eye, right, up, fwd = _camera(car)
    pts = np.array(road["points"])
    head = np.array(road["headings"])
    half = road["width"] / 2

    # Only draw road quads near/ahead of the car (cheap culling by distance).
    car_xz = np.array([car["x"], car["z"]])
    dist = np.linalg.norm(pts - car_xz, axis=1)
    near = np.where(dist < 120)[0]
    if len(near):
        lo, hi = near.min(), min(near.max(), len(pts) - 2)
        for i in range(lo, hi):
            nx, nz = np.cos(head[i]), -np.sin(head[i])
            nx2, nz2 = np.cos(head[i + 1]), -np.sin(head[i + 1])
            quad = np.array([
                [pts[i, 0] - nx * half, 0.0, pts[i, 1] - nz * half],
                [pts[i, 0] + nx * half, 0.0, pts[i, 1] + nz * half],
                [pts[i + 1, 0] + nx2 * half, 0.0, pts[i + 1, 1] + nz2 * half],
                [pts[i + 1, 0] - nx2 * half, 0.0, pts[i + 1, 1] - nz2 * half],
            ])
            sx, sy, cz = _project(quad, eye, right, up, fwd, w, h)
            if np.all(cz > 0.1):  # fully in front of camera
                _fill_poly(img, sx, sy, np.array([53, 53, 59], dtype=np.uint8))

    return img


def render_manifest_sample(sample, road, size=64):
    return render_frame(sample["state"], road, size=size)
