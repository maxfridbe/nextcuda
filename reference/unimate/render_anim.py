"""Renders frames of a GLB's skeletal animation (what `nextcuda model3d rig RIGGED.glb --model unimate` writes) to one
PNG: the mesh posed by linear blend skinning (as verify_anim.py), drawn from the side and from the front by a small
z-buffer rasterizer in numpy (flat Lambert shading, the skeleton over it in orange), the camera fixed over the whole
clip so the travel shows. No GPU, no viewer: for looking at a result, offline.

    python -I render_anim.py anim.glb out.png [--frames 0,15,30,45] [--size 320]

A GLB without an animation renders its rest pose (one column).
"""
import json
import struct
import sys
import zlib

import numpy as np


def load(path):
    b = open(path, "rb").read()
    assert b[:4] == b"glTF", "not a GLB"
    jl, _ = struct.unpack("<II", b[12:20])
    doc = json.loads(b[20:20 + jl])
    off = 20 + jl
    bl, _ = struct.unpack("<II", b[off:off + 8])
    return doc, b[off + 8:off + 8 + bl]


def accessor(doc, bin_, i):
    a = doc["accessors"][i]
    v = doc["bufferViews"][a["bufferView"]]
    n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[a["type"]]
    dt = {5126: "<f4", 5123: "<u2", 5125: "<u4", 5121: "u1"}[a["componentType"]]
    start = v.get("byteOffset", 0) + a.get("byteOffset", 0)
    return np.frombuffer(bin_, dtype=dt, count=a["count"] * n, offset=start).reshape(a["count"], n)


def quat_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def posed_frames(doc, bin_, which):
    """(faces, [positions a frame], [joint heads a frame], joint parents)"""
    prim = doc["meshes"][0]["primitives"][0]
    pos = accessor(doc, bin_, prim["attributes"]["POSITION"]).astype(np.float64)
    faces = accessor(doc, bin_, prim["indices"]).reshape(-1, 3).astype(np.int64) if "indices" in prim else np.arange(len(pos)).reshape(-1, 3)
    skins = doc.get("skins") or []
    anim = next((a for a in doc.get("animations", []) if a.get("name") == "skeleton"), None)
    if not skins or "JOINTS_0" not in prim["attributes"]:
        return faces, [pos], [np.zeros((0, 3))], []
    skin = skins[0]
    joints = skin["joints"]
    nodes = doc["nodes"]
    jnt = accessor(doc, bin_, prim["attributes"]["JOINTS_0"]).astype(int)
    wgt = accessor(doc, bin_, prim["attributes"]["WEIGHTS_0"]).astype(np.float64)
    ibm = accessor(doc, bin_, skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    parent = {}
    for i, n in enumerate(nodes):
        for c in n.get("children", []):
            parent[c] = i
    jparent = [joints.index(parent[j]) if parent.get(j) in joints else -1 for j in joints]
    rot, trans, frames = {}, {}, 1
    if anim:
        for ch in anim["channels"]:
            s = anim["samplers"][ch["sampler"]]
            out = accessor(doc, bin_, s["output"])
            (rot if ch["target"]["path"] == "rotation" else trans)[ch["target"]["node"]] = out
            frames = max(frames, len(out))
    which = [f for f in which if f < frames] or [0]

    def world(t):
        cache = {}

        def w(i):
            if i in cache:
                return cache[i]
            n = nodes[i]
            m = np.eye(4)
            m[:3, 3] = trans[i][t] if i in trans else n.get("translation", [0, 0, 0])
            m[:3, :3] = quat_mat(rot[i][t] if i in rot else n.get("rotation", [0, 0, 0, 1]))
            cache[i] = (w(parent[i]) @ m) if i in parent else m
            return cache[i]
        return [w(j) for j in joints]

    h = np.concatenate([pos, np.ones((len(pos), 1))], axis=1)
    out_p, out_j = [], []
    for t in which:
        wm = world(t)
        mats = np.stack([m @ ibm[k] for k, m in enumerate(wm)])
        acc = np.zeros((len(pos), 3))
        for k in range(4):
            acc += wgt[:, k:k + 1] * np.einsum("vij,vj->vi", mats[jnt[:, k]], h)[:, :3]
        out_p.append(acc)
        out_j.append(np.array([m[:3, 3] for m in wm]))
    return faces, out_p, out_j, jparent


def raster(pos, faces, heads, jparent, view, box, size):
    """One view: 'side' looks along -X (Z to the right), 'front' along -Z (X to the right); Y up"""
    lo, hi = box
    ax = 2 if view == "side" else 0
    depth_ax = 0 if view == "side" else 2
    span = max(hi[ax] - lo[ax], hi[1] - lo[1]) * 1.08
    cx, cy = (hi[ax] + lo[ax]) / 2, (hi[1] + lo[1]) / 2

    def px(p):
        return np.stack([(p[:, ax] - cx) / span * size + size / 2, size / 2 - (p[:, 1] - cy) / span * size], axis=1)

    img = np.full((size, size, 3), 238, np.float64)
    img[int(size / 2 - (lo[1] - cy) / span * size):, :] = 222  # the floor below the lowest point of the clip
    zb = np.full((size, size), -np.inf)
    s = px(pos)
    z = pos[:, depth_ax] * (1 if view == "side" else 1)
    tri = pos[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    light = np.array([0.4, 0.8, 0.45]) / np.linalg.norm([0.4, 0.8, 0.45])
    shade = 0.25 + 0.75 * np.abs(n @ light)
    for f in range(len(faces)):
        a, b, c = faces[f]
        p0, p1, p2 = s[a], s[b], s[c]
        x0, x1 = int(max(0, np.floor(min(p0[0], p1[0], p2[0])))), int(min(size - 1, np.ceil(max(p0[0], p1[0], p2[0]))))
        y0, y1 = int(max(0, np.floor(min(p0[1], p1[1], p2[1])))), int(min(size - 1, np.ceil(max(p0[1], p1[1], p2[1]))))
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        d = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
        if abs(d) < 1e-12:
            continue
        w1 = ((xs - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (ys - p0[1])) / d
        w2 = ((p1[0] - p0[0]) * (ys - p0[1]) - (xs - p0[0]) * (p1[1] - p0[1])) / d
        w0 = 1 - w1 - w2
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        zz = w0 * z[a] + w1 * z[b] + w2 * z[c]
        sub = zb[y0:y1 + 1, x0:x1 + 1]
        upd = inside & (zz > sub)
        sub[upd] = zz[upd]
        img[y0:y1 + 1, x0:x1 + 1][upd] = np.array([150, 160, 178]) * shade[f]
    # the skeleton, over the mesh
    if len(heads):
        hs = px(heads)
        for j, p in enumerate(jparent):
            if p < 0:
                continue
            for t in np.linspace(0, 1, 40):
                q = hs[p] * (1 - t) + hs[j] * t
                xi, yi = int(q[0]), int(q[1])
                if 0 <= xi < size and 0 <= yi < size:
                    img[max(0, yi - 1):yi + 1, max(0, xi - 1):xi + 1] = [245, 140, 20]
    return img.clip(0, 255).astype(np.uint8)


def png(path, img):
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def main():
    a = sys.argv[1:]
    src, dst = a[0], a[1]
    which = [int(x) for x in (a[a.index("--frames") + 1] if "--frames" in a else "0,15,30,45").split(",")]
    size = int(a[a.index("--size") + 1]) if "--size" in a else 320
    doc, bin_ = load(src)
    faces, ps, js, jparent = posed_frames(doc, bin_, which)
    allp = np.concatenate(ps)
    box = (allp.min(axis=0), allp.max(axis=0))
    rows = [np.concatenate([raster(p, faces, j, jparent, v, box, size) for p, j in zip(ps, js)], axis=1) for v in ("side", "front")]
    img = np.concatenate(rows, axis=0)
    png(dst, img)
    print(f"{dst}: {len(ps)} frames x side, front; {len(faces)} triangles; clip box {np.round(box[1] - box[0], 3)}")


main()
