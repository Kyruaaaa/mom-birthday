"""
Builds the four-flavour birthday cake (pizza / oatmeal / waffle / basque) as a
low-poly GLB with hand-generated textures, then renders preview images.

Run:
  /Applications/Blender.app/Contents/MacOS/Blender -b --python model/build_cake.py

Outputs:
  assets/cake.glb          the model (one object per slice, toppings parented to their slice)
  model/textures/*.png     the generated textures
  model/renders/*.png      preview renders
"""
import bpy, math, os, random
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, 'textures')
REN = os.path.join(HERE, 'renders')
OUT = os.path.normpath(os.path.join(HERE, '..', 'assets', 'cake.glb'))
os.makedirs(TEX, exist_ok=True)
os.makedirs(REN, exist_ok=True)

rng = np.random.default_rng(11)
random.seed(11)

R, H = 1.0, 0.6          # cake radius / height (Blender units)
GAP = 0.014              # visible cut between slices
S = 1.02 * R             # half-size of the top texture in world units

bpy.ops.wm.read_factory_settings(use_empty=True)


# ───────────────────────── colour + noise helpers ─────────────────────────
def C(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])

def lerp(a, b, t):
    t = np.asarray(t)
    return a + (b - a) * (t[..., None] if t.ndim else t)

def sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)

def vnoise(h, w, cy, cx, seed, wrap=False):
    g = np.random.default_rng(seed).random((cy + 1, cx + 1))
    if wrap:
        g[:, cx] = g[:, 0]
    y = np.linspace(0, cy, h, endpoint=False)
    x = np.linspace(0, cx, w, endpoint=False)
    yi, xi = np.floor(y).astype(int), np.floor(x).astype(int)
    fy, fx = y - yi, x - xi
    fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
    g00, g01 = g[np.ix_(yi, xi)], g[np.ix_(yi, xi + 1)]
    g10, g11 = g[np.ix_(yi + 1, xi)], g[np.ix_(yi + 1, xi + 1)]
    a = g00 + (g01 - g00) * fx
    b = g10 + (g11 - g10) * fx
    return a + (b - a) * fy[:, None]

def fbm(h, w, cy, cx, octs, seed, wrap=False):
    t, amp, tot = np.zeros((h, w)), 1.0, 0.0
    for o in range(octs):
        t += amp * vnoise(h, w, cy * 2 ** o, cx * 2 ** o, seed + o, wrap)
        tot += amp
        amp *= 0.5
    return t / tot

def ramp(t, stops):
    xs = [s[0] for s in stops]
    cols = np.array([C(s[1]) for s in stops])
    return np.stack([np.interp(t, xs, cols[:, i]) for i in range(3)], axis=-1)

def box_blur(arr, r, wrap_x):
    out = arr
    for _ in range(2):                                   # two passes ≈ soft gaussian
        acc = np.zeros_like(out)
        for d in range(-r, r + 1):
            acc += np.roll(out, d, axis=1) if wrap_x else np.pad(out, ((0, 0), (r, r), (0, 0)), mode='edge')[:, r + d:r + d + out.shape[1]]
        out = acc / (2 * r + 1)
        acc = np.zeros_like(out)
        for d in range(-r, r + 1):
            acc += np.pad(out, ((r, r), (0, 0), (0, 0)), mode='edge')[r + d:r + d + out.shape[0]]
        out = acc / (2 * r + 1)
    return out

def watercolor(arr, wrap_x=False, blur=2, seed=0):
    """Soft painted look: blurred edges, pastel tint, gentle pigment blooms and paper grain."""
    h, w, _ = arr.shape
    out = box_blur(arr, blur, wrap_x)
    gray = out.mean(axis=2, keepdims=True)
    out = out + (gray - out) * 0.22                      # a little less saturated
    out = lerp(out, C('#fff3e6'), 0.16)                  # warm milky wash, lifts the darks
    bloom = fbm(h, w, 3, 5, 4, seed + 500, wrap_x) - 0.5
    grain = fbm(h, w, h // 6, w // 6, 2, seed + 600, wrap_x) - 0.5
    out = out * (1 + bloom * 0.12 + grain * 0.07)[..., None]
    return out

def save_tex(name, arr, soft=True, wrap_x=False):
    if soft:
        arr = watercolor(arr, wrap_x, seed=len(name))
    h, w, _ = arr.shape
    img = bpy.data.images.new(name + '_gen', w, h, alpha=False)
    rgba = np.concatenate([np.clip(arr, 0, 1), np.ones((h, w, 1))], axis=2).astype(np.float32)
    img.pixels.foreach_set(rgba.ravel())
    path = os.path.join(TEX, name + '.png')
    img.filepath_raw = path
    img.file_format = 'PNG'
    img.save()
    bpy.data.images.remove(img)
    out = bpy.data.images.load(path)
    out.name = name
    return out


# ───────────────────────── top texture (whole cake, planar) ─────────────────────────
N = 1024
lin = (np.arange(N) + 0.5) / N * 2 * S - S
WX, WY = np.meshgrid(lin, lin)                 # row 0 = bottom (-y), matches Blender pixel order
RAD = np.hypot(WX, WY)
ANG = np.degrees(np.arctan2(WY, WX)) % 360

def w2p(x, y):
    return int((x + S) / (2 * S) * N), int((y + S) / (2 * S) * N)

def pol(deg, r):
    a = math.radians(deg)
    return r * math.cos(a), r * math.sin(a)

def stamp(img, region, cx, cy, rx, ry, rot, col, soft=0.25, edge=0.0, edgecol=None, wob=0.0, hl=0.0):
    rmax = max(rx, ry) * (1 + wob) + 0.01
    x0, y0 = w2p(cx - rmax, cy - rmax)
    x1, y1 = w2p(cx + rmax, cy + rmax)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1 + 1, N), min(y1 + 1, N)
    if x1 <= x0 or y1 <= y0:
        return
    dx, dy = WX[y0:y1, x0:x1] - cx, WY[y0:y1, x0:x1] - cy
    c, s = math.cos(rot), math.sin(rot)
    u, v = (dx * c + dy * s) / rx, (-dx * s + dy * c) / ry
    d = np.sqrt(u * u + v * v)
    if wob:
        th = np.arctan2(v, u)
        k = random.random() * 6
        d = d * (1 + wob * (np.sin(3 * th + k) * 0.6 + np.sin(5 * th + 2 * k) * 0.4))
    a = np.clip((1 - d) / soft, 0, 1)
    if region is not None:
        a = a * region[y0:y1, x0:x1]
    cw = np.broadcast_to(col, (y1 - y0, x1 - x0, 3)).copy()
    if edge:
        cw = lerp(cw, edgecol, sstep(0.5, 1.0, d) * edge)
    if hl:
        cw = lerp(cw, np.ones(3), np.clip(1 - d * 1.7, 0, 1) * hl)
    img[y0:y1, x0:x1] = lerp(img[y0:y1, x0:x1], cw, a)

def build_top():
    n1 = fbm(N, N, 6, 6, 5, 1)
    n2 = fbm(N, N, 40, 40, 3, 2)
    top = np.zeros((N, N, 3)) + C('#f1e8d6')
    mP = ((ANG >= 90) & (ANG < 180)).astype(float)
    mO = (ANG < 90).astype(float)
    mW = ((ANG >= 180) & (ANG < 270)).astype(float)
    mB = (ANG >= 270).astype(float)

    # pizza: sauce, golden crust rim, mozzarella, tomato, basil
    piz = lerp(C('#c4351f'), C('#ee6d3e'), n1 * 0.7 + n2 * 0.3)
    piz = lerp(piz, lerp(C('#dca45a'), C('#f3d79d'), n2), sstep(0.86, 0.94, RAD))
    top = lerp(top, piz, mP)
    for _ in range(13):
        x, y = pol(random.uniform(98, 172), random.uniform(0.15, 0.8))
        rr = random.uniform(0.07, 0.12)
        stamp(top, mP, x, y, rr, rr * random.uniform(0.75, 1), random.uniform(0, 6), C('#fbf1dc'),
              soft=0.18, edge=0.6, edgecol=C('#efcf98'), wob=0.18, hl=0.25)
    for _ in range(9):
        x, y = pol(random.uniform(96, 174), random.uniform(0.15, 0.85))
        rr = random.uniform(0.035, 0.055)
        stamp(top, mP, x, y, rr, rr, 0, C('#d4232f'), soft=0.15, edge=0.7, edgecol=C('#93121c'), hl=0.18)
    for _ in range(220):
        x, y = pol(random.uniform(92, 178), random.uniform(0.08, 0.9))
        stamp(top, mP, x, y, random.uniform(0.01, 0.022), random.uniform(0.005, 0.01), random.uniform(0, 6),
              C(random.choice(['#3f7f33', '#5c9d43', '#78b257'])), soft=0.5)

    # oatmeal: cream base covered in oat flakes
    oat = lerp(C('#f4ecdd'), C('#e8dbc2'), n1)
    top = lerp(top, oat, mO)
    for _ in range(900):
        x, y = pol(random.uniform(1, 89), 0.95 * math.sqrt(random.uniform(0.01, 1)))
        rx = random.uniform(0.026, 0.04)
        stamp(top, mO, x, y, rx, rx * random.uniform(0.55, 0.7), random.uniform(0, 6),
              lerp(C('#efdcb2'), C('#d4b680'), random.random()), soft=0.3, edge=0.55, edgecol=C('#bf9c62'), hl=0.25)

    # waffle: diamond grid of pockets
    p = (WX + WY) / math.sqrt(2)
    q = (WX - WY) / math.sqrt(2)
    cs = 0.115
    wall = np.maximum(np.abs((p / cs) % 1 - 0.5), np.abs((q / cs) % 1 - 0.5)) * 2
    pocket = lerp(C('#c07f36'), C('#d9a052'), np.clip(wall / 0.7, 0, 1))
    waf = lerp(pocket, C('#f2cd85'), sstep(0.68, 0.8, wall))
    waf = waf * (0.93 + 0.14 * n2[..., None])
    waf = lerp(waf, C('#b8772f'), sstep(0.9, 0.99, RAD) * 0.45)
    top = lerp(top, waf, mW)

    # basque: pale custard centre burning to dark caramel at the rim
    t = np.clip((RAD - 0.3) / 0.65, 0, 1) ** 1.5 + (n1 - 0.5) * 0.35 + (n2 - 0.5) * 0.15
    bas = ramp(np.clip(t, 0, 1), [(0, '#f8e8b0'), (0.35, '#f1cd80'), (0.6, '#cd8a4a'), (0.8, '#9a5228'), (1, '#6e3519')])
    top = lerp(top, bas, mB)
    return save_tex('cake_top', top)


# ───────────────────────── side textures (layered cross-sections) ─────────────────────────
SW, SH = 1024, 616      # 1 world unit wide x cake height (0.6) tall
U, VV = np.meshgrid(np.linspace(0, 1, SW, endpoint=False), np.linspace(0, 1, SH, endpoint=False))

def band_sponge(seed):
    n = fbm(SH, SW, 8, 14, 4, seed, True)
    pores = fbm(SH, SW, 40, 70, 2, seed + 9, True)
    col = lerp(C('#efd28a'), C('#f8e8b6'), n)
    return col * (1 - 0.13 * sstep(0.62, 0.75, pores))[..., None]

def band_cream(seed):
    return lerp(C('#f3eee6'), C('#fffdf9'), fbm(SH, SW, 6, 10, 3, seed, True))

def band_flat(a, b, seed, cy=8, cx=14):
    return lerp(C(a), C(b), fbm(SH, SW, cy, cx, 4, seed, True))

def band_strawberry_cream(lo, hi, seed):
    col = band_cream(seed)
    mid, ry = (lo + hi) / 2, (hi - lo) * 0.45
    for k in range(3):
        uc = (k + 0.5) / 3
        du = ((U - uc + 0.5) % 1) - 0.5
        d = np.sqrt((du / 0.135) ** 2 + ((VV - mid) / ry) ** 2)
        sl = ramp(np.clip(d, 0, 1), [(0, '#fde5de'), (0.45, '#f6a596'), (0.85, '#ec6c5c'), (1, '#de4a3d')])
        col = lerp(col, sl, np.clip((1 - d) / 0.06, 0, 1))
    return col

def band_waffle(seed):
    col = band_flat('#d29a4a', '#ebc27a', seed)
    lines = np.abs((U / 0.055) % 1 - 0.5) < 0.09
    col[lines] *= 0.93
    return col

def band_oatcream(seed):
    col = band_cream(seed)
    flecks = fbm(SH, SW, 30, 60, 2, seed + 4, True)
    return lerp(col, C('#dcc08c'), sstep(0.66, 0.72, flecks))

def side_tex(name, bands, seed):
    img = np.zeros((SH, SW, 3))
    wob = (fbm(1, SW, 1, 10, 3, seed, True)[0] - 0.5) * 0.035
    Vw = VV + wob[None, :]
    lo = 0.0
    for hi, fn in bands:
        m = (Vw >= lo) & ((Vw < hi) if hi < 1 else True)
        col = fn(lo, hi, seed + int(hi * 100))
        img[m] = col[m]
        lo = hi
    img *= (0.9 + 0.1 * sstep(0.0, 0.12, VV))[..., None]       # a touch darker at the very bottom
    return save_tex(name, img, wrap_x=True)

sponge = lambda lo, hi, s: band_sponge(s)
cream = lambda lo, hi, s: band_cream(s)
flat = lambda a, b: (lambda lo, hi, s: band_flat(a, b, s))


# ───────────────────────── materials ─────────────────────────
def lin_color(hexstr):
    c = C(hexstr)
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return (*c, 1.0)

def make_mat(name, color='#ffffff', img=None, rough=0.88, spec=0.1):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Roughness'].default_value = rough
    try:
        b.inputs['Specular IOR Level'].default_value = spec
    except KeyError:
        pass
    if img:
        tex = m.node_tree.nodes.new('ShaderNodeTexImage')
        tex.image = img
        m.node_tree.links.new(tex.outputs['Color'], b.inputs['Base Color'])
    else:
        b.inputs['Base Color'].default_value = lin_color(color)
    return m

print('building textures…')
M_TOP = make_mat('CakeTop', img=build_top(), rough=0.7)
M_SIDE = {
    'pizza': make_mat('PizzaSide', img=side_tex('side_pizza', [
        (0.08, flat('#c08a46', '#d6a25a')), (0.8, flat('#ecd29a', '#f8e8c2')),
        (0.88, flat('#f6e6c4', '#fdf4df')), (1.0, flat('#c93a24', '#e9633a'))], 21)),
    'oat': make_mat('OatSide', img=side_tex('side_oat', [
        (0.07, flat('#e3c27f', '#efd395')), (0.36, sponge), (0.45, cream), (0.70, sponge),
        (0.78, cream), (1.0, lambda lo, hi, s: band_oatcream(s))], 31)),
    'waffle': make_mat('WaffleSide', img=side_tex('side_waffle', [
        (0.12, lambda lo, hi, s: band_waffle(s)), (0.44, band_strawberry_cream),
        (0.54, lambda lo, hi, s: band_waffle(s)), (0.82, cream), (1.0, lambda lo, hi, s: band_waffle(s))], 41)),
    'basque_cut': make_mat('BasqueCut', img=side_tex('side_basque_cut', [
        (0.05, flat('#4e2512', '#7a3b1b')), (0.1, flat('#c98446', '#dda061')),
        (0.88, flat('#f6dc96', '#fbeec4')), (0.94, flat('#c98446', '#dda061')), (1.0, flat('#4e2512', '#7a3b1b'))], 51)),
    # golden-amber baked outside, darkening to a caramelised crust at the top edge
    'basque_out': make_mat('BasqueOuter', img=side_tex('side_basque_out', [
        (1.0, lambda lo, hi, s: ramp(np.clip(fbm(SH, SW, 4, 8, 4, s, True) * 0.55 + sstep(0.6, 1.0, VV) * 0.55, 0, 1),
                                     [(0, '#e7bb78'), (0.35, '#d69a55'), (0.65, '#b06a33'), (1, '#6a3318')]))], 61), rough=0.6),
}

def straw_tex():
    n = 256
    u, v = np.meshgrid(np.linspace(0, 1, n, endpoint=False), np.linspace(0, 1, n, endpoint=False))
    col = lerp(C('#c41d32'), C('#e9444f'), fbm(n, n, 4, 8, 3, 70, True))
    col = lerp(col, C('#f4a39a'), sstep(0.82, 0.97, v) * 0.6)
    fu = (u * 12 + 0.5 * (np.floor(v * 9) % 2)) % 1 - 0.5
    fv = (v * 9) % 1 - 0.5
    d = np.sqrt((fu / 0.16) ** 2 + (fv / 0.24) ** 2)
    seeds = (d < 1) & (v > 0.06) & (v < 0.86)
    col[seeds] = lerp(C('#f6dc78'), C('#b88a2a'), np.clip(d[seeds], 0, 1))
    return save_tex('strawberry', col)

def candle_tex():
    n = 256
    u, v = np.meshgrid(np.linspace(0, 1, n, endpoint=False), np.linspace(0, 1, n, endpoint=False))
    stripe = ((u * 4 + v * 2.5) % 1) < 0.5
    col = np.zeros((n, n, 3)) + C('#fffaf2')
    col[stripe] = C('#b9a2e6')
    return save_tex('candle', col)

# pastel, matte colours so everything reads like the soft painted reference
M_STRAW = make_mat('Strawberry', img=straw_tex(), rough=0.6, spec=0.2)
M_LEAF = make_mat('Leaf', '#6fa765', rough=0.8)
M_CREAM = make_mat('Cream', '#fdfaf4', rough=0.95, spec=0.05)
M_BLUE = make_mat('Blueberry', '#5466a3', rough=0.7, spec=0.15)
M_BLUE_TOP = make_mat('BlueberryCrown', '#36406e', rough=0.9)
M_MOZZ = make_mat('Mozzarella', '#fdf4e3', rough=0.9, spec=0.05)
M_BASIL = make_mat('Basil', '#6a9e57', rough=0.8)
M_OAT1 = make_mat('Oat', '#efdcb4', rough=0.9)
M_OAT2 = make_mat('OatDark', '#e2c899', rough=0.9)
M_NUT = make_mat('Hazelnut', '#c08250', rough=0.7, spec=0.15)
PLATE_R = 1.56           # outer radius of the lace edge (scallop tips)
PLATE_N = 32             # number of lace scallops

def plate_tex():
    """White porcelain with a painted lavender garland and a lace border, mapped flat from above."""
    n = 1024
    lin_ = (np.arange(n) + 0.5) / n * 2 * PLATE_R - PLATE_R
    x, y = np.meshgrid(lin_, lin_)
    r = np.hypot(x, y)
    a = np.arctan2(y, x)
    col = lerp(C('#fdfbff'), C('#f1ecfa'), fbm(n, n, 6, 6, 4, 81) * 0.6 + sstep(0.9, 1.25, r) * 0.25)
    lav, deep, sage = C('#b9a2e6'), C('#8d72cc'), C('#9db893')
    def ring(r0, w, c, k=1.0):
        nonlocal col
        col = lerp(col, c, np.clip(1 - np.abs(r - r0) / w, 0, 1) ** 0.7 * k)
    ring(1.13, 0.008, deep, 0.8)                         # fine line around the well
    ring(1.165, 0.004, lav, 0.6)
    # garland: one little five-petal flower per scallop, leaves and dots between them
    seg = 2 * np.pi / PLATE_N
    t = (a / seg) % 1 - 0.5                              # -0.5..0.5 within one scallop
    du = t * seg * 1.36                                  # arc distance on the rim (approx.)
    for cx_, cy_, rad, c, k in [(0, 1.33, 0.05, lav, 0.9), (0.5, 1.31, 0.018, deep, 0.7)]:
        dx = (t - cx_ + 0.5) % 1 - 0.5
        d = np.hypot(dx * seg * 1.33, r - cy_)
        if rad > 0.03:
            ang = np.arctan2(r - cy_, dx * seg * 1.33)
            petal = rad * (0.62 + 0.38 * np.abs(np.cos(2.5 * ang)))
            col = lerp(col, c, sstep(petal, petal * 0.75, d) * k)
            col = lerp(col, C('#f6e7a6'), sstep(0.016, 0.01, d))
        else:
            col = lerp(col, c, sstep(rad, rad * 0.6, d) * k)
    for side in (-1, 1):                                 # leaves either side of each flower
        dx = t - side * 0.24
        d = np.hypot(dx * seg * 1.33 / 0.045, (r - 1.335) / 0.016)
        col = lerp(col, sage, sstep(1.0, 0.7, d) * 0.75)
    # lace border: scalloped outline following the edge with eyelets
    edge = 1.45 + 0.09 * np.abs(np.sin(np.pi * ((a / seg) % 1))) ** 0.55
    ring_d = np.abs(r - (edge - 0.045))
    col = lerp(col, deep, sstep(0.008, 0.003, ring_d) * 0.85)
    eye = np.hypot(t * seg * 1.5, r - 1.465)
    col = lerp(col, lav, sstep(0.024, 0.016, eye) * 0.9)
    col = lerp(col, C('#fdfbff'), sstep(0.011, 0.006, eye))
    for k in (-1, 1):
        e2 = np.hypot((t - k * 0.3) * seg * 1.5, r - 1.49)
        col = lerp(col, lav, sstep(0.01, 0.005, e2) * 0.8)
    col = lerp(col, lav, sstep(0.01, 0.0, edge - r) * 0.55)    # tinted rim edge
    return save_tex('plate', col)

M_PLATE = make_mat('Plate', img=plate_tex(), rough=0.22, spec=0.5)
M_CANDLE = make_mat('Candle', img=candle_tex(), rough=0.7)
M_WICK = make_mat('Wick', '#4a3a30', rough=0.9)


# ───────────────────────── geometry ─────────────────────────
def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj

def wedge(name, a0d, a1d, side_mat, cut_mat, seg):
    """One closed slice; UVs are per face type, then the edges get a soft rounded bevel."""
    a0, a1 = math.radians(a0d), math.radians(a1d)
    angs = [a0 + (a1 - a0) * i / seg for i in range(seg + 1)]
    P = lambda a, z: (R * math.cos(a), R * math.sin(a), z)
    verts = [(0, 0, H), (0, 0, 0)] + [P(a, H) for a in angs] + [P(a, 0) for a in angs]
    Ct, Cb = 0, 1
    t = list(range(2, 2 + len(angs)))
    b = list(range(2 + len(angs), 2 + 2 * len(angs)))
    ang_of = {**{t[i]: a for i, a in enumerate(angs)}, **{b[i]: a for i, a in enumerate(angs)}}

    faces, kinds = [], []
    faces.append([Ct] + t); kinds.append('top')
    faces.append(([Cb] + b)[::-1]); kinds.append('bottom')
    for i in range(seg):
        faces.append([b[i], b[i + 1], t[i + 1], t[i]]); kinds.append('side')
    faces.append([Cb, b[0], t[0], Ct]); kinds.append('cut')
    faces.append([Cb, Ct, t[-1], b[-1]]); kinds.append('cut')

    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    uvl = me.uv_layers.new(name='UVMap')
    for poly in me.polygons:
        kind = kinds[poly.index]
        poly.material_index = {'top': 0, 'bottom': 1, 'side': 1, 'cut': 2}[kind]
        for li in poly.loop_indices:
            vi = me.loops[li].vertex_index
            x, y, z = verts[vi]
            if kind in ('top', 'bottom'):
                uv = ((x + S) / (2 * S), (y + S) / (2 * S))
            elif kind == 'side':
                uv = (ang_of[vi] * R, z / H)
            else:
                uv = (math.hypot(x, y), z / H)
            uvl.data[li].uv = uv
    for m in (M_TOP, side_mat, cut_mat):
        me.materials.append(m)
    me.update()
    o = link(bpy.data.objects.new(name, me))

    bpy.ops.object.select_all(action='DESELECT')
    o.select_set(True)
    bpy.context.view_layer.objects.active = o
    bev = o.modifiers.new('soften', 'BEVEL')
    bev.width = 0.055
    bev.segments = 4
    bev.limit_method = 'ANGLE'
    bev.angle_limit = math.radians(30)
    bpy.ops.object.modifier_apply(modifier=bev.name)
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(50))
    return o

def active():
    return bpy.context.view_layer.objects.active

def finish(o, mat, smooth=True):
    o.data.materials.append(mat)
    for p in o.data.polygons:
        p.use_smooth = smooth
    return o

def jitter(o, amt):
    for v in o.data.vertices:
        v.co += Vector(rng.uniform(-amt, amt, 3))

def join(objs, name):
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    o = active()
    o.name = o.data.name = name
    return o

def ico(sub, r, loc, scale=(1, 1, 1), rot=(0, 0, 0), mat=None, jit=0.0, smooth=True):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=sub, radius=r, location=loc, rotation=rot)
    o = active()
    o.scale = scale
    if jit:
        jitter(o, jit * r)
    return finish(o, mat, smooth)

def on_top(deg, r, z=0.0):
    x, y = pol(deg, r)
    return (x, y, H + z)

def strawberry(loc, size, rot):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=12, radius=1)
    o = active()
    for v in o.data.vertices:
        t = (v.co.z + 1) / 2
        s = 0.2 + 0.8 * math.sin(min(t * 1.25, 1) * math.pi / 2)
        v.co.x *= s; v.co.y *= s; v.co.z *= 1.2
    finish(o, M_STRAW)
    bpy.ops.mesh.primitive_cone_add(vertices=10, radius1=0.8, radius2=0.12, depth=0.16, location=(0, 0, 1.18))
    c = active()
    for v in c.data.vertices:
        if v.co.z < 0:
            k = round(math.atan2(v.co.y, v.co.x) / (2 * math.pi / 10))
            if k % 2:
                v.co.x *= 0.45; v.co.y *= 0.45
    finish(c, M_LEAF, smooth=False)
    o = join([o, c], 'strawberry')
    o.scale = (size,) * 3
    o.rotation_euler = rot
    o.location = loc
    return o

def blueberry(loc, r=0.042):
    b = ico(2, r, loc, scale=(1, 1, 0.9), mat=M_BLUE, jit=0.04)
    bpy.ops.mesh.primitive_cone_add(vertices=5, radius1=r * 0.35, radius2=r * 0.15, depth=r * 0.15,
                                    location=(loc[0], loc[1], loc[2] + r * 0.88))
    return [b, finish(active(), M_BLUE_TOP, smooth=False)]

def dollop(loc, r=0.085):
    a = ico(3, r, (loc[0], loc[1], loc[2] + r * 0.45), scale=(1, 1, 0.72), mat=M_CREAM, jit=0.03)
    b = ico(3, r * 0.62, (loc[0], loc[1], loc[2] + r * 1.05), scale=(1, 1, 0.8), mat=M_CREAM, jit=0.03)
    return [a, b]


print('building slices…')
cake_root = link(bpy.data.objects.new('Cake', None))

SLICES = [
    # name,           from, to,  side,                cut,                 segments
    ('Slice_Oat_A',     0,  45, M_SIDE['oat'],        M_SIDE['oat'],        6),
    ('Slice_Oat_B',    45,  90, M_SIDE['oat'],        M_SIDE['oat'],        6),
    ('Slice_Pizza',    90, 180, M_SIDE['pizza'],      M_SIDE['pizza'],     12),
    ('Slice_Waffle_A', 180, 225, M_SIDE['waffle'],    M_SIDE['waffle'],     6),
    ('Slice_Waffle_B', 225, 270, M_SIDE['waffle'],    M_SIDE['waffle'],     6),
    ('Slice_Basque',  270, 360, M_SIDE['basque_out'], M_SIDE['basque_cut'], 12),
]

toppings = {name: [] for name, *_ in SLICES}

# pizza: mozzarella blobs + basil leaves
for deg, r in [(104, 0.46), (122, 0.72), (147, 0.42), (165, 0.7), (128, 0.22), (152, 0.85), (100, 0.82)]:
    rr = random.uniform(0.075, 0.105)
    toppings['Slice_Pizza'].append(ico(3, rr, on_top(deg, r, rr * 0.12), scale=(1, random.uniform(0.8, 1), 0.32),
                                       rot=(0, 0, random.uniform(0, 6)), mat=M_MOZZ, jit=0.12))
for _ in range(16):
    toppings['Slice_Pizza'].append(ico(1, 1, on_top(random.uniform(95, 175), random.uniform(0.15, 0.9), 0.03),
                                       scale=(0.042, 0.018, 0.004),
                                       rot=(random.uniform(-.3, .3), random.uniform(-.3, .3), random.uniform(0, 6)), mat=M_BASIL))

# oatmeal: loose 3D oat flakes + two hazelnuts on cream
for name, lo, hi in (('Slice_Oat_A', 4, 41), ('Slice_Oat_B', 49, 86)):
    for _ in range(26):
        toppings[name].append(ico(1, 1, on_top(random.uniform(lo, hi), 0.93 * math.sqrt(random.uniform(0.05, 1)), 0.006),
                                  scale=(0.03, 0.02, 0.006),
                                  rot=(random.uniform(-.35, .35), random.uniform(-.35, .35), random.uniform(0, 6)),
                                  mat=random.choice([M_OAT1, M_OAT2])))
for deg, r in [(14, 0.68), (31, 0.45)]:
    x, y, z = on_top(deg, r)
    toppings['Slice_Oat_A'] += dollop((x, y, z), 0.06)
    nut = ico(2, 0.05, (x, y, z + 0.1), scale=(1, 1, 0.9), mat=M_NUT, jit=0.05)
    top_v = max(nut.data.vertices, key=lambda v: v.co.z)
    top_v.co.z *= 1.35
    toppings['Slice_Oat_A'].append(nut)

# waffle A: cream swirls + strawberries
for deg, r in [(192, 0.78), (204, 0.42), (214, 0.72), (188, 0.5)]:
    toppings['Slice_Waffle_A'] += dollop(on_top(deg, r), random.uniform(0.07, 0.09))
toppings['Slice_Waffle_A'].append(strawberry(on_top(200, 0.62, 0.11), 0.075, (math.radians(70), 0, math.radians(30))))
toppings['Slice_Waffle_A'].append(strawberry(on_top(196, 0.86, 0.1), 0.065, (math.radians(-65), 0, math.radians(120))))
toppings['Slice_Waffle_A'].append(strawberry(on_top(210, 0.3, 0.08), 0.06, (math.radians(180), 0, 0)))

# waffle B: blueberries, a strawberry and a little cream
for deg, r in [(236, 0.78), (247, 0.6), (241, 0.42), (258, 0.84), (252, 0.34), (263, 0.62)]:
    toppings['Slice_Waffle_B'] += blueberry(on_top(deg, r, 0.035))
toppings['Slice_Waffle_B'] += dollop(on_top(232, 0.6), 0.065)
toppings['Slice_Waffle_B'].append(strawberry(on_top(232, 0.6, 0.15), 0.06, (math.radians(180), 0, 0)))

slice_objs = {}
for name, a0, a1, side_m, cut_m, seg in SLICES:
    w = wedge(name, a0, a1, side_m, cut_m, seg)
    if toppings[name]:
        t = join(toppings[name], name + '_Toppings')
        t.parent = w
    bis = math.radians((a0 + a1) / 2)
    w.location = (GAP * math.cos(bis), GAP * math.sin(bis), 0)
    w.parent = cake_root
    slice_objs[name] = w

# plate
# porcelain plate: flat well, rim that curves up into a fluted, scalloped lace edge
def plate_mesh():
    M = PLATE_N * 8
    radii = [0.0, 0.3, 0.6, 0.85, 1.05, 1.12, 1.17, 1.22, 1.28, 1.34, 1.4, 1.46, 1.5, 1.53, 1.0]   # last = edge
    verts, faces = [], []
    for j in range(M):
        th = 2 * math.pi * j / M
        sc = abs(math.sin(math.pi * ((th / (2 * math.pi / PLATE_N)) % 1))) ** 0.55
        edge = 1.45 + 0.09 * sc
        for k, rr in enumerate(radii):
            if k == len(radii) - 1:
                rr = edge
            else:
                rr = min(rr, edge - 0.012 * (len(radii) - 1 - k))
            lift = sstep(1.1, 1.3, rr) * 0.06 + max(0.0, rr - 1.3) * 0.12
            flute = sstep(1.3, 1.5, rr) * 0.012 * math.cos(PLATE_N * th)
            verts.append((rr * math.cos(th), rr * math.sin(th), float(lift + flute)))
    nr = len(radii)
    for j in range(M):
        j2 = (j + 1) % M
        for k in range(nr - 1):
            faces.append((j * nr + k, j * nr + k + 1, j2 * nr + k + 1, j2 * nr + k))
    me = bpy.data.meshes.new('Plate')
    me.from_pydata(verts, [], faces)
    o = link(bpy.data.objects.new('Plate', me))
    bpy.ops.object.select_all(action='DESELECT')
    o.select_set(True)
    bpy.context.view_layer.objects.active = o
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=1e-5)
    bpy.ops.object.mode_set(mode='OBJECT')
    uv = me.uv_layers.new(name='UVMap')
    for loop in me.loops:
        v = me.vertices[loop.vertex_index].co
        uv.data[loop.index].uv = (0.5 + v.x / (2 * PLATE_R), 0.5 + v.y / (2 * PLATE_R))
    sol = o.modifiers.new('Thick', 'SOLIDIFY')
    sol.thickness, sol.offset, sol.use_even_offset = 0.03, -1, True
    bev = o.modifiers.new('Soft', 'BEVEL')
    bev.width, bev.segments, bev.limit_method = 0.01, 2, 'ANGLE'
    o.location = (0, 0, 0.0)
    finish(o, M_PLATE)
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(60))
    return o

plate = plate_mesh()
plate.parent = cake_root

# candle, wick, and an empty marking where the flame goes
bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.045, depth=0.42, location=(0, 0, H + 0.21))
candle = finish(active(), M_CANDLE)
candle.name = 'Candle'
candle.parent = cake_root
bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=0.008, depth=0.06, location=(0, 0, H + 0.42 + 0.03))
wick = finish(active(), M_WICK, smooth=False)
wick.name = 'Wick'
wick.parent = cake_root
anchor = link(bpy.data.objects.new('FlameAnchor', None))
anchor.location = (0, 0, H + 0.42 + 0.055)
anchor.parent = cake_root


# ───────────────────────── export ─────────────────────────
print('exporting', OUT)
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.export_scene.gltf(filepath=OUT, export_format='GLB', export_apply=True,
                          export_image_format='JPEG', export_jpeg_quality=90)
# Also wrap it in a .js file, so the page can load the cake even when opened straight from disk.
import base64
with open(OUT, 'rb') as f:
    b64 = base64.b64encode(f.read()).decode()
with open(OUT.replace('.glb', '-glb.js'), 'w') as f:
    f.write('window.CAKE_GLB_BASE64 = "' + b64 + '";\n')


# ───────────────────────── preview renders ─────────────────────────
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 96
scene.cycles.use_denoising = True
try:
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'METAL'
    prefs.get_devices()
    for d in prefs.devices:
        d.use = True
    scene.cycles.device = 'GPU'
except Exception as e:
    print('GPU not available, using CPU', e)
scene.render.resolution_x, scene.render.resolution_y = 1100, 800
scene.view_settings.view_transform = 'Standard'

world = bpy.data.worlds.new('World')
scene.world = world
world.use_nodes = True
world.node_tree.nodes['Background'].inputs['Color'].default_value = lin_color('#efe8fa')
world.node_tree.nodes['Background'].inputs['Strength'].default_value = 1.15

bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, -0.061))
finish(active(), make_mat('Floor', '#e9e0f7', rough=0.9))

sun = bpy.data.lights.new('Sun', 'SUN')
sun.energy = 1.9
sun.angle = math.radians(35)          # big, soft shadows
sun_o = link(bpy.data.objects.new('Sun', sun))
sun_o.rotation_euler = (math.radians(40), math.radians(-18), math.radians(-35))

target = link(bpy.data.objects.new('Target', None))
target.location = (0, 0, 0.3)
cam = link(bpy.data.objects.new('Cam', bpy.data.cameras.new('Cam')))
cam.data.lens = 55
con = cam.constraints.new('TRACK_TO')
con.target = target
scene.camera = cam

def render(fname, loc):
    cam.location = loc
    scene.render.filepath = os.path.join(REN, fname)
    bpy.ops.render.render(write_still=True)

render('hero.png', (3.1, -3.4, 2.6))
render('top.png', (0, -0.02, 6.0))
w = slice_objs['Slice_Waffle_B']
bis = math.radians(247.5)
w.location = (0.5 * math.cos(bis), 0.5 * math.sin(bis), 0)
render('slice_out.png', (1.6, -3.9, 1.9))
print('done')
