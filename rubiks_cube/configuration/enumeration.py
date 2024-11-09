from enum import Enum
from enum import unique


@unique
class Face(Enum):
    up = "up"
    front = "front"
    right = "right"
    back = "back"
    left = "left"
    down = "down"
    empty = "empty"


@unique
class Piece(Enum):
    center = "center"
    corner = "corner"
    edge = "edge"


@unique
class Metric(Enum):
    ETM = "Execution Turn Metric"
    HTM = "Half Turn Metric"
    STM = "Slice Turn Metric"
    QTM = "Quarter Turn Metric"


@unique
class Status(Enum):
    Success = "success"
    Failure = "failure"


@unique
class Tag(Enum):
    layer = "layer"
    line = "line"
    block_1x1x3 = "1x1x3"
    block_1x2x2 = "1x2x2"
    block_1x2x3 = "1x2x3"
    block_2x2x2 = "2x2x2"
    block_2x2x3 = "2x2x3"
    two_blocks = "two_blocks"
    centers = "centers"
    corners = "corners"
    cross = "cross"
    edges = "edges"
    co = "co"
    co_fb = "co-fb"
    co_lr = "co-lr"
    co_ud = "co-ud"
    co_face = "co_face"
    co_htr = "co_htr"
    cp_layer = "cp_layer"
    dr = "dr"
    dr_fb = "dr-fb"
    dr_lr = "dr-lr"
    dr_ud = "dr-ud"
    eo = "eo"
    eo_fb = "eo-fb"
    eo_lr = "eo-lr"
    eo_ud = "eo-ud"
    eo_face = "eo_face"
    eo_fb_lr = "eo-fb+lr"
    eo_fb_ud = "eo-fb+ud"
    eo_lr_ud = "eo-lr+ud"
    eo_line = "eo_line"
    eo_floppy = "eo_floppy"
    eo_floppy_fb = "eo_floppy-fb"
    eo_floppy_lr = "eo_floppy-lr"
    eo_floppy_ud = "eo_floppy-ud"
    eo_htr = "eo-htr"
    ep_layer = "ep_layer"
    face = "face"
    f2l = "f2l"
    f2l_co = "f2l+co"
    f2l_cp = "f2l+cp"
    f2l_eo = "f2l+eo"
    f2l_ep = "f2l+ep"
    f2l_ep_co = "f2l+ep+co"
    f2l_eo_cp = "f2l+eo+cp"
    f2l_face = "f2l+face"
    floppy = "floppy"
    floppy_fb = "floppy-fb"
    floppy_lr = "floppy-lr"
    floppy_ud = "floppy-ud"
    floppy_col = "floppy-columns"
    floppy_fb_col = "floppy-fb-columns"
    floppy_lr_col = "floppy-lr-columns"
    floppy_ud_col = "floppy-ud-columns"
    htr = "htr"
    htr_like = "htr_like"
    minus_slice = "minus_slice"
    minus_slice_e = "minus_slice-e"
    minus_slice_m = "minus_slice-m"
    minus_slice_s = "minus_slice-s"
    leave_slice = "leave_slice"
    leave_slice_e = "leave_slice-e"
    leave_slice_m = "leave_slice-m"
    leave_slice_s = "leave_slice-s"
    solved = "solved"
    xo_fb = "xo_fb"
    xo_lr = "xo_lr"
    xo_ud = "xo_ud"
    xo_all = "xo_all"
    xp_face = "xp_face"
    x_cross = "x_cross"
    xx_cross = "xx_cross"
    xx_cross_adjacent = "xx_cross_adjacent"
    xx_cross_diagonal = "xx_cross_diagonal"
    xxx_cross = "xxx_cross"


@unique
class Symmetry(Enum):
    # face = color
    left = "left"
    right = "right"
    front = "front"
    back = "back"
    up = "up"
    down = "down"
    # Corners
    ubl = "ubl"
    ubr = "ubr"
    ufl = "ufl"
    ufr = "ufr"
    dbl = "dbl"
    dbr = "dbr"
    dfl = "dfl"
    dfr = "dfr"
    # Edges
    ul = "ul"
    ur = "ur"
    uf = "uf"
    ub = "ub"
    dl = "dl"
    dr = "dr"
    df = "df"
    db = "db"
    fl = "fl"
    fr = "fr"
    bl = "bl"
    br = "br"
    # Axis 1
    lr = "lr"
    fb = "fb"
    ud = "ud"
    # Axis 2
    m = "m"
    s = "s"
    e = "e"
    # Face + corner
    up_bl = "up-bl"
    up_br = "up-br"
    up_fl = "up-fl"
    up_fr = "up-fr"
    down_bl = "down-bl"
    down_br = "down-br"
    down_fl = "down-fl"
    down_fr = "down-fr"
    left_ub = "left-ub"
    left_db = "left-db"
    left_uf = "left-uf"
    left_df = "left-df"
    right_ub = "right-ub"
    right_db = "right-db"
    right_uf = "right-uf"
    right_df = "right-df"
    front_ul = "front-ul"
    front_ur = "front-ur"
    front_dl = "front-dl"
    front_dr = "front-dr"
    back_ul = "back-ul"
    back_ur = "back-ur"
    back_dl = "back-dl"
    back_dr = "back-dr"
    # Face + edge
    up_l = "up-l"
    up_r = "up-r"
    up_f = "up-f"
    up_b = "up-b"
    down_l = "down-l"
    down_r = "down-r"
    down_f = "down-f"
    down_b = "down-b"
    left_u = "left-u"
    left_d = "left-d"
    left_f = "left-f"
    left_b = "left-b"
    right_u = "right-u"
    right_d = "right-d"
    right_f = "right-f"
    right_b = "right-b"
    front_u = "front-u"
    front_d = "front-d"
    front_l = "front-l"
    front_r = "front-r"
    back_u = "back-u"
    back_d = "back-d"
    back_l = "back-l"
    back_r = "back-r"


@unique
class Step(Enum):
    inspection = "inspection"
    rotation = "rotation"
    skip = "skip"
    cancelation = "cancelation"
    finish = "finish"
    auf = "auf"
    coll = "coll"
    eoll = "eoll"
    cpll = "cpll"
    epll = "epll"
    zbll = "zbll"
    oll = "oll"
    pll = "pll"
    drm = "drm"


@unique
class Progress(Enum):
    draft = "draft"
    skeleton = "skeleton"
    insertion = "insertion"
    rewrite = "rewrite"
    blocks = "blocks"
