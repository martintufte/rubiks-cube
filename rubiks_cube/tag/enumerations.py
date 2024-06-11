from enum import Enum


class Step(Enum):
    inspection = "inspection"
    rotation = "rotation"
    skip = "skip"
    cancelation = "cancelation"
    finish = "finish"
    auf = "auf"


class Progress(Enum):
    solved = "solved"
    draft = "draft"
    skeleton = "skeleton"
    insertion = "insertion"
    rewrite = "rewrite"
    trigger = "trigger"
    blocks = "blocks"
    drm = "drm"
    drm_4c4e = "drm-4c4e"
    drm_4c2e = "drm-4c2e"
    drm_3c2e = "drm-3c2e"


class Basic(Enum):
    face = "face"  # ok
    face_eo = "face+eo"  # ok
    face_co = "face+co"  # ok
    face_ep = "face+ep"  # ok
    face_cp = "face+cp"  # ok
    layer = "layer"  # ok


class CFOP(Enum):
    cross = "cross"  # ok
    x_cross = "x-cross"  # ok
    xx_cross_adjacent = "xx-cross-adjacent"  # ok
    xx_cross_diagonal = "xx-cross-diagonal"  # ok
    xx_cross = "xx-cross"  # ok
    xxx_cross = "xxx-cross"  # ok
    f2l = "f2l"  # ok
    f2l_eo = "f2l+eo"  # ok
    f2l_co = "f2l+co"  # ok
    f2l_ep = "f2l+ep"  # ok
    f2l_cp = "f2l+cp"  # ok
    f2l_miuns_1 = "f2l-1"  # ok (= xxx-cross)
    oll = "oll"  # ok
    coll = "coll"
    eoll = "eoll"
    pll = "pll"
    cpll = "cpll"
    epll = "epll"
    zbll = "zbll"
    vls = "vls"


class FewestMoves(Enum):
    solved_corners = "solved-corners"  # ok
    solved_edges = "solved-edges"  # ok
    co = "co"  # ok
    co_fb = "co-fb"  # ok
    co_lr = "co-lr"  # ok
    co_ud = "co-ud"  # ok
    eo = "eo"  # ok
    eo_fb = "eo-fb"  # ok
    eo_lr = "eo-lr"  # ok
    eo_ud = "eo-ud"  # ok
    dr = "dr"  # ok
    dr_fb = "dr-fb"  # ok
    dr_lr = "dr-lr"  # ok
    dr_ud = "dr-ud"  # ok
    htr = "htr"
    htr_fake = "fake-htr"
    floppy = "floppy"
    floppy_fb = "floppy-fb"
    floppy_lr = "floppy-lr"
    floppy_ud = "floppy-ud"
    minus_slice_m = "minus-slice-m"  # ok
    minus_slice_s = "minus-slice-s"  # ok
    minus_slice_e = "minus-slice-e"  # ok
    minus_slice = "minus-slice"  # ok
    leave_slice_m = "leave-slice-m"  # ok
    leave_slice_s = "leave-slice-s"  # ok
    leave_slice_e = "leave-slice-e"  # ok
    leave_slice = "leave-slice"  # ok
    block_1x2x2 = "1x2x2-block"  # ok
    block_1x2x3 = "1x2x3-block"  # ok
    block_2x2x2 = "2x2x2-block"  # ok
    block_2x2x3 = "2x2x3-block"  # ok
    block_2x3x3 = "2x3x3-block"  # ok (= f2l)


# Additional patterns that can be implementated later
class Roux(Enum):
    fb = "fb"
    sb = "sb"
    cmll = "cmll"
    lse = "lse"
    two_blocks = "two-blocks"


class ZZ(Enum):
    eo_line = "eo-line"
    eo_cross = "eo-cross"
    zz_f2l = "zz-f2l"
    ll = "last-layer"


class Petrus(Enum):
    pass


class Patterns(Enum):
    superflip = "superflip"
    checkerboard = "checkerboard"
    cube_in_cube = "cube-in-cube"
    cube_in_cube_in_cube = "cube-in-cube-in-cube"