"""Watermark trace piece-picking: inside a padded YOLO box keep the watermark's pieces, drop art poking in from the
crop's edge (ears, hair), shape-blind (a round crest is kept like text). Pure numpy, no SAM."""
import numpy as np

from fizgig.dataset.detectors import pick_pieces

H, W = 100, 160
INNER = (40, 30, 120, 70)          # the YOLO box in crop coords


def _blank():
    return np.zeros((H, W), bool)


def test_centre_kept_edge_blobs_dropped():
    t = _blank()
    t[40:60, 55:105] = True        # signature, centred
    t[0:35, 10:30] = True          # ear tip from the top edge
    t[0:32, 125:150] = True        # hair tip from the top edge
    out = pick_pieces(t, INNER)
    assert out[50, 80] and not out[10, 20] and not out[10, 140]
    assert out.sum() == 20 * 50


def test_round_crest_kept():
    t = _blank()
    yy, xx = np.mgrid[:H, :W]
    t[(yy - 50) ** 2 + (xx - 80) ** 2 <= 15 ** 2] = True
    t[85:100, 0:20] = True          # art in the corner
    out = pick_pieces(t, INNER)
    assert out[50, 80] and out[50, 94] and not out[95, 5]


def test_letters_chain_kept():
    t = _blank()
    for x in (45, 62, 79, 96):     # separate letters a few px apart
        t[40:60, x:x + 12] = True
    out = pick_pieces(t, INNER)
    assert out.sum() == 4 * 20 * 12


def test_nothing_inside_keeps_most_inside():
    t = _blank()
    t[0:35, 30:60] = True          # 5 rows in the box
    t[0:20, 130:160] = True        # none in the box
    out = pick_pieces(t, INNER)
    assert out[5, 40] and not out[5, 140]
