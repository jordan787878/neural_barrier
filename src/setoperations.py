import numpy as np
from typing import List


def rect_diff_2d(
    outer: np.ndarray, 
    inner: np.ndarray,
    tol: float = 0.0
) -> List[np.ndarray]:
    """
    Compute outer \ inner for two 2D axis-aligned rectangles.

    Parameters
    ----------
    outer : array_like, shape (2,2)
        [[x0, x1],
         [y0, y1]]
    inner : array_like, shape (2,2)
        [[ix0, ix1],
         [iy0, iy1]]
    tol : float
        small tolerance to treat edges as touching

    Returns
    -------
    boxes : list of (2,2) arrays
        Disjoint rectangles whose union = outer minus inner.
    """
    outer = np.asarray(outer, dtype=float)
    inner = np.asarray(inner, dtype=float)
    x0, x1 = outer[0]
    y0, y1 = outer[1]
    ix0, ix1 = inner[0]
    iy0, iy1 = inner[1]

    # 1. Clamp the inner to the outer’s extent
    ix0_cl = max(x0, ix0)
    ix1_cl = min(x1, ix1)
    iy0_cl = max(y0, iy0)
    iy1_cl = min(y1, iy1)

    # 2. No overlap?
    if ix0_cl >= ix1_cl - tol or iy0_cl >= iy1_cl - tol:
        return [outer.copy()]

    # 3. Full cover?
    if ix0_cl <= x0 + tol and ix1_cl >= x1 - tol \
       and iy0_cl <= y0 + tol and iy1_cl >= y1 - tol:
        return []

    boxes = []

    # Left strip
    if ix0_cl > x0 + tol:
        boxes.append(np.array([[x0, ix0_cl], [y0, y1]], dtype=float))

    # Right strip
    if ix1_cl < x1 - tol:
        boxes.append(np.array([[ix1_cl, x1], [y0, y1]], dtype=float))

    # Bottom strip (only between ix0_cl and ix1_cl)
    if iy0_cl > y0 + tol:
        boxes.append(np.array([[ix0_cl, ix1_cl], [y0, iy0_cl]], dtype=float))

    # Top strip
    if iy1_cl < y1 - tol:
        boxes.append(np.array([[ix0_cl, ix1_cl], [iy1_cl, y1]], dtype=float))

    return boxes


def refine_2d_box(box: np.ndarray, refine_N: int) -> List[np.ndarray]:
    """
    Subdivide a 2D axis‑aligned rectangle into an N×N grid of smaller rectangles.
    
    Parameters
    ----------
    box : array_like, shape (2,2)
        [[x0, x1],
         [y0, y1]]
    refine_N : int
        Number of subdivisions along each axis.
    
    Returns
    -------
    List of length refine_N**2, each an ndarray of shape (2,2):
        [[x_lo, x_hi],
         [y_lo, y_hi]]
    """
    box = np.asarray(box, dtype=float)
    if box.shape != (2,2):
        raise ValueError("`box` must be shape (2,2)")
    x0, x1 = box[0]
    y0, y1 = box[1]
    # grid lines
    xs = np.linspace(x0, x1, refine_N+1)
    ys = np.linspace(y0, y1, refine_N+1)

    rects: List[np.ndarray] = []
    for i in range(refine_N):
        for j in range(refine_N):
            rect = np.array([
                [xs[i],   xs[i+1]],
                [ys[j],   ys[j+1]]
            ], dtype=box.dtype)
            rects.append(rect)
    return rects


def refine_list_2d_box(list_2dbox, refine_N):
    result = []
    for rect in list_2dbox:
        result = result + refine_2d_box(rect, refine_N)
    return result

