import numpy as np  # type: ignore

def normalize_landmarks(raw_63):
    """Normalize 21 landmarks relative to wrist position."""
    pts = np.array(raw_63).reshape(21, 3)
    wrist = pts[0]
    pts = pts - wrist                    # translate to origin
    scale = np.max(np.linalg.norm(pts, axis=1)) + 1e-8
    pts = pts / scale                    # scale invariant
    return pts.flatten().tolist()

def build_angle_features(raw_63):
    """Compute finger bend angles as additional features."""
    pts = np.array(raw_63).reshape(21, 3)
    # Finger tip indices: [4, 8, 12, 16, 20]
    # MCP indices:        [2, 5,  9, 13, 17]
    angles = []
    for tip, mcp in zip([4,8,12,16,20], [2,5,9,13,17]):
        v = pts[tip] - pts[mcp]
        angle = np.arctan2(np.linalg.norm(v[:2]), v[2])
        angles.append(angle)
    return angles