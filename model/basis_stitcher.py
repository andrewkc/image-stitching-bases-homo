import cv2
import numpy as np
import torch

# (A) Convertir flujo denso BasisHomo -> Homografía 3×3 aproximada
def flow_to_homography(flow):
    flow = flow.detach().cpu().numpy()
    _, h, w = flow.shape

    corners = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)

    dst = []
    for (x, y) in corners:
        dx, dy = flow[:, int(y), int(x)]
        dst.append([x + dx, y + dy])

    src_pts = corners
    dst_pts = np.array(dst, dtype=np.float32)

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return H


# (B) Warpeo directo usando flujo denso BasisHomo (sin matriz)
def warp_with_flow(img, flow):
    if isinstance(flow, torch.Tensor):
        flow = flow.detach().cpu().numpy()
    img_h, img_w = img.shape[:2]
    _, h, w = flow.shape

    # Reescalar flujo al tamaño de la imagen original
    flow_resized = np.zeros((2, img_h, img_w), dtype=np.float32)
    flow_resized[0] = cv2.resize(flow[0], (img_w, img_h))
    flow_resized[1] = cv2.resize(flow[1], (img_w, img_h))

    # generar coordenadas base
    X, Y = np.meshgrid(np.arange(img_w), np.arange(img_h))

    map_x = (X + flow_resized[0]).astype(np.float32)
    map_y = (Y + flow_resized[1]).astype(np.float32)

    warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    return warped

# (C) Stitching con blending (usa matriz 3×3 o flow)
def stitch_images(img1, img2, H=None, flow=None):
    if flow is not None:
        img1_warp = warp_with_flow(img1, flow)
    elif H is not None:
        # Warpeo por homografía clásica
        h2, w2 = img2.shape[:2]
        img1_warp = cv2.warpPerspective(img1, H, (w2, h2))
    else:
        raise ValueError("Debes pasar H o flow")

    # Máscaras
    mask1 = (img1_warp.sum(axis=2) > 0).astype(np.uint8)
    mask2 = (img2.sum(axis=2) > 0).astype(np.uint8)

    mask1 = cv2.GaussianBlur(mask1.astype(np.float32), (51,51), 0)
    mask2 = cv2.GaussianBlur(mask2.astype(np.float32), (51,51), 0)

    mask1 = mask1 / (mask1 + mask2 + 1e-5)
    mask2 = 1 - mask1

    stitched = img1_warp * mask1[...,None] + img2 * mask2[...,None]
    stitched = stitched.astype(np.uint8)
    return stitched


# (D) Integración directa con tu modelo BasisHomo
def basis_homo_stitch(model_output, img1, img2, use_direct_flow=True):
    # H_flow_f es el flujo hacia adelante Ia → Ib
    flow_f = model_output["H_flow"][0][0]  # batch 0

    if use_direct_flow:
        stitched = stitch_images(img1, img2, flow=flow_f)
    else:
        H = flow_to_homography(flow_f)
        stitched = stitch_images(img1, img2, H=H)

    return stitched
