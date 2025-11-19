from basis_stitcher import basis_homo_stitch

with torch.no_grad():
    output = model(batch)

img1 = cv2.imread("A.png")
img2 = cv2.imread("B.png")

stitched = basis_homo_stitch(output, img1, img2, use_direct_flow=True)
cv2.imwrite("stitched.png", stitched)
