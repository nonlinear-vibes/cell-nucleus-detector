import cv2
import numpy as np

def process_tile(tile: np.ndarray):
    L_mask = tile[:, :, 0] < 140
    b_mask = tile[:, :, 2] < 110

    # L mask for brownish nuclei
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    L_mask = cv2.medianBlur(L_mask.astype(np.uint8) * 255, 5)
    L_mask = cv2.dilate(L_mask, kernel, iterations=2)
    dist_L = cv2.distanceTransform(L_mask, cv2.DIST_L1, 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    dist_L_dilated = cv2.dilate(dist_L, kernel)
    local_max_L = (dist_L == dist_L_dilated)
    local_max_L[L_mask == 0] = 0
    local_max_L = local_max_L.astype(np.uint8) * 255
    
    # b mask for bluish nuclei
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    b_mask = cv2.medianBlur(b_mask.astype(np.uint8) * 255, 11)
    b_mask = cv2.dilate(b_mask, kernel, iterations=3)
    dist_b = cv2.distanceTransform(b_mask, cv2.DIST_L1, 0)

    dist_b_dilated = cv2.dilate(dist_b, kernel)
    local_max_b = (dist_b == dist_b_dilated)
    local_max_b[b_mask == 0] = 0
    local_max_b = local_max_b.astype(np.uint8) * 255

    # combine masks
    mask = L_mask.astype(bool) | b_mask.astype(bool)
    mask = mask.astype(np.uint8)*255

    # dilate markers to merge close ones
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    max_L_merged = cv2.dilate(local_max_L, kernel)
    max_b_merged = cv2.dilate(local_max_b, kernel, iterations=2)
    all_markers = cv2.bitwise_or(max_L_merged, max_b_merged)

    # watershed segmentation
    _, markers = cv2.connectedComponents(all_markers)
    unknown = cv2.subtract(mask, all_markers.astype(np.uint8))
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = markers.astype(np.int32)
    markers = cv2.watershed(tile, markers)
    labels = np.unique(markers)
    labels = labels[labels > 1]

    # compute bounding boxes (allows much faster processing downstream)
    bboxes = {}
    for label in labels:
        ys, xs = np.nonzero(markers == label)
        bboxes[label] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)


    # helper functions
    def touches_border(region_mask):
        return (
            region_mask[1, :].any() or
            region_mask[H - 2, :].any() or
            region_mask[:, 1].any() or
            region_mask[:, W - 2].any()
        )
    
    def classify_brownness(score):
        if score < 115:
            return 0
        elif score < 140:
            return 1
        elif score < 160:
            return 2
        else:
            return 3

    CLASS_COLORS = {
        0: (255, 0, 0),     # blue
        1: (0, 255, 255),   # yellow
        2: (0, 165, 255),   # orange
        3: (0, 0, 255),     # red
    }

    

    H, W = markers.shape
    output = np.zeros_like(tile)

    # analyze each marker region
    for label in labels:
        region_mask = (markers == label)
        if touches_border(region_mask):
            continue

        x_min, y_min, x_max, y_max = bboxes[label]
        roi = region_mask[y_min:y_max, x_min:x_max].astype(np.uint8)

        mean_L, mean_a, mean_b, _ = cv2.mean(tile[y_min:y_max, x_min:x_max], mask=roi)
        score = (mean_b + 0.5 * mean_a + (255 - mean_L)) / 2.5
        cls = classify_brownness(score)
        color = CLASS_COLORS[cls]

        # get contours (they are a bit ragged, could be rounded)
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cnt[:, 0, :] += (x_min, y_min)
        cv2.drawContours(output, contours, -1, color, thickness=2)
        
    return output


def detect_nuclei(input_path: str, output_path: str, n: int = 32, overlap: int = 128):
    
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Image at path {input_path} not found.")
    
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    outlines = np.zeros_like(image)
    H, W, _ = image.shape

    H_step = H // n
    W_step = W // n

    # process each tile (embarassingly parallel: can be delegeted to multiple CPU cores in future)
    for i in range(0, n):
        for j in range(0, n):
            
            y0 = max(0, H_step * i - overlap)
            y1 = min(H, H_step * (i + 1))
            x0 = max(0, W_step * j - overlap)
            x1 = min(W, W_step * (j + 1))

            tile_outlines = process_tile(lab[y0:y1, x0:x1])
            outlines[y0:y1, x0:x1] = cv2.max(outlines[y0:y1, x0:x1], tile_outlines)

    overlay = image.copy()
    mask = np.any(outlines > 0, axis=2)
    overlay[mask] = outlines[mask]

    cv2.imwrite(output_path, overlay)
    


def main():
        detect_nuclei("img/src.jpg", "img/output.jpg")
        
if __name__ == "__main__":
    main()