import sys
sys.path.insert(0, "cuda")
import nms_cuda
import numpy as np

boxes  = np.random.rand(8400, 4).astype(np.float32)
scores = np.random.rand(8400).astype(np.float32)
keep   = nms_cuda.run_nms(boxes, scores, iou_threshold=0.45, conf_threshold=0.25)
print(keep.sum(), "boxes kept out of 8400")