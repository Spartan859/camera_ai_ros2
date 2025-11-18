from __future__ import annotations

import os
import time
from typing import List, Dict, Any

import cv2
import numpy as np

import mindspore_lite as mslite
from mindyolo.data import COCO80_TO_COCO91_CLASS  # noqa: F401  # kept for parity
from mindyolo.utils.metrics import non_max_suppression, scale_coords, xyxy2xywh
from mindyolo.utils import logger


class MSLiteYOLODetector:
    def __init__(
        self,
        mindir_path: str,
        img_size: int = 640,
        conf_thres: float = 0.25,
        iou_thres: float = 0.7,
        conf_free: bool = True,
        nms_time_limit: float = 60.0,
        device_target: str = 'Ascend',
        single_cls: bool = False,
        names: List[str] | None = None,
        logger: Any | None = None,
    ) -> None:
        self._log = logger
        self._log.info('Initializing MSLiteYOLODetector...')
        self.mindir_path = mindir_path
        if not os.path.isfile(self.mindir_path):
            raise FileNotFoundError(f'mindir file not found: {self.mindir_path}')
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.conf_free = conf_free
        self.nms_time_limit = nms_time_limit
        self.device_target = device_target
        self.single_cls = single_cls
        self.names = names or []

        try:
            logger.setup_logging(logger_name='MindYOLO', log_level='INFO')
        except Exception:
            pass
        if self._log:
            self._log.info('Initializing MSLite model context...')
        self.context = mslite.Context()
        self.context.target = [self.device_target]
        self.model = mslite.Model()
        if self._log:
            self._log.info('Building model from mindir...')
        self.model.build_from_file(self.mindir_path, mslite.ModelType.MINDIR, self.context)
        self.inputs = self.model.get_inputs()
        if self._log:
            self._log.info('Model loaded successfully.')

    def _resize_and_pad(self, img: np.ndarray) -> np.ndarray:
        h_ori, w_ori = img.shape[:2]
        r = self.img_size / max(h_ori, w_ori)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w_ori * r), int(h_ori * r)), interpolation=interp)
        h, w = img.shape[:2]
        if h < self.img_size or w < self.img_size:
            new_shape = (self.img_size, self.img_size)
            dh, dw = (new_shape[0] - h) / 2, (new_shape[1] - w) / 2
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        img = self._resize_and_pad(img)
        img = img[:, :, ::-1].transpose(2, 0, 1) / 255.0
        img = np.array(img[None], dtype=np.float32)
        return np.ascontiguousarray(img)

    def predict(self, img: np.ndarray) -> Dict[str, Any]:
        if not isinstance(img, np.ndarray) or img.ndim != 3:
            raise ValueError('img must be a HxWx3 numpy.ndarray')
        h_ori, w_ori = img.shape[:2]

        processed = self._preprocess(img)

        t0 = time.time()
        self.model.resize(self.inputs, [list(processed.shape)])
        self.inputs[0].set_data_from_numpy(processed)
        outputs = self.model.predict(self.inputs)
        outputs = [o.get_data_to_numpy().copy() for o in outputs]
        out = outputs[0]
        infer_time = time.time() - t0

        t1 = time.time()
        out = non_max_suppression(
            out,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            conf_free=self.conf_free,
            multi_label=True,
            time_limit=self.nms_time_limit,
        )
        nms_time = time.time() - t1

        result_dict: Dict[str, Any] = {"category_id": [], "bbox": [], "score": []}
        total_category_ids, total_bboxes, total_scores = [], [], []
        for pred in out:
            if len(pred) == 0:
                continue
            predn = np.copy(pred)
            scale_coords(processed.shape[2:], predn[:, :4], (h_ori, w_ori))
            box = xyxy2xywh(predn[:, :4])
            box[:, :2] -= box[:, 2:] / 2
            for p, b in zip(pred.tolist(), box.tolist()):
                cls_id = int(p[5])
                total_category_ids.append(cls_id)
                total_bboxes.append([int(x) for x in b])
                total_scores.append(round(p[4], 5))

        result_dict['category_id'].extend(total_category_ids)
        result_dict['bbox'].extend(total_bboxes)
        result_dict['score'].extend(total_scores)

        if self._log:
            self._log.info('Speed: %.1f/%.1f/%.1f ms inference/NMS/total for %gx%g image.' % (
                infer_time * 1e3, nms_time * 1e3, (infer_time + nms_time) * 1e3, self.img_size, self.img_size
            ))
        return result_dict

    __call__ = predict


__all__ = ['MSLiteYOLODetector']
