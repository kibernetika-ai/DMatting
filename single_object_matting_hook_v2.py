import logging

import numpy as np
from scipy import ndimage
import cv2
from ml_serving.utils.helpers import get_param, load_image, boolean_string

LOG = logging.getLogger(__name__)


def init_hook(**params):
    LOG.info('Loaded.')


obj_classes = {
    'Person': 1
}


def limit(v, l, r, d):
    if v < l:
        return d
    if v > r:
        return d
    return v


def process(inputs, ct_x, **kwargs):
    original_image, is_video = load_image(inputs, 'inputs')
    if original_image is None:
        raise RuntimeError('Missing "inputs" key in inputs. Provide an image in "inputs" key')

    def _return(result):
        if not is_video:
            result = result[:, :, ::-1]
            result = cv2.imencode('.jpg', result)[1].tostring()
        return {'output': result}

    ratio = 1.0
    w = float(original_image.shape[1])
    h = float(original_image.shape[0])
    if w > h:
        if w > 1024:
            ratio = w / 1024.0
    else:
        if h > 1024:
            ratio = h / 1024.0

    if ratio > 1:
        image = cv2.resize(original_image, (int(w / ratio), int(h / ratio)))
    else:
        image = original_image

    if not boolean_string(get_param(inputs, 'return_origin_size', False)):
        original_image = image

    try:
        area_threshold = int(get_param(inputs, 'area_threshold', 0))
    except:
        area_threshold = 0
    area_threshold = limit(area_threshold, 0, 100, 0)
    try:
        max_objects = int(get_param(inputs, 'max_objects', 1))
    except:
        max_objects = 1
    max_objects = limit(max_objects, 1, 10, 1)

    try:
        pixel_threshold = int(float(get_param(inputs, 'pixel_threshold', 0.5)) * 255)
    except:
        pixel_threshold = int(0.5 * 255)

    pixel_threshold = limit(pixel_threshold, 1, 254, int(0.5 * 255))

    object_classes = [obj_classes.get(get_param(inputs, 'object_class', 'Person'), 1)]
    effect = get_param(inputs,'effect', 'Remove background')  # Remove background,Mask,Blur

    try:
        blur_radius = int(get_param(inputs, 'blur_radius', 2))
    except:
        blur_radius = 2

    blur_radius = limit(blur_radius, 1, 10, 2)

    outputs = ct_x.drivers[0].redict({'inputs': image})
    num_detection = int(outputs['num_detections'][0])
    if num_detection < 1:
        return _return(original_image)

    process_width = image.shape[1]
    process_height = image.shape[0]
    image_area = process_width * process_height
    detection_boxes = outputs["detection_boxes"][0][:num_detection]
    detection_boxes = detection_boxes * [process_height, process_width, process_height, process_width]
    detection_boxes = detection_boxes.astype(np.int32)
    detection_classes = outputs["detection_classes"][0][:num_detection]
    detection_masks = outputs["detection_masks"][0][:num_detection]
    masks = []
    for i in range(num_detection):
        if int(detection_classes[i]) not in object_classes:
            continue
        box = detection_boxes[i]
        mask_image = cv2.resize(detection_masks[i], (box[3] - box[1], box[2] - box[0]), interpolation=cv2.INTER_LINEAR)
        left = max(0, box[1] - 50)
        right = min(process_width, box[3] + 50)
        upper = max(0, box[0] - 50)
        lower = min(process_height, box[2] + 50)
        box_mask = np.pad(mask_image, ((box[0] - upper, lower - box[2]), (box[1] - left, right - box[3])), 'constant')
        area = int(np.sum(np.greater_equal(box_mask, 0.5).astype(np.int32)))
        if area * 100 / image_area < area_threshold:
            continue
        masks.append((area, box_mask, [upper, left, lower, right]))

    if len(masks) < 1:
        return _return(original_image)
    masks = sorted(masks, key=lambda row: -row[0])
    total_mask = np.zeros((process_height, process_width), np.float32)
    min_left = process_width
    min_upper = process_height
    max_right = 0
    max_lower = 0
    for i in range(min(len(masks), max_objects)):
        pre_mask = masks[i][1]
        box = masks[i][2]
        left = max(0, box[1])
        right = min(process_width, box[3])
        upper = max(0, box[0])
        lower = min(process_height, box[2])
        box_mask = np.pad(pre_mask, ((upper, process_height - lower), (left, process_width - right)), 'constant')
        total_mask = np.maximum(total_mask, box_mask)
        if left < min_left:
            min_left = left
        if right > max_right:
            max_right = right
        if upper < min_upper:
            min_upper = upper
        if lower > max_lower:
            max_lower = lower
    mask = np.uint8(total_mask[min_upper:max_lower, min_left:max_right] * 255)
    box = (min_upper, min_left, max_lower, max_right)
    if len(mask.shape) > 2:
        logging.warning('Mask shape is {}'.format(mask.shape))
        mask = mask[:, :, 0]
    image = cv2.resize(image[box[0]:box[2], box[1]:box[3], :], (320, 320))
    mask = cv2.resize(mask, (320, 320))
    mask[np.less_equal(mask, pixel_threshold)] = 0
    mask[np.greater(mask, pixel_threshold)] = 255
    input_trimap = generate_trimap(mask)
    input_trimap = np.expand_dims(input_trimap.astype(np.float32), 2)
    image = image.astype(np.float32)
    input_image = image - g_mean
    outputs = ct_x.drivers.predict({'input': [input_image], 'trimap': [input_trimap]})
    mask = outputs.get('mask', None)
    if mask is None:
        mask = outputs['output'][0] * 255
        mask = np.reshape(mask, (320, 320))
        mask = np.clip(mask, 0, 255)
        mask = mask.astype(np.uint8)
        mask = cv2.resize(mask, (box[3] - box[1], box[2] - box[0]))
    mask = mask.astype(np.float32) / 255
    mask = np.pad(mask,
                  ((box[0], process_height - box[2]), (box[1], process_width - box[3])),
                  'constant')
    if mask.shape != original_image.shape:
        mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
    mask = cv2.GaussianBlur(mask, (21, 21), 11)
    if effect == 'Remove background':
        image = original_image.astype(np.float32)
        mask = np.expand_dims(mask, 2)
        image = image * mask
        image = image.astype(np.uint8)
    elif effect == "Mask":
        mask = mask * 255
        image = mask.astype(np.uint8)
    else:
        image = original_image.astype(np.float32)
        mask = np.expand_dims(mask, 2)
        foreground = mask * image
        radius = min(max(blur_radius, 2), 10)
        if effect == 'Grey':
            background = rgb2gray(original_image)
        else:
            background = cv2.GaussianBlur(original_image, (radius, radius), 10)
        background = (1.0 - mask) * background.astype(np.float32)
        image = foreground + background
        image = image.astype(np.uint8)

    return _return(image)


def rgb2gray(rgb):
    gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    return np.stack([gray, gray, gray], axis=2)


g_mean = np.array(([126.88, 120.24, 112.19])).reshape([1, 1, 3])
unknown_code = 128


def generate_trimap(alpha):
    trimap = np.copy(alpha)
    k_size = 20
    trimap[np.where((ndimage.grey_dilation(alpha[:, :], size=(k_size, k_size)) - ndimage.grey_erosion(alpha[:, :],
                                                                                                      size=(k_size,
                                                                                                            k_size))) != 0)] = unknown_code
    return trimap
