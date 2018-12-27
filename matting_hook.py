import io
import logging

import numpy as np
from PIL import Image
from scipy import ndimage

LOG = logging.getLogger(__name__)


def init_hook(**params):
    LOG.info('Loaded.')

interploations = {
    0:Image.LANCZOS,
    1:Image.NEAREST,
    2:Image.BICUBIC,
    3:Image.BILINEAR,
}

g_mean = np.array(([126.88, 120.24, 112.19])).reshape([1, 1, 3])
unknown_code = 128

def generate_trimap(alpha):
    trimap = np.copy(alpha)
    k_size = 20
    trimap[np.where((ndimage.grey_dilation(alpha[:, :], size=(k_size, k_size)) - ndimage.grey_erosion(alpha[:, :],
                                                                                                      size=(k_size,
                                                                                                            k_size))) != 0)] = unknown_code
    return trimap

def preprocess(inputs, ctx):
    logging.info('Inputs test: {}'.format(inputs['test_opt']))
    image = inputs.get('image')
    if image is None:
        raise RuntimeError('Missing "image" key in inputs. Provide an image in "image" key')
    mask = inputs.get('mask')
    if mask is None:
        raise RuntimeError('Missing "mask" key in inputs. Provide an mask in "mask" key')
    ctx.interpolation = interploations[int(inputs.get('interpolation', 0))]
    image = Image.open(io.BytesIO(image[0]))
    image = image.convert('RGB')
    ctx.image = image
    image = image.resize((320,320),ctx.interpolation)
    mask = Image.open(io.BytesIO(mask[0]))
    mask = mask.resize((320,320),ctx.interpolation)
    np_mask = np.array(mask)
    input_trimap = generate_trimap(np_mask)
    input_trimap = input_trimap.astype(np.float32)
    image = np.array(image).astype(np.float32)
    input_image = image-g_mean
    return {'input': [input_image],'trimap':[input_trimap]}


def postprocess(outputs, ctx):
    logging.info('Outputs: {}'.format(outputs))
    mask = outputs['refinement_predictions']*255
    mask_image = Image.fromarray((mask*255).astype(np.uint8))
    mask_image = mask_image.resize((ctx.image.size[0],ctx.image.size[1]),ctx.interpolation)
    mask_image = np.array(mask_image).astype(np.float32)/255
    image = np.array(ctx.image).astype(np.float32)
    result = (mask_image*image).astype(np.uint8)
    image_bytes = io.BytesIO()
    Image.fromarray(result).save(image_bytes, format='PNG')
    outputs['output'] = image_bytes.getvalue()
    return outputs