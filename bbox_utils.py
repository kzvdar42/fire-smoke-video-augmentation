# Part of the code is from https://github.com/Paperspace/DataAugmentationForObjectDetection.
import cv2
import numpy as np
from threading import Thread, Lock
from itertools import cycle


def synchronized(func):
    """Decorator for the syncronized usage of the function."""
    func.__lock__ = Lock()

    def synced_func(*args, **kws):
        with func.__lock__:
            return func(*args, **kws)

    return synced_func


class ImagesReader:
    def __init__(self, images, buffer_size=32):
        self.images = images
        self.total = len(images)
        self.buffer_size = buffer_size
        self.next_id = 0
        self.buffer = []
        self._fill_buffer()

    def is_open(self):
        return self.__is_open() or len(self.buffer)

    def __is_open(self):
        return self.next_id < self.total

    @synchronized
    def __get_next_id(self):
        old_next_id = self.next_id
        self.next_id += 1
        return old_next_id

    def __read_to_buffer(self):
        image_path = self.images[self.__get_next_id()]
        self.buffer.append((image_path, cv2.imread(image_path)))

    @synchronized
    def _fill_buffer(self):
        if self.__is_open():
            for _ in range(self.buffer_size - len(self.buffer)):
                Thread(target=self.__read_to_buffer).start()

    def __iter__(self):
        return self

    def __next__(self):
        if not self.is_open():
            raise StopIteration
        while self.is_open():
            if len(self.buffer):
                if self.__is_open():
                    Thread(target=self.__read_to_buffer).start()
                return self.buffer.pop()


def resize(image, size, bboxes=None):
    scale_ratio = get_scale_ratio(image, size)
    image = resize_by_max_side(image, scale_ratio)
    if bboxes is not None:
        bboxes = bboxes * scale_ratio
    return image, bboxes


def flip(image, bboxes=None):
    image = cv2.flip(image, 1)

    if bboxes is not None:
        bboxes[:, [0, 2]] += 2 * ((image.shape[1] / 2) - bboxes[:, [0, 2]])

        box_w = abs(bboxes[:, 0] - bboxes[:, 2])

        bboxes[:, 0] -= box_w
        bboxes[:, 2] += box_w
    return image, bboxes


def rotate(image, angle, bboxes=None):
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2

    # Rotate image.
    image = rotate_im(image, angle)

    # Scale back
    scale_factor_x = image.shape[1] / w
    scale_factor_y = image.shape[0] / h

    image = cv2.resize(image, (w, h))

    if bboxes is not None:
        corners = get_corners(bboxes)
        # Rotate corners.
        corners = rotate_box(corners, angle, cx, cy, h, w)
        # Calculate enclosing bbox.
        bboxes = get_enclosing_box(corners)
        # Scale back.
        bboxes[:, :4] /= [scale_factor_x, scale_factor_y] * 2

    # new_bboxes = clip_box(new_bboxes, [0, 0, w, h], 0.25)
    return image, bboxes


def gamma_correction(image, gamma=1):
    lut = pow(np.array([range(256)], dtype=np.float32) / 255.0, gamma) * 255.0
    lut = lut.astype(np.uint8)
    return cv2.LUT(image, lut)


def from_ratio_to_pixel(boxes, width, height):
    boxes[:, 0] *= width
    boxes[:, 1] *= height
    boxes[:, 2] *= width
    boxes[:, 3] *= height
    return boxes.astype(np.int32)


def get_scale_ratio(img, to_max_size):
    """Calculate scale ratio.

    :param img: image to scale
    :param to_max_size: max image size after the scale."""
    return np.divide(to_max_size, np.max(img.shape[:2]), dtype=np.float32)


def resize_by_max_side(img, ratio):
    resize_shape = (np.array(img.shape[:2][::-1], dtype=np.float32) * ratio).astype(np.uint)
    return cv2.resize(img, tuple(resize_shape))


def convert_xywh_xyxy(bbox, width, height):
    width, height = width - 1, height - 1
    x1 = np.max((0, bbox[0]))
    y1 = np.max((0, bbox[1]))
    x2 = np.min((width, x1 + np.max((0, bbox[2] - 1))))
    y2 = np.min((height, y1 + np.max((0, bbox[3] - 1))))
    return np.array([x1, y1, x2, y2])


def convert_xyxy_xywh(bbox):
    bbox = bbox.copy()
    bbox[2] = bbox[2] - bbox[0]
    bbox[3] = bbox[3] - bbox[1]
    return bbox


def rotate_im(image, angle):
    """Rotate the image.

    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black. 

    Parameters
    ----------

    image : numpy.ndarray
        numpy image

    angle : float
        angle by which the image is to be rotated

    Returns
    -------

    numpy.ndarray
        Rotated Image

    """
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def get_corners(bboxes):
    """Get corners of bounding boxes

    Parameters
    ----------

    bboxes: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`      

    """
    width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
    height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)

    x1 = bboxes[:, 0].reshape(-1, 1)
    y1 = bboxes[:, 1].reshape(-1, 1)

    x2 = x1 + width
    y2 = y1

    x3 = x1
    y3 = y1 + height

    x4 = bboxes[:, 2].reshape(-1, 1)
    y4 = bboxes[:, 3].reshape(-1, 1)

    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

    return corners


def rotate_box(corners, angle,  cx, cy, h, w):
    """Rotate the bounding box.


    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    angle : float
        angle by which the image is to be rotated

    cx : int
        x coordinate of the center of image (about which the box will be rotated)

    cy : int
        y coordinate of the center of image (about which the box will be rotated)

    h : int 
        height of the image

    w : int 
        width of the image

    Returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """

    corners = corners.reshape(-1, 2)
    corners = np.hstack(
        (corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M, corners.T).T

    calculated = calculated.reshape(-1, 8)

    return calculated


def get_enclosing_box(corners):
    """Get an enclosing box for ratated corners of a bounding box

    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`  

    Returns 
    -------

    numpy.ndarray
        Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    """
    x_ = corners[:, [0, 2, 4, 6]]
    y_ = corners[:, [1, 3, 5, 7]]

    xmin = np.min(x_, 1).reshape(-1, 1)
    ymin = np.min(y_, 1).reshape(-1, 1)
    xmax = np.max(x_, 1).reshape(-1, 1)
    ymax = np.max(y_, 1).reshape(-1, 1)

    final = np.hstack((xmin, ymin, xmax, ymax, corners[:, 8:]))

    return final
