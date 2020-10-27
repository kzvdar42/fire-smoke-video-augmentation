# Part of the code is from https://github.com/Paperspace/DataAugmentationForObjectDetection.
import cv2
import numpy as np


def prepare_image(image):
    if image.dtype == np.uint8:
        return image
    else:
        image = image / np.iinfo(image.dtype).max * 255
        return image.astype(np.uint8)


def resize(image, size, bboxes=None, segments=None):
    scale_ratio = get_scale_ratio(image, size)
    image = resize_by_max_side(image, scale_ratio)
    if bboxes is not None:
        bboxes = bboxes * scale_ratio
    if segments is not None:
        segments = segments * scale_ratio
    return image, bboxes, segments


def flip(image, bboxes=None, segments=None):
    image = cv2.flip(image, 1)

    if bboxes is not None:
        bboxes[:, [0, 2]] += 2 * ((image.shape[1] / 2) - bboxes[:, [0, 2]])

        box_w = abs(bboxes[:, 0] - bboxes[:, 2])

        bboxes[:, 0] -= box_w
        bboxes[:, 2] += box_w
    if segments is not None:
        for segment in segments:
            segment[..., [0]] += 2 * ((image.shape[1] / 2) - segment[..., [0]])
    return image, bboxes, segments


def rotate(image, angle, bboxes=None, segments=None):
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2

    # Rotate image.
    M, (nW, nH) = get_rotation_matrix(angle, cx, cy, h, w)
    image = cv2.warpAffine(image, M, (nW, nH))

    # Scale back
    scale_factor_x = image.shape[1] / w
    scale_factor_y = image.shape[0] / h

    image = cv2.resize(image, (w, h))

    enclosing_polygons = None
    if segments is not None:
        enclosing_polygons = []
        for segment in segments:
            seg = rotate_poly(segment, M)
            seg = seg / (scale_factor_x, scale_factor_y)
            enclosing_polygons.append(seg)
        segments = enclosing_polygons = np.array(enclosing_polygons)
    elif bboxes is not None:
        corners = get_corners(bboxes)
        # Rotate corners.
        corners = rotate_box(corners, M)
        # Calculate enclosing bbox.
        bboxes = get_enclosing_box(corners)
        # Scale back.
        bboxes[:, :4] /= [scale_factor_x, scale_factor_y] * 2
    return image, bboxes, segments


def gamma_correction(image, gamma=1):
    lut = pow(np.array([range(256)], dtype=np.float32) / 255.0, gamma) * 255.0
    lut = lut.astype(np.uint8)
    return cv2.LUT(image, lut)


def blur_contour(image, blur_radius, contour_radius, blur_image=True, blur_alpha=True):
    image, alpha = image[:, :, :3], image[:, :, 3:]

    if blur_image:
        contours, _ = cv2.findContours(alpha.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        b_rad = (blur_radius, blur_radius)
        blurred_img = cv2.GaussianBlur(image, b_rad, 0)
        mask = np.zeros((*image.shape[:2], 1), np.ubyte)
        cv2.drawContours(mask, contours, -1, (1), contour_radius)
        image = np.where(mask, blurred_img, image)
    
    if blur_alpha:
        alpha[:, :, 0] = cv2.GaussianBlur(alpha, b_rad, 0)

    return np.concatenate((image, alpha), -1)


def trim_empty_pixels(image, segments=None, tolerance=0):
    if segments is not None:
        raise NotImplementedError()
    # img is 2D or 3D image data
    mask = image > tolerance
    if image.ndim == 3:
        mask = mask.all(2)
    m, n = mask.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
    return image[row_start:row_end,
                 col_start:col_end]


def add_shadow(e_image, offset, blur_radius, max_shadow_opacity, segments=None, e_info=None):
    def warpImage(img, offset, obj_off, shape, start_pos=0):
        h, w = shape
        ow, oh = obj_off
        offx, offy = offset
        pts1 = np.float32([[ow, oh],           [ow+w-1, oh],           [ow, oh+h-1-start_pos], [ow+w-1, oh+h-1-start_pos]])
        pts2 = np.float32([[ow-offx, oh-offy], [ow+w-1-offx, oh-offy], [ow, oh+h-1-start_pos], [ow+w-1, oh+h-1-start_pos]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, M, img.shape[:2][::-1])
        return dst

    h, w = e_image.shape[:2]

    # Check if close to zero (+- 3)
    is_close_to_zero = np.isclose(offset, [0, 0], atol=3)
    offset[0] = 0 if is_close_to_zero[0] else int(e_image.shape[1] * (offset[0] / 100))
    offset[1] = 0 if is_close_to_zero[1] else int(e_image.shape[0] * (offset[1] / 100))


    if offset[1] < -h:
        new_h = -offset[1]
        # Assume that object starts not at the edge of the image
        start_pos = int(e_image.shape[1] * 0.05)
    else:
        new_h = h + (offset[1] if offset[1] > 0 else 0)
        start_pos = 0
    new_w = w + abs(offset[0])
    new_image = np.zeros((new_h, new_w, 4), dtype=np.uint8)

    if offset[0] > 0:
        if offset[1] < 0:
            obj_off = [offset[0], 0]
        else:
            obj_off = [offset[0], offset[1]]
    else:
        if offset[1] <= 0:
            obj_off = [0, 0]
        else:
            obj_off = [0, offset[1]]
    
    new_image[obj_off[1]: h + obj_off[1],
              obj_off[0]: w + obj_off[0]] = e_image
    if segments is not None:
        segments[..., [0]] += obj_off[0]
        segments[..., [1]] += obj_off[1]
    if e_info is not None:
        e_info.offset = [e_info.offset[0] - obj_off[0],
                         e_info.offset[1] - obj_off[1]]
        e_info.c_offset = [e_info.c_offset[0] - obj_off[0],
                           e_info.c_offset[1] - obj_off[1]]

    # Padd shadow for proper blur.
    blur_pad = blur_radius // 4
    shadow = np.zeros((blur_pad * 2 + new_image.shape[0],
                       blur_pad * 2 + new_image.shape[1],
                       new_image.shape[2] - 3))
    shadow[blur_pad:-blur_pad,
           blur_pad:-blur_pad] = new_image[:, :, 3:] * max_shadow_opacity
    shadow = warpImage(shadow, offset, obj_off, (h, w), start_pos)
    if blur_radius > 0:
        b_rad = (blur_radius, blur_radius)
        shadow = cv2.GaussianBlur(shadow, b_rad, 0)
    if len(shadow.shape) == 2:
        shadow = np.expand_dims(shadow, -1)
    shadow = shadow[blur_pad:-blur_pad, blur_pad:-blur_pad]
    
    mask = new_image[..., 3:] / 255
    new_image[..., :3] = new_image[..., :3] * mask
    new_image[..., 3:] = new_image[..., 3:] + shadow * (1 - mask)
    return new_image, segments


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

def get_rotation_matrix(angle, cx, cy, h, w):
    # grab the dimensions of the image and then determine the
    # centre

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    return M, (nW, nH)


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


def rotate_box(corners, M):
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
    corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))
    calculated = np.dot(M, corners.T).T
    return calculated.reshape(-1, 8)

def rotate_poly(poly, M):
    """Rotate the polygon."""
    shape = poly.shape
    poly = poly.reshape(-1, 2)
    poly = np.hstack((poly, np.ones((poly.shape[0], 1), dtype=poly[0].dtype)))
    calculated = np.dot(M, poly.T).T
    return calculated.reshape(*shape)

def get_intersection(bb1, bb2):
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left   = max(bb1[0], bb2[0])
    y_top    = max(bb1[1], bb2[1])
    x_right  = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    return (x_right - x_left) * (y_bottom - y_top)


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : list
        Keys: [x1, y1, x2, y2]
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : list
        Keys: [x1, y1, x2, y2]
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left   = max(bb1[0], bb2[0])
    y_top    = max(bb1[1], bb2[1])
    x_right  = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def get_enclosing_box(corners):
    """Get an enclosing box for rotated corners of a bounding box

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
