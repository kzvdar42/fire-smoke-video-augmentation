# Code taken and improved from https://github.com/jrosebr1/color_transfer
# import the necessary packages
import numpy as np
import cv2

def color_transfer(source, target, clip=True, preserve_paper=True):
    """
    Transfers the color distribution from the source to the target
    image using the mean and standard deviations of the L*a*b*
    color space.

    This implementation is (loosely) based on to the "Color Transfer
    between Images" paper by Reinhard et al., 2001.

    Parameters:
    -------
    source: NumPy array
        OpenCV image in BGR color space (the source image)
    target: NumPy array
        OpenCV image in BGR color space (the target image)
    clip: Should components of L*a*b* image be scaled by np.clip before 
        converting back to BGR color space?
        If False then components will be min-max scaled appropriately.
        Clipping will keep target image brightness truer to the input.
        Scaling will adjust image brightness to avoid washed out portions
        in the resulting color transfer that can be caused by clipping.
    preserve_paper: Should color transfer strictly follow methodology
        layed out in original paper? The method does not always produce
        aesthetically pleasing results.
        If False then L*a*b* components will scaled using the reciprocal of
        the scaling factor proposed in the paper.  This method seems to produce
        more consistently aesthetically pleasing results 

    Returns:
    -------
    transfer: NumPy array
        OpenCV image (w, h, 3) NumPy array (uint8)
    """
    # convert the images from the RGB to L*ab* color space, being
    # sure to utilizing the floating point data type (note: OpenCV
    # expects floats to be 32-bit, so use that instead of 64-bit)
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_alpha = target[:, :, 3]
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    # compute color statistics for the source and target images
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target, target_alpha)

    # subtract the means from the target image
    (l, a, b) = cv2.split(target.copy())
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar

    try:
        if preserve_paper:
            # scale by the standard deviations using paper proposed factor
            # if lStdSrc != 0:
                # l = (lStdTar / lStdSrc) * l
            if aStdSrc != 0:
                a = (aStdTar / aStdSrc) * a
            if bStdSrc != 0:
                b = (bStdTar / bStdSrc) * b
        else:
            # scale by the standard deviations using reciprocal of paper proposed factor
            # if lStdTar != 0:
                # l = (lStdSrc / lStdTar) * l
            if aStdTar != 0:
                a = (aStdSrc / aStdTar) * a
            if bStdTar != 0:
                b = (bStdSrc / bStdTar) * b
    except RuntimeWarning:
        print(lStdSrc, )

    # add in the source mean
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc

    # clip/scale the pixel intensities to [0, 255] if they fall
    # outside this range
    l = _scale_array(l, clip=clip)
    a = _scale_array(a, clip=clip)
    b = _scale_array(b, clip=clip)

    # merge the channels together and convert back to the RGB color
    # space, being sure to utilize the 8-bit unsigned integer data
    # type

    # (_, a, b) = cv2.split(target)
    # (l, aa, bb) = cv2.split(target)

    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # return the color transferred image
    return transfer

def image_stats(image, alpha=None):
    """
    Parameters:
    -------
    image: NumPy array
        OpenCV image in L*a*b* color space

    Returns:
    -------
    Tuple of mean and standard deviations for the L*, a*, and b*
    channels, respectively
    """
    # compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    if alpha is None:
        (lMean, lStd) = (l.mean(), l.std())
        (aMean, aStd) = (a.mean(), a.std())
        (bMean, bStd) = (b.mean(), b.std())
    else:
        (lMean, lStd) = (l[alpha != 0].mean(), l[alpha != 0].std())
        (aMean, aStd) = (a[alpha != 0].mean(), a[alpha != 0].std())
        (bMean, bStd) = (b[alpha != 0].mean(), b[alpha != 0].std())

    # return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)

def _min_max_scale(arr, new_range=(0, 255)):
	"""
	Perform min-max scaling to a NumPy array

	Parameters:
	-------
	arr: NumPy array to be scaled to [new_min, new_max] range
	new_range: tuple of form (min, max) specifying range of
		transformed array

	Returns:
	-------
	NumPy array that has been scaled to be in
	[new_range[0], new_range[1]] range
	"""
	# get array's current min and max
	mn = arr.min()
	mx = arr.max()

	# check if scaling needs to be done to be in new_range
	if mn < new_range[0] or mx > new_range[1]:
		# perform min-max scaling
		scaled = (new_range[1] - new_range[0]) * (arr - mn) / (mx - mn) + new_range[0]
	else:
		# return array if already in range
		scaled = arr

	return scaled

def _scale_array(arr, clip=True):
	"""
	Trim NumPy array values to be in [0, 255] range with option of
	clipping or scaling.

	Parameters:
	-------
	arr: array to be trimmed to [0, 255] range
	clip: should array be scaled by np.clip? if False then input
		array will be min-max scaled to range
		[max([arr.min(), 0]), min([arr.max(), 255])]

	Returns:
	-------
	NumPy array that has been scaled to be in [0, 255] range
	"""
	if clip:
		scaled = np.clip(arr, 0, 255)
	else:
		scale_range = (max([arr.min(), 0]), min([arr.max(), 255]))
		scaled = _min_max_scale(arr, new_range=scale_range)

	return scaled

def hist_norm(source, template):

    olddtype = source.dtype
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    interp_t_values = interp_t_values.astype(olddtype)

    return interp_t_values[bin_idx].reshape(oldshape)


"""Apply Simplest Color Balance algorithm
Reimplemented based on https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc"""
def simplest_cb(img, percent):
    out_channels = []
    channels = cv2.split(img)
    totalstop = channels[0].shape[0] * channels[0].shape[1] * percent / 200.0
    for channel in channels:
        bc = np.bincount(channel.ravel(), minlength=256)
        lv = np.searchsorted(np.cumsum(bc), totalstop)
        hv = 255-np.searchsorted(np.cumsum(bc[::-1]), totalstop)
        out_channels.append(cv2.LUT(channel, np.array(tuple(0 if i < lv else 255 if i > hv else round((i-lv)/(hv-lv)*255) for i in np.arange(0, 256)), dtype="uint8")))
    return cv2.merge(out_channels)