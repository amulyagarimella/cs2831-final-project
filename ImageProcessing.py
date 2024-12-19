import numpy as np
import cv2 as cv
from scipy.signal import convolve2d
import math
import scipy.ndimage
import torch.nn.functional as F
import torch 


def imBrightness3D(img, In=([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]), Out=([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])):
    # "J = low_out +(high_out - low_out).* ((I - low_in)/(high_in - low_in)).^ gamma"
    # Modified from this code: https://www.programmersought.com/article/32635116380/

    if img.max() > 1 or img.min() < 0:
        raise ValueError('Pixel values must be rescaled to zero to one.')

    def imgconvert(simg, h, w, k, low_in, high_in, low_out, high_out):
        imgOut = np.zeros((h,w))
        [x_low, y_low] = np.where(simg <= low_in)
        [x_high, y_high] = np.where(simg > high_in)
        [x_mid, y_mid] = np.where((simg > low_in) & (simg <= high_in))
        imgOut[x_low, y_low] = low_out
        imgOut[x_high, y_high] = high_out
        imgOut[x_mid, y_mid] = k * (simg[x_mid,y_mid] - low_in) + low_out
        return imgOut

    ([r_low_in, g_low_in, b_low_in], [r_high_in, g_high_in, b_high_in]) = In
    ([r_low_out, g_low_out, b_low_out], [r_high_out, g_high_out, b_high_out]) = Out


    r_k = (r_high_out - r_low_out) / (r_high_in - r_low_in)
    g_k = (g_high_out - g_low_out) / (g_high_in - g_low_in)
    b_k = (b_high_out - b_low_out) / (b_high_in - b_low_in)

    h, w = img.shape[:2]

    r_imgOut = imgconvert(img[:,:,0], h, w, r_k, r_low_in, r_high_in, r_low_out, r_high_out)
    g_imgOut = imgconvert(img[:,:,1], h, w, g_k, g_low_in, g_high_in, g_low_out, g_high_out)
    b_imgOut = imgconvert(img[:,:,2], h, w, b_k, b_low_in, b_high_in, b_low_out, b_high_out)

    imgOut = cv.merge((r_imgOut, g_imgOut, b_imgOut))

    return imgOut

def auto_canny(image, sigma = 0.33):
    """
    Canny edge detection without lower- and upper-bound setting.
    Referred from: https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
    :param image: gray scale image
    :param sigma: I don't know
    :return: edge image
    """
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged  = cv.Canny(image, lower, upper)

    return edged

def verticalLine(M, L=60, H=120):
    """
    :param M: A binary matrix consisting of 0 and 1.
    :return: Pixel coordinates with vertical line
    """
    Gx = [[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]
    Gy = [[1, 2, 1],[0, 0, 0],[-1, -2, -1]]

    Filtered_X = convolve2d(M, Gx, mode='same')
    Filtered_Y = convolve2d(M, Gy, mode='same')

    orientation = np.arctan2(Filtered_X, Filtered_Y)
    orientation = orientation * 180/math.pi
    [x_neg, y_neg] = np.where(orientation < 0)
    orientation[x_neg, y_neg] += 360
    [x_one, y_one] = np.where((orientation > L) & (orientation < H))
    #[_, y_two] = np.where((orientation > L+180) & (orientation < H+180))
    y_one = np.subtract(y_one,1)
    del orientation
    orientation = np.zeros(M.shape)
    orientation[x_one, y_one] = 1
    #orientation[x_two, y_two] = 1
    return orientation
'''
    # Direction and orientations
    orientation = np.zeros(M.shape)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ori = math.atan2(Filtered_Y[i,j],Filtered_X[i,j])
            ori = ori * 180 / math.pi
            if (ori >= H*(-1) and ori <= L*(-1)) or (ori >= L and ori <= H):
                orientation[i,j] = 1
            else:
                orientation[i,j] = 0

    orientation2 = np.arctan2(Filtered_X, Filtered_Y)
    sum(sum(orientation == orientation))
'''

def block(mat,c):
    """

    :param mat: 2D array
    :param c: column index
    :return: t, END
    """

    count = 0;
    MAX = 0;
    S = mat.shape[1]
    #value = mat[:,c]

    #if c > 0:
    #    value_L = mat[:, c-1]
    #else:
    #    value_L = np.zeros(S)
    #if c < S-1:
    #    value_R = mat[:, c+1]
    #else:
    #    value_R = np.zeros(S)

    value = mat[:,max(0,c-1):min(S,c+2)]
    value = value.max(axis = 1)
    END = 0
    #END_REV = 0
    buffer = 0
    for i in range(S):
        if value[i] == 1:
            count += 1
            J = i
        elif buffer < 5:
            buffer += 1
        else:
            if count > MAX:
                MAX = count
                END = J
            count = 0
            buffer = 0
    if count > MAX:
        MAX = count
        END = J
    '''
    for i in range(S):
        if value[i,:].max() == 1:
            count += 1
            J = i
        elif buffer < 5:
            buffer += 1
        else:
            if count > MAX:
                MAX = count
                END = J
            count = 0
            buffer = 0

    if count > MAX:
        MAX = count
    t = MAX
    '''
    '''
    t0 = time.time()
    count_rev = 0
    MAX_REV = 0
    buffer = 0
    for i in range(S-1, -1, -1):
        if value[i,:].max() == 1:
            count_rev += 1
            J = i
        elif buffer < 5:
            buffer += 1
        else:
            if count_rev > MAX_REV:
                MAX_REV = count_rev
                END_REV = J
            count_rev = 0
            buffer = 0

    if count_rev > MAX_REV:
        MAX_REV = count_rev
        END_REV = J
    print(time.time()-t0)
    test1 = abs(c - END)
    test2 = abs(c - END_REV)
    
    if test1 > test2:
        t = MAX
        END = END
    else:
        t = MAX_REV
        END = END_REV
    '''
    t = MAX
    if END < c:
        END = END - t + 1
    return t, END

def Canny(img, threshold):

    imgsize = img.shape
    nrow = imgsize[0]
    ncol = imgsize[1]

    # Magic numbers
    PercentOfPixelsNotEdges = .7 # Used for selecting thresholds
    ThresholdRatio = .4          # Low thresh is this freaction of the high.
    thresh = threshold

    # Calculate gradients using a derivative of Gaussian
    dx, dy = smoothGradient(img)

    # Calculate Magnitude of Gradient
    magGrad = hypot(dx, dy)

    # Normalize for threshold selection
    magmax = magGrad.max()
    if magmax > 0:
        magGrad = magGrad / magmax

    # Calculate directions/orientation
    arah = np.zeros(imgsize)
    vert = np.zeros(imgsize)
    arah2 = np.zeros(imgsize)
    for i in range(nrow):
        for j in range(ncol):
            ori = math.atan2(dy[i,j],dx[i,j])
            ori = ori * 180 / math.pi
            if ori < 0:
                ori += 360
            arah[i,j] = ori

    for i in range(nrow):
        for j in range(ncol):
            if (arah[i,j] >= 0 and arah[i,j] < 22.5) or (arah[i,j] >=157.5 and arah[i,j] < 202.5) or (arah[i,j] >= 337.5 and arah[i,j] <=360):
                arah2[i,j] = 0
            elif (arah[i,j] >= 22.5 and arah[i,j] < 67.5) or (arah[i,j] >= 202.5 and arah[i,j] < 247.5):
                arah2[i,j] = 45
            elif (arah[i,j] >=67.5 and arah[i,j] < 112.5) or (arah[i,j] >= 247.5 and arah[i,j] < 292.5):
                arah2[i,j] = 90
            elif (arah[i,j] >= 112.5 and arah[i,j] < 157.5) or (arah[i,j] >= 292.5 and arah[i,j] < 337.5):
                arah2[i,j] = 135

    BW = np.zeros(imgsize)
    for i in range(1,nrow-1,1):
        for j in range(1,ncol-1,1):
            if arah2[i,j] == 0:
                BW[i,j] = magGrad[i,j] == max(magGrad[i,j], magGrad[i,j+1], magGrad[i,j-1])
            elif arah2[i,j] == 45:
                BW[i,j] = magGrad[i,j] == max(magGrad[i,j], magGrad[i+1,j-1], magGrad[i-1,j+1])
            elif arah2[i,j] == 90:
                BW[i,j] = magGrad[i,j] == max(magGrad[i,j], magGrad[i+1,j], magGrad[i-1,j])
            elif arah2[i,j] == 135:
                BW[i,j] = magGrad[i,j] == max(magGrad[i,j], magGrad[i+1, j+1], magGrad[i-1,j-1])

    BW = np.multiply(BW, magGrad)

    # Hysteresis Thresholding
    T_Low = thresh * ThresholdRatio * BW.max()
    T_High = thresh * BW.max()
    T_res = np.zeros(imgsize)

    for i in range(nrow):
        for j in range(ncol):
            if BW[i,j] < T_Low:
                T_res[i,j] = 0
            elif BW[i,j] > T_High:
                T_res[i,j] = 1
            elif BW[i+1,j]>T_High or BW[i-1,j]>T_High or BW[i,j+1]>T_High or BW[i,j-1]>T_High or BW[i-1, j-1]>T_High or BW[i-1, j+1]>T_High or BW[i+1, j+1]>T_High or BW[i+1, j-1]>T_High:
                T_res[i,j] = 1

    return T_res



def hypot(a, b):
    if type(a) != np.ndarray:
        a = np.array(a)
    if type(b) != np.ndarray:
        b = np.array(b)
    if a.shape != b.shape:
        raise ValueError('Two arrays should have same dimension!')
    res = a**2 + b**2
    res = res ** 0.5
    return res

def smoothGradient(I, sigma=math.sqrt(2)):
    """

    :param I: Image object
    :param sigma: Standard deviation of the filter, specified as a numeric scalar. Default = sqrt(2)
    :return: dx, dy
    """
    # Determine filter length
    filterExtent = math.ceil(4 * sigma)
    x = list(range(-1*filterExtent, filterExtent+1, 1))

    # Create 1-D Gaussian Kernel
    c = 1/(math.sqrt(2*math.pi)*sigma)
    gaussKernel = [c * math.exp(-(i**2)/(2*sigma**2)) for i in x]

    # Normalize to ensure kernel sums to one
    gaussKernel = [i/sum(gaussKernel) for i in gaussKernel]

    # Create 1-D Derivative of Gauss Kernel
    derivGaussKernel = simple_gradient(gaussKernel)

    # Normalize to ensure kernel sums to zero
    negVals = derivGaussKernel < 0
    posVals = derivGaussKernel > 0
    derivGaussKernel[posVals] = derivGaussKernel[posVals]/sum(derivGaussKernel[posVals])
    derivGaussKernel[negVals] = derivGaussKernel[negVals]/abs(sum(derivGaussKernel[negVals]))

    gaussKernel = np.array([gaussKernel])
    derivGaussKernel = np.array([derivGaussKernel])
    # Compute smoothed numerical gradient of image I along x (horizontal)
    # deriction. GX corresponds to dG/dx, where G is the Gaussian Smoothed
    # version of image I.
    GX = scipy.ndimage.convolve(I, np.transpose(gaussKernel), mode='nearest')
    GX = scipy.ndimage.convolve(GX, derivGaussKernel, mode='nearest')

    # Compute smoothed numerical gradient of image I along y (vertical)
    # direction. GY corresponds to dG/dy, where G is the Gaussian Smoothed
    # version of image I.
    GY = scipy.ndimage.convolve(I, gaussKernel, mode='nearest')
    GY = scipy.ndimage.convolve(GY, np.transpose(derivGaussKernel), mode='nearest')

    return GX, GY

def simple_gradient(f):
    rowflag = False
    ndim = 1
    indx = [len(f), 1]
    loc = [[i for i in range(indx[0])],[1]]
    siz = indx

    # first dimension
    g = np.zeros(siz[0])
    h = loc[0]
    n = siz[0]

    # take forward differences on left and right edges
    if n > 1:
        g[0] = (f[1] - f[0]) / (h[1] - h[0])
        g[n-1] = (f[n-1] - f[n-2])/(h[n-1] - h[n-2])

    if n > 2:
        g[1:n-1] = [(f[i]-f[j])/(h[k]-h[l]) for i,j,k,l in zip(range(2,n), range(0,n-2), range(2,n), range(0,n-2))]

    return g


def integral_image(image):
    return np.cumsum(np.cumsum(image, axis=0), axis=1)

def integral_image_torch(image):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    image = torch.tensor(image, dtype=torch.float32).to(device)
    return torch.cumsum(torch.cumsum(image, dim=0), dim=1).cpu().numpy()

def box_filter(integral_img, x, y, w, h):
    x1, y1 = x - w // 2, y - h // 2
    x2, y2 = x + w // 2, y + h // 2
    return (integral_img[y2, x2] - integral_img[y1-1, x2] - 
            integral_img[y2, x1-1] + integral_img[y1-1, x1-1])

"""def box_filter_vpi(integral_img, x, y, w, h):
    x1, y1 = x - w // 2, y - h // 2
    x2, y2 = x + w // 2, y + h // 2
    with vpi.Backend.CUDA:
        integral_img_vpi = vpi.asimage(integral_img)
        result = vpi.Image(1, 1, vpi.Format.F32)
        vpi.Subtract(integral_img_vpi[y2, x2], integral_img_vpi[y1-1, x2], result)
        vpi.Subtract(result, integral_img_vpi[y2, x1-1], result)
        vpi.Add(result, integral_img_vpi[y1-1, x1-1], result)
        return result.cpu().numpy()[0, 0]"""
    
def low_complexity_corner_detector(image, kernel_size=9, window_size=9, k=0.04, threshold=0.01):
    # 1. Calculate the integral image of the original image I
    #im_gray = np.apply_along_axis(lambda x : x.dot([0.07, 0.72, 0.21]), 2, image)
    #print("ENTERING LOCOCO")
    #print(image.shape)
    int_image = integral_image(image)
    # 2. Approximate the Gaussian derivative kernel by the Box kernel
    # and compute gx and gy using the integral image
    half_kernel = kernel_size // 2
    gx = np.zeros_like(image, dtype=float)
    gy = np.zeros_like(image, dtype=float)
    
    for y in range(half_kernel, image.shape[0] - half_kernel):
        for x in range(half_kernel, image.shape[1] - half_kernel):
            gx[y, x] = (box_filter(int_image, x + half_kernel//2, y, half_kernel, kernel_size) - 
                        box_filter(int_image, x - half_kernel//2, y, half_kernel, kernel_size)) / kernel_size**2
            gy[y, x] = (box_filter(int_image, x, y + half_kernel//2, kernel_size, half_kernel) - 
                        box_filter(int_image, x, y - half_kernel//2, kernel_size, half_kernel)) / kernel_size**2
    
    # 3. Create the integral image of gx^2, gy^2 and gx*gy
    gx2 = integral_image(gx**2)
    gy2 = integral_image(gy**2)
    gxy = integral_image(gx*gy)
    
    # 4. Evaluate R (Harris corner response)
    R = np.zeros_like(image, dtype=float)

    #print("image shape:", np.shape(image), "\ngx shape:", np.shape(gx), "\ngy shape:", np.shape(gy), "\nR shape:", np.shape(R), "\n")
    half_window = window_size // 2
    
    for y in range(half_window+1, image.shape[0] - half_window-1):
        for x in range(half_window, image.shape[1] - half_window-1):
            Gxx = box_filter(gx2, x, y, window_size, window_size)
            Gyy = box_filter(gy2, x, y, window_size, window_size)
            Gxy = box_filter(gxy, x, y, window_size, window_size)
            
            det = Gxx * Gyy - Gxy**2
            trace = Gxx + Gyy
            R[y, x] = det - k * trace**2
    #print("R matrix")
    #print(R)
    # 5. Efficient non-maximum suppression
    corners = []
    for y in range(half_window+1, image.shape[0] - half_window-1):
        for x in range(half_window+1, image.shape[1] - half_window-1):
            if y == x:
                continue
            #print("R matrix", "x:", x, "y:", y)
            if R[y, x] > threshold and R[y, x] == np.max(R[y-1:y+2, x-1:x+2]):
                # TODO figure out WHY THIS HAPPENS
                """if y > R.shape[1] or x > R.shape[0]:
                    print(f"OOB: corner at x={x}, y={y}")
                    continue"""
                corners.append((x, y))

    #print("R range:", np.min(R), np.max(R))
    #print("LOCOCO DONE")
    return corners, gx, gy, R


def low_complexity_corner_detector_torch(image, kernel_size=9, window_size=9, k=0.04, threshold=0.01):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    #print("device:", device)
    
    image = torch.tensor(image, dtype=torch.float32).to(device)
    int_image = torch.cumsum(torch.cumsum(image, dim=0), dim=1)
    
    half_kernel = kernel_size // 2
    gx = torch.zeros_like(image, dtype=torch.float32).to(device)
    gy = torch.zeros_like(image, dtype=torch.float32).to(device)
    
    for y in range(half_kernel, image.shape[0] - half_kernel):
        for x in range(half_kernel, image.shape[1] - half_kernel):
            gx[y, x] = (box_filter(int_image, x + half_kernel//2, y, half_kernel, kernel_size) - 
                        box_filter(int_image, x - half_kernel//2, y, half_kernel, kernel_size)) / kernel_size**2
            gy[y, x] = (box_filter(int_image, x, y + half_kernel//2, kernel_size, half_kernel) - 
                        box_filter(int_image, x, y - half_kernel//2, kernel_size, half_kernel)) / kernel_size**2
    
    gx2 = torch.cumsum(torch.cumsum(gx**2, dim=0), dim=1)
    gy2 = torch.cumsum(torch.cumsum(gy**2, dim=0), dim=1)
    gxy = torch.cumsum(torch.cumsum(gx*gy, dim=0), dim=1)
    
    R = torch.zeros_like(image, dtype=torch.float32).to(device)
    half_window = window_size // 2
    
    for y in range(half_window+1, image.shape[0] - half_window-1):
        for x in range(half_window, image.shape[1] - half_window-1):
            Gxx = box_filter(gx2, x, y, window_size, window_size)
            Gyy = box_filter(gy2, x, y, window_size, window_size)
            Gxy = box_filter(gxy, x, y, window_size, window_size)
            
            det = Gxx * Gyy - Gxy**2
            trace = Gxx + Gyy
            R[y, x] = det - k * trace**2
    
    corners = []
    for y in range(half_window+1, image.shape[0] - half_window-1):
        for x in range(half_window+1, image.shape[1] - half_window-1):
            if R[y, x] > threshold and R[y, x] == torch.max(R[y-1:y+2, x-1:x+2]):
                corners.append((x, y))
    
    return corners, gx.cpu().numpy(), gy.cpu().numpy(), R.cpu().numpy()

"""def low_complexity_corner_detector_vpi(image, kernel_size=9, window_size=9, k=0.04, threshold=0.01):
    input_image = vpi.asimage(np.asarray(image))
    with vpi.Backend.CUDA:
        int_image = vpi.Image.integral(input_image)
        
        half_kernel = kernel_size // 2
        gx = np.zeros_like(image, dtype=float)
        gy = np.zeros_like(image, dtype=float)
        
        for y in range(half_kernel, image.shape[0] - half_kernel):
            for x in range(half_kernel, image.shape[1] - half_kernel):
                gx[y, x] = (box_filter_vpi(int_image.cpu().numpy(), x + half_kernel//2, y, half_kernel, kernel_size) - 
                            box_filter_vpi(int_image.cpu().numpy(), x - half_kernel//2, y, half_kernel, kernel_size)) / kernel_size**2
                gy[y, x] = (box_filter_vpi(int_image.cpu().numpy(), x, y + half_kernel//2, kernel_size, half_kernel) - 
                            box_filter_vpi(int_image.cpu().numpy(), x, y - half_kernel//2, kernel_size, half_kernel)) / kernel_size**2
        
        gx2 = vpi.Image.integral(vpi.asimage(gx**2))
        gy2 = vpi.Image.integral(vpi.asimage(gy**2))
        gxy = vpi.Image.integral(vpi.asimage(gx*gy))
        
        R = np.zeros_like(image, dtype=float)
        half_window = window_size // 2
        
        for y in range(half_window+1, image.shape[0] - half_window-1):
            for x in range(half_window, image.shape[1] - half_window-1):
                Gxx = box_filter_vpi(gx2.cpu().numpy(), x, y, window_size, window_size)
                Gyy = box_filter_vpi(gy2.cpu().numpy(), x, y, window_size, window_size)
                Gxy = box_filter_vpi(gxy.cpu().numpy(), x, y, window_size, window_size)
                
                det = Gxx * Gyy - Gxy**2
                trace = Gxx + Gyy
                R[y, x] = det - k * trace**2
        
        corners = []
        for y in range(half_window+1, image.shape[0] - half_window-1):
            for x in range(half_window+1, image.shape[1] - half_window-1):
                if R[y, x] > threshold and R[y, x] == np.max(R[y-1:y+2, x-1:x+2]):
                    corners.append((x, y))
        return corners, gx, gy, R"""


def harris_corner_detector (image, blockSize=9, kSize=9, k=0.04, threshold_multiplier=0.01):
    image_in = np.float32(image)
    dst = cv.cornerHarris(image_in, blockSize=blockSize, ksize=kSize, k=k)
    #print("DST SHAPE")
    #print(np.shape(dst))
    t_val = threshold_multiplier * dst.max()
    corners = np.where(dst > t_val)
    #print("CORNERS SHAPE")
    #print(np.shape(corners))
    return list(zip(corners[1], corners[0]))

"""def harris_corner_detector_cuda (image, gradient_size, block_Size, strength, sensitivity, min_nms_distance):
    input = vpi.asimage(np.asarray(image))
    with vpi.Backend.CUDA:
        output = vpi.Image.harriscorners(input, gradient_size, block_Size, strength, sensitivity, min_nms_distance)
        return output"""
