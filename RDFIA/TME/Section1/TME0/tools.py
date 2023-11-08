import os
#from tqdm import tqdm
from glob import glob

import matplotlib
import matplotlib.pyplot as mplt
import matplotlib.pyplot as plt
import numpy as np


# Very simplified tqdm
class tqdm:
    def __init__(self, data):
        self.data = data
        self.iter = iter(data)

    def __iter__(self):
        for i, x in enumerate(self.data):
            print("{}/{}".format(i+1, len(self.data)))
            yield x

    def __len__(self):
        return len(self.data)


# size: mask size (will be square)
# sigma: sigma gaussian parameter
def gaussian_mask(size=16, sigma=0.5):
    sigma *= size
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

# s: stride
def dense_sampling(im, s=8):
    w, h = im.shape
    x = np.arange(0, w, s)
    y = np.arange(0, h, s)
    return x, y

def auto_padding(im, k=16, s=8):
    w, h = im.shape
    x = np.arange(0, w, s)
    y = np.arange(0, h, s)
    # last region could be smaller
    last_r = im[x[-1]:x[-1]+k, y[-1]:y[-1]+k]
    if last_r.shape == (k, k):
        return im
    dif_w = k - last_r.shape[0]
    dif_h = k - last_r.shape[1]
    n_im = np.zeros((w+dif_w, h+dif_h))
    id_w = dif_w // 2
    id_h = dif_h // 2
    n_im[id_w:id_w+w, id_h:id_h+h] = im
    return n_im

def conv_separable(im, h_x, h_y, pad=1):
    h_x = h_x.reshape(1,3)
    h_y = h_y.reshape(3,1)

    im_w, im_h = im.shape
    hx_w, hx_h = h_x.shape
    hy_w, hy_h = h_y.shape

    h_x_t = h_x.transpose()
    h_y_t = h_y.transpose()

    if hx_w != 1:
        raise ValueError()
    if hx_h % 2 != 1:
        raise ValueError()
    if hy_h != 1:
        raise ValueError()
    if hy_w % 2 != 1:
        raise ValueError()
    if hx_h != hy_w:
        raise ValueError()

    dim_p = (hx_h - 1) // 2 # dim padding

    # Toeplitz matrices
    t_x = np.zeros((im_h+2*dim_p, im_h))
    t_y = np.zeros((im_w, im_w+2*dim_p))
    for i in range(im_h):
        t_x[i:i+hx_h,[i]] = h_x_t
    for i in range(im_w):
        t_y[[i],i:i+hy_w] = h_y_t

    # padding on colomn
    im_yp = np.zeros((im_w, im_h+2*dim_p))
    im_yp[:,dim_p:im_h+dim_p] = im

    if pad == 1:
        # copy padding
        for i in range(dim_p):
            im_yp[:,i] = im[:,0]
            im_yp[:,im_h+dim_p+i] = im[:,im_h-1]

    # conv of filtre h_x
    g = np.dot(im_yp, t_x)

    # padding on line
    g_xp = np.zeros((im_w+2*dim_p, im_h))
    g_xp[dim_p:im_w+dim_p,:] = g

    if pad == 1:
        # copy padding
        for i in range(dim_p):
            g_xp[i,:] = g[0,:]
            g_xp[im_w+dim_p+i,:] = g[im_w-1,:]

    g = np.dot(t_y, g_xp)
    return g

# g_x: image gradient x-axis (w,h)
# g_y: image gradient y-axis (w,h)
# g_m: image gradient module (w,h)
# b: orientation bins
# g_o: image gradient orientation (w,h)
def compute_grad_ori(g_x, g_y, g_m, b=8):
    ori = np.zeros((b, 2))
    for i in range(b):
        ori[i,0] = np.cos(2 * np.pi * i / b)
        ori[i,1] = np.sin(2 * np.pi * i / b)
    w, h = g_m.shape
    # TODO: algebraic form
    g_o = np.zeros((w, h))
    for i in range(w):
        for j in range(h):
            if g_m[i,j] > 0:
                v = np.array([g_y[i,j], -g_x[i,j]])
                v = v / np.linalg.norm(v, ord=2)
                prod = np.dot(ori,v)
                g_o[i,j] = np.argmax(prod)
            else:
                g_o[i,j] = -1
    g_o = g_o.astype(int)
    return g_o

def read_grayscale(path):
    img = mplt.imread(path)
    if len(img.shape) > 2:
        if img.shape[2] == 1:
            img = img[:, :, 0]
        else:
            img = 0.2 * img[:, :, 0] + 0.7 * img[:, :, 1] + 0.1 * img[:, :, 2]
    return img

def orientation_colors():
    w = 100
    h = 100
    b = 8
    pix = 3
    ori = np.zeros((b,2))
    for i in range(b):
        ori[i,0] = np.cos(2 * np.pi * i / b)
        ori[i,1] = np.sin(2 * np.pi * i / b)
    g_o = np.zeros((w,h))
    for i in range(w):
        for j in range(h):
            v = np.array([j-h/2, -i+w/2])
            #v = np.array([i-w/2, -j+h/2])
            prod = np.dot(ori,v)
            g_o[i,j] = np.argmax(prod)
    g_o[w//2-pix:w//2+pix, h//2-pix:h//2+pix] = -1
    return g_o

def display_sift_region(im, compute_grad_mod_ori, compute_sift_region, x=200, y=78, k=16, gausm=True):
    g_m, g_o = compute_grad_mod_ori(im)
    if gausm:
        m = gaussian_mask()
    else:
        m = None
    g_m_r = g_m[x:x+k, y:y+k]
    g_o_r = g_o[x:x+k, y:y+k]
    sift = compute_sift_region(g_m_r, g_o_r, m)

    b = 8
    cmap = cmap_discretize('jet', b+1)

    mplt.figure(figsize=(10,6))
    ax = mplt.subplot(2,3,1)
    mplt.imshow(im, cmap='gray', vmin=0, vmax=255)
    rect = matplotlib.patches.Rectangle((x,y),k,k,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    mplt.title("Image")

    mplt.subplot(2,3,2)
    mplt.imshow(im[y:y+k, x:x+k], cmap='gray', vmin=0, vmax=255)
    mplt.title("Patch")

    mplt.subplot(2,3,4)
    mplt.imshow(g_m[y:y+k, x:x+k], cmap='jet')
    mplt.colorbar()
    mplt.title("Gradient module")

    mplt.subplot(2,3,6)
    ori_map = orientation_colors()
    mplt.imshow(ori_map.T, cmap=cmap, vmin=-1, vmax=b-1)
    mplt.colorbar()
    mplt.title("Orientations")

    mplt.subplot(2,3,5)
    mplt.imshow(g_o[y:y+k, x:x+k], cmap=cmap, vmin=-1, vmax=b-1)
    mplt.colorbar()
    mplt.title("Gradient orientation")

    mplt.subplot(2,3,3)
    mplt.plot(sift)
    mplt.title("SIFT")

    mplt.show()

    return sift

def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet.
        N: number of colors.
    """
    if type(cmap) == str:
        cmap = matplotlib.cm.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki, key in enumerate(('red','green','blue')):
        cdict[key] = [(indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki]) for i in range(N+1)]
    # Return colormap object.
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

def toy_im():
    im = np.zeros((256,256))
    im[:100,:100] = 255
    im[:100,105:] = 255
    im[105:,105:] = 255
    im[105:,:100] = 255
    return im

def marche_im():
    im = np.zeros((256,256))
    im[:130,:130] = 255
    return im

def listdir(path):
    return [os.path.basename(p) for p in glob(os.path.join(path, '*'))]

def load_dataset(dir_sc, images_per_class=None):
    inames = []
    ilabls = []
    cnames = sorted(listdir(dir_sc))
    for ilabl, cl in enumerate(cnames):
        dir_cl = os.path.join(dir_sc, cl)
        for iname in listdir(dir_cl)[:images_per_class]:
            inames.append(os.path.join(cl, iname))
            ilabls.append(ilabl)
    ilabls = np.array(ilabls)
    return inames, ilabls, cnames

def compute_sift(dir_imgs, dir_sift, iname, compute_sift_image):
    ipath = os.path.join(dir_imgs, iname)
    im = read_grayscale(ipath)
    if im is None:
        print('Failed:', ipath)
        return None
    sift = compute_sift_image(im)
    sift = (sift * 255).astype('uint8')  # For faster K-means
    spath = os.path.join(dir_sift, iname)[:-4] # remove .jpg
    sdir = os.path.dirname(spath)
    os.makedirs(sdir, exist_ok=True)
    np.save(spath, sift)
    return sift

def compute_load_sift_dataset(dir_imgs, dir_sift, inames, compute_sift_image):
    sift = []
    print("Computing or loading SIFTs")
    for iname in tqdm(inames):
        spath = os.path.join(dir_sift, iname)[:-4]+'.npy'
        if os.path.isfile(spath):
            sift.append(load_sift(dir_sift, iname))
        else:
            sift.append(compute_sift(dir_imgs, dir_sift, iname, compute_sift_image))
    return sift

def compute_sift_dataset(dir_imgs, dir_sift, inames, compute_sift_image):
    sift = []
    for iname in tqdm(inames):
        sift.append(compute_sift(dir_imgs, dir_sift, iname, compute_sift_image))
    return sift

def load_sift(dir_sift, iname):
    spath = os.path.join(dir_sift, iname)[:-4]+'.npy'
    sift = np.load(spath)
    return sift

def load_sift_dataset(dir_sift, inames):
    sift = []
    for iname in tqdm(inames):
        sift.append(load_sift(dir_sift, iname))
    return sift

def compute_split(length, seed=1337, pc=0.80):
    train_ids = np.random.RandomState(seed=seed).choice(
        length,
        size=int(length * pc),
        replace=False)
    test_ids = np.array(list(set(np.arange(length)) - set(train_ids)))
    return train_ids, test_ids


def compute_or_load_vdict(dir_sc, dir_sift, inames, compute_sift_image,  path_vdict, compute_vdict):
    print("Computing or loading visual dict")
    if os.path.isfile(path_vdict):
        return np.load(open(path_vdict, "rb"))
    else:
        sifts_list_by_image = compute_load_sift_dataset(dir_sc, dir_sift, inames, compute_sift_image)

        vdict = compute_vdict(sifts_list_by_image)
        np.save(open(path_vdict, "wb"), vdict)
        return vdict

def compute_regions(im, k=16, s=8):
    x, y = dense_sampling(im) # before padding
    im = auto_padding(im)
    images = np.zeros((x.shape[0], y.shape[0], k, k))
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            images[i,j] = im[x[i]:x[i]+k, y[j]:y[j]+k]
    return images

def get_regions_and_sifts(dir_sc, inames, image_sifts):
    vdpaths = [os.path.join(dir_sc, iname) for iname in inames]

    regions = []
    for path in vdpaths:
        im = read_grayscale(path)
        regions.append(compute_regions(im))

    k = regions[0].shape[-1]
    n_reg = np.array([r.shape[0]*r.shape[1] for r in regions])
    cs_reg = np.cumsum(n_reg)

    regions = [r.reshape(-1, k, k) for r in regions]
    regions = np.concatenate(regions, axis=0)

    sift = [s.reshape(-1, image_sifts[0].shape[-1]) for s in image_sifts]
    sift = np.concatenate(sift, axis=0)
    return regions, sift

def display_images(images):
    n_images,w,h = images.shape
    n = int(np.ceil(np.sqrt(n_images)))
    im = np.zeros((n*w, n*h))
    for k in range(n_images):
        i = k % n
        j = k // n
        im[i*w:i*w+w, j*h:j*h+h] = images[k]

    mplt.figure(figsize=(0.7*n,0.7*n))
    mplt.gray()
    mplt.imshow(im)
    mplt.axis('off')
    mplt.show()


def display_vdregions(images, colors=None):
    n_images,w,h = images.shape
    n = int(np.ceil(np.sqrt(n_images)))
    im = np.zeros((n*w, n*h))
    for k in range(n_images):
        i = k % n
        j = k // n
        im[i*w:i*w+w, j*h:j*h+h] = images[k]

    fig, ax = plt.subplots(1)
    plt.axis('off')
    ax.imshow(im, cmap='gray', vmin=0, vmax=255)

    for k in range(n_images):
        i = k % n * w
        j = k // n * h
        rect = matplotlib.patches.Rectangle(
            (i,j),w-1,h-1,
            linewidth=4,
            edgecolor=colors[k],
            facecolor='none')
        ax.add_patch(rect)

    plt.show()

def display_vdregions_image(im, vdict, sift, feats, colors=None, vdregions=None):
    from sklearn.metrics.pairwise import euclidean_distances
    if colors is None:
        colors = ['tab:blue',
                  'tab:orange',
                  'tab:green',
                  'tab:red',
                  'tab:purple',
                  'tab:brown',
                  'tab:pink',
                  'tab:olive',
                  'tab:cyan',
                  'tab:gray']

    plt.figure()
    plt.bar(range(feats.shape[0]), feats)
    plt.show()

    ids = feats.argsort()[-9:][::-1]
    if vdregions is not None:
        vdregions = vdregions[ids]

    fig, ax = plt.subplots(1)
    fig.set_size_inches(10, 10)
    ax.imshow(im, cmap='gray', vmin=0, vmax=255)

    if vdregions is None:
        vdregions = [None] * 9
    w = h = 16
    for i in range(sift.shape[0]):
        for j in range(sift.shape[1]):
            dist = euclidean_distances(vdict, sift[i,j].reshape(1,-1))
            word_id = int(dist.argmin(axis=0)[0])

            nonzero = np.nonzero(ids == word_id)[0]
            if nonzero.size == 0:
                continue

            id_ = nonzero[0]
            if vdregions[id_] is None:
                vdregions[id_] = im[i*8:i*8+16,j*8:j*8+16]
                # if vdregions[id_].shape[0] != 16: # hack if problem, TODO find reason
                #     vdregions[id_] = np.zeros(16,16)

            rect = matplotlib.patches.Rectangle(
                (j*8,i*8),w-1,h-1,
                linewidth=2,
                edgecolor=colors[id_],
                facecolor='none')
            ax.add_patch(rect)

    plt.axis('off')
    plt.show()

    for i in range(9):
        if vdregions[i] is None:
            vdregions[i] = np.zeros((16,16))
        if vdregions[i].shape != (16, 16):
            shape = vdregions[i].shape
            vdregions[i] = np.pad(vdregions[i], ((0, 16 - shape[0]), (0, 16 - shape[1])))
    if type(vdregions) == list:
        vdregions = np.stack(vdregions)
    print(vdregions.shape)
    display_vdregions(vdregions, colors=colors)
