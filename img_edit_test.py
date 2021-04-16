from scipy import misc
import imageio

# read image
img = imageio.imread("./images/experiment_20210210_1/2021_02_10_10_44_51multi_0CH0.tiff")
h,w = img.shape

# cut image in half
width_cutoff = w // 2
height_cutoff = h // 2
tl = img[:height_cutoff, :width_cutoff]
tr = img[:height_cutoff, width_cutoff:]
bl = img[height_cutoff:, :width_cutoff]
br = img[height_cutoff:, width_cutoff:]

# save each half
imageio.imsave("tl.png", tl)
imageio.imsave("tr.png", tr)
imageio.imsave("bl.png", bl)
imageio.imsave("br.png", br)
