import cv2
import skimage.measure as measure
import numpy as np


def remove_glare(image):
    image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (12, 12), 0)
    _, thresh_img = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
    thresh_img = cv2.erode(thresh_img, None, iterations=2)
    thresh_img = cv2.dilate(thresh_img, None, iterations=4)
    # perform a connected component analysis on the threshold image,
    # then initialize a mask to store only the "large" components
    labels = measure.label(thresh_img, connectivity=2, background=0)
    mask = np.zeros(thresh_img.shape, dtype="uint8")
    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels
        label_mask = np.zeros(thresh_img.shape, dtype="uint8")
        label_mask[labels == label] = 255
        num_pixels = cv2.countNonZero(label_mask)
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if num_pixels > 300:
            mask = cv2.add(mask, label_mask)
    return cv2.inpaint(image, mask, 1, cv2.INPAINT_NS)

if __name__ == "__main__":
    img_path = "C:\\Users\\Jesus Bueno\\Desktop\\BeCode\\faktion_anomaly_detection\\assets\\data\\normal_dice\\0\\0.jpg"
    img = cv2.imread(img_path)
    cv2.imwrite("out.jpg", remove_glare(img))