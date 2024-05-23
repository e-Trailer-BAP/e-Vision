import numpy as np
import cv2
import os
import glob
# You should replace these 3 lines with the output in calibration step
DIM=(318, 179)
K=np.array([[156.6898311547844, 0.0, 165.65406423712332], [0.0, 156.02491268777231, 114.07749665062885], [0.0, 0.0, 1.0]])
D=np.array([[0.08339903277680968], [-0.8915924890188938], [2.5157017941353943], [-2.3250440015292693]])
def undistort(img_path):
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (638, 358), cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort(p)


# def undistort(img_path, balance=0.0, dim2=None, dim3=None):
#     img = cv2.imread(img_path)
#     dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
#     print(dim1[0])
#     assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
#     if not dim2:
#         dim2 = dim1
#     if not dim3:
#         dim3 = dim1
#     scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
#     scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
#     print(scaled_K)
#     # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
#     new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
#     map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
#     undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
#     cv2.imshow("undistorted", undistorted_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


undistort('data\distort.jpg')