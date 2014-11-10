import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import transformations

path = "/Users/scope/slam_data/rgbd_dataset_freiburg1_rpy/"
rgb_path = path + "rgb.txt"
depth_path = path + "depth.txt"
ground_truth_path = path + "groundtruth.txt"

rgbs = pd.read_csv(rgb_path, sep=" ", skiprows=[0, 1, 2], header=None, names=["timestamp", "path"])
depths = pd.read_csv(depth_path, sep=" ", skiprows=[0, 1, 2], header=None, names=["timestamp", "path"])

ground_truth = pd.read_csv(ground_truth_path, sep=" ", skiprows=[0, 1, 2],
        header=None, names=["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])

r_times = rgbs["timestamp"].values
d_times = depths["timestamp"].values
r_paths = rgbs["path"]
d_paths = depths["path"]

def get_rgb(idx):
    val = cv2.imread(path+r_paths[idx])
    return val

def get_rgb_time(idx):
    return r_times[idx]

def get_depth(idx):
    val = cv2.imread(path+d_paths[idx], cv2.IMREAD_ANYDEPTH)
    return val

def get_depth_near(time):
    inx = np.argmin(np.abs(d_times-time))
    return get_depth(inx)

def get_transform_between_times(t1, t2):
    # TODO FIXME don't ignore rotation
    g_times = ground_truth['timestamp']
    t1_idx = np.argmin(np.abs(g_times - t1))
    t2_idx = np.argmin(np.abs(g_times - t2))
    dx = ground_truth["tx"][t2_idx] - ground_truth["tx"][t1_idx]
    dy = ground_truth["ty"][t2_idx] - ground_truth["ty"][t1_idx]
    dz = ground_truth["tz"][t2_idx] - ground_truth["tz"][t1_idx]
    trans = np.eye(3,4)
    trans[0, 3] = dx
    trans[0, 3] = dy
    trans[0, 3] = dz

    return trans

def pixel_to_3d(point, depth):
    fx = 525.0
    fy = 525.0
    cx = 319.5
    cy = 239.5
    factor = 5000.0

    z = depth[point[1], point[0]] / factor
    if z <= 1e-10:
        #error
        return None

    x = (point[0] - cx) * z / fx
    y = (point[1] - cy) * z / fy
    return np.array((x,y,z))

def get_matching_camera_space(frame1, frame2):
    #HYPERPARAM
    #corner_threshold = 0.035
    corner_threshold = 0.025
    ratio_threshold = 1.0


    im1, d_im1 = frame1
    im2, d_im2 = frame2

    descriptor_name = "SIFT"
    detector = cv2.FeatureDetector_create(descriptor_name)
    extractor = cv2.DescriptorExtractor_create(descriptor_name)
    matcher = cv2.BFMatcher()

    im1_bw = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2_bw = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)


    kp1 = detector.detect(im1_bw)
    kp2 = detector.detect(im2_bw)

    dc, des1 = extractor.compute(im1_bw,kp1)
    dc, des2 = extractor.compute(im2_bw,kp2)


    matches = matcher.knnMatch(des1,des2,k=2)

    good_matches = []
    for m,n in matches:
        # make sure the distance to the closest match is sufficiently better than the second closest
        if (m.distance < ratio_threshold*n.distance and
            kp1[m.queryIdx].response > corner_threshold and
            kp2[m.trainIdx].response > corner_threshold):
            good_matches.append((m.queryIdx, m.trainIdx))

    im = np.array(np.hstack((im1,im2)))


    pts1 = np.zeros((len(good_matches),2))
    pts2 = np.zeros((len(good_matches),2))

    for idx in range(len(good_matches)):
        match = good_matches[idx]
        pts1[idx,:] = kp1[match[0]].pt
        pts2[idx,:] = kp2[match[1]].pt
    # plot the points
    #for i in range(pts1.shape[0]):
    for i in range(3):
        cv2.circle(im,(int(pts1[i,0]),int(pts1[i,1])),2,(255,0,0),2)
        cv2.circle(im,(int(pts2[i,0]+im1.shape[1]),int(pts2[i,1])),2,(255,0,0),2)
        cv2.line(im,(int(pts1[i,0]),int(pts1[i,1])),(int(pts2[i,0]+im1.shape[1]),int(pts2[i,1])),(0,255,0))

    cv2.imshow("MYWIN",im)

    cv2.imshow("Asdf", d_im1)

    camera_pts_1 = [pixel_to_3d(kp1[query_idx].pt, d_im1) for query_idx, _ in good_matches]
    camera_pts_2 = [pixel_to_3d(kp2[train_idx].pt, d_im2) for _, train_idx in good_matches]

    camera_space_pts_1 = [pt1 for pt1, pt2 in zip(camera_pts_1, camera_pts_2) if pt1 is not None and pt2 is not None]
    camera_space_pts_2 = [pt2 for pt1, pt2 in zip(camera_pts_1, camera_pts_2) if pt1 is not None and pt2 is not None]


    camera_space_pts_1 = np.vstack(camera_space_pts_1)
    camera_space_pts_2 = np.vstack(camera_space_pts_2)
    return camera_space_pts_1, camera_space_pts_2;

def cv_transform(pts1, pts2):
    ransac_threshold = 1
    confidence = 0.99
    res, trans, err = cv2.estimateAffine3D(pts1, pts2, ransacThreshold=ransac_threshold, confidence=confidence)
    return trans

def transformations_transform(pts1, pts2):
    print pts1.T.shape
    print pts2.T.shape
    #trans = transformations.affine_matrix_from_points(pts1.T, pts2.T)
    trans = transformations.superimposition_matrix(pts1.T, pts2.T)
    print trans
    return trans[0:3, :]

def evaluate_trans(pts1, pts2, trans):
    est = pts1.dot(trans)
    est = est[:, 0:3]
    error = pts2 - est
    plt.figure()
    plt.plot(error, "o")
    plt.legend(["x", "y", "z"])

    print np.mean(error * error)

idx_1 = 0
idx_2 = 1

im1 = get_rgb(idx_1)
im2 = get_rgb(idx_2)

t1 = get_rgb_time(idx_1)
t2 = get_rgb_time(idx_2)

d_im1 = get_depth_near(idx_1)
d_im2 = get_depth_near(idx_2)

pts1, pts2 = get_matching_camera_space((im1, d_im1), (im2, d_im2))
trans = cv_transform(pts1, pts2)
trans = transformations_transform(pts1, pts2)

good_trans = get_transform_between_times(t1, t2)
zero_trans = np.eye(3,4)

evaluate_trans(pts1, pts2, trans, name="trans")
evaluate_trans(pts1, pts2, zero_trans, name="zero")
plt.show()


cv2.waitKey(0)
