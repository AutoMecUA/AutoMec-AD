#!/usr/bin/env python3
import pathlib
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

def cut_sky(img):
    height, width = img.shape[:2]
    start_row = int(0.2*height)
    end_row = int(height)
    start_col = int(0)
    end_col = int(width)
    nimg = img[start_row:end_row,start_col:end_col]
    return nimg



def color_threshold (image):

    # For discarding colors
    # How much can pixels deviate from black/white color
    #deviation = 40  # re-adjustable
    # minim, maxim = deviation, 255 - deviation
    # # For each of the r, g and b scales
    # for i in range(3):
    #     # If < maxim -> to_zero
    #     _, image[:][:][i] = cv2.threshold(image[:][:][i], maxim, 255, cv2.THRESH_TOZERO)

    # RGB to Grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    th = 100  # reasonably working (best value?)

    #height, width = image.shape[:2]
    # Binary image
    _, bin_img = cv2.threshold(gray_image, th, 255, cv2.THRESH_BINARY)
    #start_row = int(0)
    #end_row = int(width)
    #start_col = int(0.2*height)
    #end_col = int(height)
    #bin_img = bin_img[start_row:end_row,start_col:end_col]

    return bin_img

def perspective_aux(height, angle, roadwidth):
    # Calculate the values to more accurately transform the image in the function perspective_transform. Heavily inspired on Bruno Monteiro's MATLAB code
    fovV = 43 * math.pi/180
    fovH = 57 * math.pi/180
    Xmin = math.tan(angle - fovV) * height
    Xmax = math.tan(angle + fovV) * height
    Dmin = math.sqrt(Xmin ** 2 + height ** 2)
    Dmax = math.sqrt(Xmax ** 2 + height ** 2)
    Lmin = 2 * (math.tan(fovH/2) * Dmin)
    Lmax = 2 * (math.tan(fovH/2) * Dmax)
    BotRatio = roadwidth / Lmin
    TopRatio = roadwidth / Lmax

    XminO = Xmin

    # If the bottom ratio is larger then one, the full width of the lane is hidden. With several interactions
    # we can find the first pixels where the full width is shown.
    while BotRatio > 1:
        inc = 0.01
        counter = 1
        fovV = fovV - inc
        Xmin = math.tan(angle - fovV) * height
        Dmin = math.sqrt(Xmin ** 2 + height ** 2)
        Lmin = 2 * (math.tan(fovH / 2) * Dmin)
        BotRatio = roadwidth / Lmin
        counter = counter + 1
    SideRatio = (Xmax-Xmin)/(Xmax-XminO)
    print(BotRatio, TopRatio, SideRatio, counter)
    return BotRatio, TopRatio, SideRatio





def perspective_transform(img, BR, TR, SR):
    # Transform the image into a top-down perspective using the previously mentioned ratios
    imshape = img.shape
    if SR != 1:
        src = np.float32([[((0.5 + TR/2) * imshape[1], 0), ((0.5 + BR/2) * imshape[1], SR*imshape[0]),
                           ((0.5 - BR/2) * imshape[1], SR*imshape[0]), ((0.5 - TR/2) * imshape[1], 0)]])
    else:
        src = np.float32([[((0.5 + TR / 2) * imshape[1], 0), ((0.5 + BR / 2) * imshape[1], imshape[0]),
                           ((0.5 - BR / 2) * imshape[1], imshape[0]), ((0.5 - TR / 2) * imshape[1], 0)]])

    dst = np.float32([[0.75 * img.shape[1], 0], [0.75 * img.shape[1], img.shape[0]],
                      [0.25 * img.shape[1], img.shape[0]], [0.25 * img.shape[1], 0]])

    M = cv2.getPerspectiveTransform(src, dst)

    # Generate the contrary transformation, to easily re-warp the image
    Minv = cv2.getPerspectiveTransform(dst, src)

    img_size = (imshape[1], imshape[0])
    perspective_img = cv2.warpPerspective(img, M, img_size, flags = cv2.INTER_LINEAR)
    inv_perspective_img = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)
    return perspective_img, inv_perspective_img


def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist


def find_lane_pixels(binary_warped):
    """
    only for the first frame of the video this function is run.

    input - 'binary_warped' perspective transformed thresholded image
    output - x and y coordinates of the lane pixels

    step 1: Get the histogram of warped input image
    step 2: find peaks in the histogram that serves as midpoint for our first window
    step 3: choose hyperparameter for windows
    step 4: Get x, y coordinates of all the non zero pixels in the image
    step 5: for each window in number of windows get indices of all non zero pixels falling in that window
    step 6: Get the x, y coordinates based off of these indices
    """
    # Take a histogram of the bottom half of the image
    histogram = get_hist(binary_warped)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9

    # Set the width of the windows +/- margin
    margin = 100

    # Set minimum number of pixels found to recenter window
    minpix = 200

    # Set height of windows - based on nwindows above and image shape
    window_height = int(binary_warped.shape[0] // nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):

        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height

        # the four boundaries of the window ###
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]

        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # if the no of pixels in the current window > minpix then update window center
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def fit_poly(leftx, lefty, rightx, righty):
    """
    Given x and y coordinates of lane pixels for 2nd order polynomial through them

    here the function is of y and not x that is

    x = f(y) = Ay**2 + By + C

    returns coefficients A, B, C for each lane (left lane and right lane)
    """
    # Fit a second order polynomial to each with np.polyfit()
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit


def search_around_poly(binary_warped, left_fit, right_fit):
    """
    This function is extension to function find_lane_pixels().

    From second frame onwards of the video this function will be run.

    the idea is that we dont have to re-run window search for each and every frame.

    once we know where the lanes are, we can make educated guess about the position of lanes in the consecutive frame,

    because lane lines are continuous and dont change much from frame to frame(unless a very abruspt sharp turn).

    This function takes in the fitted polynomial from previous frame defines a margin and looks for non zero pixels
    in that range only. it greatly increases the speed of the detection.
    """
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # we have left fitted polynomial (left_fit) and right fitted polynomial (right_fit) from previous frame,
    # using these polynomial and y coordinates of non zero pixels from warped image,
    # we calculate corrsponding x coordinate and check if lies within margin, if it does then
    # then we count that pixel as being one from the lane lines.
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin))).nonzero()[0]
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin))).nonzero()[0]

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx = fit_poly(leftx, lefty, rightx, righty)

    return leftx, lefty, rightx, righty


def measure_curvature_real(left_fit_cr, right_fit_cr, img_shape):
    '''
    Calculates the curvature of polynomial functions in meters.
    and returns the position of vehical relative to the center of the lane
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = img_shape[0]

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    # It was meentioned in one the course note that camera was mounted in the middle of the car,
    # so the postiion of the car is middle of the image, the we calculate the middle of lane using
    # two fitted polynomials
    car_pos = img_shape[1] / 2
    left_lane_bottom_x = left_fit_cr[0] * img_shape[0] ** 2 + left_fit_cr[1] * img_shape[0] + left_fit_cr[2]
    right_lane_bottom_x = right_fit_cr[0] * img_shape[0] ** 2 + right_fit_cr[1] * img_shape[0] + right_fit_cr[2]
    lane_center_position = ((right_lane_bottom_x - left_lane_bottom_x) / 2) + left_lane_bottom_x
    car_center_offset = np.abs(car_pos - lane_center_position) * xm_per_pix

    return (left_curverad, right_curverad, car_center_offset)


def draw_lane(warped_img, undistorted_img, left_fit, right_fit, BR, TR, SR):
    """
    Given warped image and original undistorted original image this function
    draws final lane on the undistorted image
    """
    # Generate x and y values for plotting
    ploty = np.linspace(0, undistorted_img.shape[0] - 1, undistorted_img.shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    _ , newwarp = perspective_transform(color_warp, BR, TR, SR)
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted_img, 1, newwarp, 0.3, 0)

    return result


first_run = True
gleft_fit = gright_fit = None


# def Pipeline(img):
#     global first_run, gleft_fit, gright_fit
#
#     threshold_img, undistorted_img = img_threshold(img)
#
#     warped_img = perspective_transform(threshold_img, BR, TR)
#
#     #warped_img=img
#
#     if first_run:
#         leftx, lefty, rightx, righty = find_lane_pixels(warped_img)
#         left_fit, right_fit = fit_poly(leftx, lefty, rightx, righty)
#         gleft_fit = left_fit
#         gright_fit = right_fit
#         first_run = False
#     else:
#         leftx, lefty, rightx, righty = search_around_poly(warped_img, gleft_fit, gright_fit)
#         left_fit, right_fit = fit_poly(leftx, lefty, rightx, righty)
#         gleft_fit = left_fit
#         gright_fit = right_fit
#
#     # print(left_fitx, right_fitx)
#     measures = measure_curvature_real(left_fit, right_fit, img_shape=warped_img.shape)
#
#     final_img = draw_lane(warped_img, undistorted_img, left_fit, right_fit)
#
#     # writing lane curvature and vehical offset on the image
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     fontColor = (0, 0, 0)
#     fontSize = 1
#     cv2.putText(final_img, 'Lane Curvature: {:.0f} m'.format(np.mean([measures[0], measures[1]])),
#                 (500, 620), font, fontSize, fontColor, 2)
#     cv2.putText(final_img, 'Vehicle offset: {:.4f} m'.format(measures[2]), (500, 650), font, fontSize, fontColor, 2)
#
#     return final_img


def Image_GET(image):
    global cv_image
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    global see_image
    see_image=True

def video():
    # Set global parameters
    height = 0.59
    angle = 0.4
    roadwidth = 0.75
    # Calculate the ratio
    BR, TR, SR = perspective_aux(height, angle, roadwidth)

    s = str(pathlib.Path(__file__).parent.absolute())

    #Get gazebo files
    global cv_image
    rospy.init_node('Robot_Send', anonymous=True)
    rospy.Subscriber('/robot/camera/rgb/image_raw', Image, Image_GET)

    global see_image
    see_image = False
    while not rospy.is_shutdown():
        if see_image == False:
            continue
        img=cv_image
        cut_img = cut_sky(img)
        bin_img = color_threshold(cut_img)
        cv2.imshow('Binary', bin_img)


        perspective_img, _ = perspective_transform(bin_img, BR, TR, SR)

        cv2.imshow('perspective', perspective_img)

        leftx, lefty, rightx, righty = find_lane_pixels(perspective_img)
        left_fit, right_fit = fit_poly(leftx, lefty, rightx, righty)          #TODO if values very small = 0

        #print(left_fit, right_fit)
        measures = measure_curvature_real(left_fit, right_fit, img_shape=cut_img.shape)
        print(measures)
        final_img = draw_lane(perspective_img, cut_img, left_fit, right_fit, BR, TR, SR)

        cv2.imshow('Final', final_img)
        #plt.show()


        #cv2.waitKey(0)

        cv2.waitKey(1)
        rospy.Rate(1).sleep()

def image():
    # Set global parameters
    height = 0.59
    angle = 0.4
    roadwidth = 0.75
    # Calculate the ratio
    BR, TR, SR = perspective_aux(height, angle, roadwidth)

    s = str(pathlib.Path(__file__).parent.absolute())
    img = cv2.imread(s + '/lane_test/right_curve.png', cv2.IMREAD_COLOR)

    cut_img = cut_sky(img)

    bin_img = color_threshold(cut_img)
    cv2.imshow('Binary', bin_img)


    perspective_img, _ = perspective_transform(bin_img, BR, TR, SR)

    cv2.imshow('perspective', perspective_img)

    leftx, lefty, rightx, righty = find_lane_pixels(perspective_img)
    left_fit, right_fit = fit_poly(leftx, lefty, rightx, righty)          #TODO if values very small = 0


    measures = measure_curvature_real(left_fit, right_fit, img_shape=cut_img.shape)
    print(measures)
    final_img = draw_lane(perspective_img, cut_img, left_fit, right_fit, BR, TR, SR)

    cv2.imshow('Final', final_img)
    #plt.show()


    #cv2.waitKey(0)

    cv2.waitKey(0)






if __name__ == '__main__':
    image()
