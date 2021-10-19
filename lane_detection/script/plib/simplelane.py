import cv2
import numpy as np
import math

"""
old_lslope = 0
old_rslope = 0
old_left_x_start = 0
old_left_x_end = 0
old_right_x_start = 0
old_right_x_end = 0

    global old_lslope
    global old_rslope
    global old_left_x_start
    global old_left_x_end
    global old_right_x_start
    global old_right_x_end

"""

# lane
# https://medium.com/@mrhwick/simple-lane-detection-with-opencv-bfeb6ae54ec0
#
class simplelane():
    def __init__(self, cfg_img, roi):
        self.roi = roi

    def getRoiColorImage(self, img, vertices):
        # Define a blank matrix that matches the image height/width.
        mask = np.zeros_like(img)
        # Retrieve the number of color channels of the image.
        channel_count = img.shape[2]
        # Create a match color with the same color channel counts.
        match_mask_color = (255,) * channel_count
        # Fill inside the polygon
        cv2.fillPoly(mask, np.int32([vertices]), match_mask_color)
        # Returning the image only where mask pixels match
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def getRoiGrayImage(self, img, vertices):
        mask = np.zeros_like(img)
        match_mask_color = 255
        cv2.fillPoly(mask, np.int32([vertices]), match_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def drawLines(self, img, lines, color=[0, 0, 255], thickness=3):
        # If there are no lines to draw, exit.
        if lines is None:
            return

        # Make a copy of the original image.
        img = np.copy(img)
        # Create a blank image that matches the original in size.
        line_image = np.zeros([ img.shape[0], img.shape[1], 3], dtype=np.uint8)
        # Loop over all lines and draw them on the blank image.
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
        # Merge the image with the lines onto the original.
        img = cv2.addWeighted(img, 0.8, line_image, 1.0, 0.0)
        # Return the modified image.
        return img


    def image_pipeline(self, bev_img):
        # Convert to grayscale here.
        img = cv2.cvtColor(bev_img, cv2.COLOR_RGB2GRAY)

        # crop image if whe need a color image 
        # use alternate metodo in class
        img = self.getRoiGrayImage(img, self.roi)

        # Call Canny edge detection
        aux_img = cv2.Canny(img, 100, 200)

        # get lines
        lines = cv2.HoughLinesP(
            aux_img,
            rho=6,
            theta=np.pi / 60,
            threshold=160,
            lines=np.array([]),
            minLineLength=40,
            maxLineGap=25
        )
        #print(lines)
        line_image = self.drawLines(bev_img, lines)

        # bof: Creating a Single Linear Representation of each Line Group
        left_line_x = []
        left_line_y = []
        right_line_x = []
        right_line_y = []
        center_width = bev_img.shape[1] / 2

        # select left and right groups
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1)
                if math.fabs(slope) < 1: # 0.5 <-- Only consider extreme slope
                    continue
                if (x1+x2) / 2 > center_width:
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])
                else: 
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])

        # top and bottom of the image
        min_y = 0
        max_y = bev_img.shape[0]

        # try to fit left lane marker
        left_lane = False
        if len(left_line_x):
            poly_left = np.poly1d(np.polyfit(
                left_line_y,
                left_line_x,
                deg=1
            ))

            left_x_start = int(poly_left(max_y))
            left_x_end = int(poly_left(min_y))

            left_lane = True


        # try to fit right lane marker
        right_lane = False
        if len(right_line_x):
            poly_right = np.poly1d(np.polyfit(
                right_line_y,
                right_line_x,
                deg=1
            ))

            right_x_start = int(poly_right(max_y))
            right_x_end = int(poly_right(min_y))

            right_lane = True

        # eof: Creating a Single Linear Representation of each Line Group

        # check if both lines are acceptable
        if left_lane and right_lane:
            #print("left: ", left_x_start, left_x_start)
            #print("right: ", right_x_start, right_x_end)
            temp = math.fabs(((left_x_start - right_x_start) + (left_x_end - right_x_end))/2)
            #print ("temp: ", temp)
            if(temp) < 100:
                left_lane = False

            else:
                lslope = left_x_start - left_x_end 
                rslope = right_x_start - right_x_end 
                #print(lslope, rslope) 
            if(lslope - rslope) > 100:
                if math.fabs(lslope) > 100:
                    left_lane = False
                if math.fabs(rslope) > 100:
                    right_lane = False

        """
        if old_lslope == 0:
            old_lslope = lslope

        if old_rslope == 0:
            old_lslope = rslope

        old_left_x_start = left_x_start
        old_left_x_end = left_x_end
        old_right_x_start = right_x_start
        old_right_x_end = right_x_end
        """


        # bof: display lines
        if left_lane and right_lane:
            line_image = self.drawLines(bev_img,
                [[
                    [left_x_start, max_y, left_x_end, min_y],
                    [right_x_start, max_y, right_x_end, min_y],
                ]],
                thickness=5,
            )

        elif left_lane:
            line_image = self.drawLines(bev_img,
                [[
                    [left_x_start, max_y, left_x_end, min_y],
                ]],
                thickness=5,
            )

        elif right_lane:
            line_image = self.drawLines(bev_img,
                [[
                    [right_x_start, max_y, right_x_end, min_y],
                ]],
                thickness=5,
            )
        # eof: display lines

        return line_image