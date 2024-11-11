import cv2
import numpy as np
import random
import typing
import sys


COORD3D = typing.Tuple[float, float, float]
COORD2D = typing.Tuple[float, float]

class TrackModel:
    """
    Simple linear tracking model to track a 3d sphere on an image
    Uses spehere apparent size for pseudo 3d tracking
    """
    def __init__(self, x: float, y: float, r: float, alpha: float = 0.5):
        self.x = x
        self.y = y
        self.z = 1
        self.r0 = r # radious of circle
        self.dx = 0
        self.dy = 0
        self.dz = 0
        self.alpha = alpha
        self.path = [(self.x, self.y, self.z)]
        self.path3d = [self.project3d()]

    def predict(self):
        self.x += self.dx
        self.y += self.dy
        self.z += self.dz

    def update(self, xt: float, yt: float, rt: float):
        # recall the previous tracked point
        x_prev, y_prev, z_prev = self.path[-1]
        a = self.alpha
        zt = self.r0/rt

        # update to a new point based on weighting of (1-a) for observation and (a) for prediction
        self.x = a * self.x + (1. - a) * xt
        self.y = a * self.y + (1. - a) * yt
        self.z = a * self.z + (1. - a) * zt
        self.dx = a * self.dx + (1 - a) * (self.x - x_prev)
        self.dy = a * self.dy + (1 - a) * (self.y - y_prev)
        self.dz = a * self.dz + (1 - a) * (self.z - z_prev)

        self.path.append((self.x, self.y, self.z))        
        self.path3d.append(self.project3d())        

        # accumulate a track path but truncate it to 1500 pts
        if len(self.path) > 1500: self.path = self.path[1:]
        if len(self.path3d) > 1500: self.path3d = self.path3d[1:]

    def center(self) -> COORD2D:
        return (self.x, self.y)

    def r(self) -> float:
        return max(0, 1/self.z * self.r0)

    def project3d(self) -> COORD3D:
        return (self.x * self.z, self.y * self.z, self.z)


def plot(x: typing.List[float], y: typing.List[float], figure: np.array):
    """
    Plots x vs y as a curve on a given image with adaptive windowing
    """
    figure[:] = 255
    width, height = figure.shape[:2]
    top = max(y)
    left = min(x)
    bottom = min(y)
    right = max(x)
    w, h = max(right - left, 0.01), max(top - bottom, 0.01)
    pts = [[int(((xi - left)/w * 0.9 + 0.05)*width), int(((1 - (yi - bottom)/h) * 0.9 + 0.05)*height)] for xi, yi in zip(x, y)]
    cv2.polylines(figure, [np.array(pts)], False, (255, 0, 0))
    

def coord_to_img(p: COORD2D, display: np.array) -> typing.Tuple[int, int]:
    """
    Translates a coordinate in [-.5,.5] x [-.5,.5] to image pixel coordinates
    """
    h, w, c = display.shape
    x, y = p
    u = int((x + 0.5)*w)
    v = int((y + 0.5)*h)
    return u, v    


def create_ball(num_points: int = 1000) -> typing.List[COORD3D]:
    """
    Creates a 3d ball of 3d points on the surface of a unit sphere at the origin
    """
    ball = []
    for i in range(num_points):
        # a random vector in the unit cube
        x = random.random() - 0.5
        y = random.random() - 0.5
        z = random.random() - 0.5
        mag = np.sqrt(x**2 + y**2 + z**2)
        # normalize to 1 so it is on the surface of the ball
        ball.append((x/mag, y/mag, z/mag))
    return ball


def project_ball(ball: typing.List[COORD3D], pos: COORD3D) -> typing.List[COORD2D]:
    """
    Projects a ball of 3d points onto a 2d plane for an observer at the origin. Plane z = 1
    """
    result = []
    x0, y0, z0 = pos
    for particle in ball:
        x, y, z = particle
        x += x0; y += y0; z += z0
        result.append((x / z, y / z))
    return result


def draw_ball(ball: typing.List[COORD3D], pos: COORD3D, display: np.array) -> typing.List[COORD2D]:
    """
    Renders a 3d ball of points on an image as white dots
    """
    h, w, c = display.shape
    projected2d = project_ball(ball, pos)
    for particle in projected2d:
        u, v = coord_to_img(particle, display)
        if 0 <= u < w and 0 <= v < h:
            display[v, u, :] = 255
    return projected2d


def centroid(pts: typing.List[COORD2D]) -> COORD2D:
    N = float(len(pts))
    X, Y = 0, 0
    for pt in pts:
        x, y = pt
        X += x; Y += y
    return (X/N, Y/N)


def max_dist2(pts: typing.List[COORD2D], pt: COORD2D) -> float:
    """
    Finds the maximum distance between a point and a series of points
    """
    x0, y0 = pt
    maxd2 = 0.
    for pt in pts:
        x, y = pt
        maxd2 = max(maxd2, (x - x0)**2 + (y - y0)**2)
    return maxd2


def add_random_noise(p: COORD2D) -> COORD2D:
    """
    Adds uniform square randon noise to a given 2d point
    """
    ex = (random.random() - 0.5) * 0.03
    ey = (random.random() - 0.5) * 0.03
    return (p[0] + ex, p[1] + ey)


def main() -> int:
    my_ball = create_ball()

    disp = np.zeros((800, 800, 3), dtype=np.uint8)
    fig = np.zeros((400, 400, 3), dtype=np.uint8)

    x, y, z = (0, 0, 10)
    t = 0
    track: TrackModel = None

    while True:
        disp[:] = 0
        # compute the next true 3d position of the ball
        x = np.cos(t / 100. * 2 * np.pi) * 5
        y = np.sin(t / 1000. * 2 * np.pi) * 3
        z = np.sin(t / 100. * 2 * np.pi) * 5 + 13

        # get noisy detection of the ball on the image
        detection = draw_ball(my_ball, (x, y, z), disp)
        detection_ctr = centroid(detection)
        # noise
        detection_ctr = add_random_noise(detection_ctr)
        r = np.sqrt(max_dist2(detection, detection_ctr))

        if track is None:
            # first initialize the track
            track = TrackModel(detection_ctr[0], detection_ctr[1], r, 0.65)
        else:
            track.predict()
            xt, yt = detection_ctr
            track.update(xt, yt, r)

        # draw the observation and the current track on screen as circles
        cv2.circle(disp, coord_to_img(detection_ctr, disp), 2, (255, 255, 0), 4)
        cv2.circle(disp, coord_to_img(detection_ctr, disp), int(r * disp.shape[0]), (0, 255, 255), 1)

        cv2.circle(disp, coord_to_img(track.center(), disp), 2, (0, 0, 255), 4)
        cv2.circle(disp, coord_to_img(track.center(), disp), int(track.r() * disp.shape[0]), (0, 0, 255), 1)

        x3d = [p[0] for p in track.path3d]
        y3d = [-p[1] for p in track.path3d]
        z3d = [p[2] for p in track.path3d]

        plot(x3d, y3d, fig)
        cv2.imshow('xy', fig)
        plot(x3d, z3d, fig)
        cv2.imshow('xz', fig)

        cv2.imshow('pseudo3d tracker (press Q to quit)', disp)


        k = cv2.waitKey(13)
        k = chr(k % 256)
        if k in ['q', 'Q']: break
        
        t += 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
