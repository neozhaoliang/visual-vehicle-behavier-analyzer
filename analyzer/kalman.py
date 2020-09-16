from typing import NamedTuple
import numpy as np
from utils import check_shape, column_vector


# t: timestamp
# xhat_, Phat_: predict state/covariance
# xhat, Phat: updated state/covariance
filter_state_field_types = [("t", int), ("xhat_", np.array), ("Phat_", np.array),
                            ("xhat", np.array), ("Phat", np.array)]

FilterState = NamedTuple("FilterState", filter_state_field_types)


class _BaseKalman(object):

    """
    Base class for Kalman filtering.
    """

    def __init__(self, dim_x, dim_z, x0=None, P0=None,
                 Q=None, R=None, timestamp=None):
        """
        dim_x: dimension of the state vector.
        dim_z: dimension of the measurement vector.
        x0: initial state vector.
        P0: initial covariance matrix.
        Q: process noise covariance matrix.
        R: observe noise covariance matrix.
        t: current timestamp.
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.set_state(x0)
        self.set_state_covariance(P0)
        self.set_process_noise_covariance(Q)
        self.set_measurement_noise_covariance(R)
        self.t = timestamp

        self._I = np.eye(self.dim_x)
        self.filter_history = []
        self.smooth_history = []
        self.x_smooth = None
        self.P_smooth = None

    def set_state(self, x):
        """
        Set initial state vector.
        """
        if x is not None:
            self.x = column_vector(x, self.dim_x)
        else:
            self.x = None

    def set_state_covariance(self, P):
        """
        Set initial state covariance matrix.
        """
        if P is not None:
            self.P = check_shape(P, (self.dim_x, self.dim_x), "initial estimation covariance matrix")
        else:
            self.P = None

    def set_process_noise_covariance(self, Q):
        """
        Set initial process noise covariance matrix.
        """
        if Q is not None:
            self.Q = check_shape(Q, (self.dim_x, self.dim_x), "process nosie covariance matrix")
        else:
            self.Q = None

    def set_measurement_noise_covariance(self, R):
        """
        Set initial measurement noise covariance matrix.
        """
        if R is not None:
            self.R = check_shape(R, (self.dim_z, self.dim_z), "measurements noise covariance matrix")
        else:
            self.R = None

    def start(self):
        assert self.x is not None, "unknown initial state"
        assert self.P is not None, "unknown initial state covariance matrix"
        assert self.Q is not None, "unknown initial process noise covariance matrix"
        assert self.R is not None, "unknown initial measurement noise covariance matrix"
        assert self.t is not None, "unknown initial timestamp"
        self.filter_history.append(FilterState(self.t, self.x, self.P, self.x, self.P))
        return self

    def state_to_xyz(self, x):
        """
        Convert state `x` to the vehicle's (x, y, z) world position.
        """
        world_x, world_y = x[:2, 0]
        return np.array([world_x, world_y, 0])

    def F(self, x, dt):
        """
        Process evolution function.
        """
        raise NotImplementedError

    def FJacobian(self, x, dt):
        """
        Jacobian matrix of `F`.
        """
        raise NotImplementedError

    def H(self, x, camera):
        """
        Measurement function.
        """
        world_point = self.state_to_xyz(x)
        image_pixel, = camera.world_to_image([world_point])
        return image_pixel

    def HJacobian(self, x, camera):
        """
        Jacobian matrix of `H`.
        """
        world_point = self.state_to_xyz(x)
        M = camera.get_jacobi_matrix(world_point)
        jH = np.zeros((self.dim_z, self.dim_x))
        jH[:2, :2] = M[:2, :2]
        return jH

    def predict(self, t1):
        """
        x_{k|k-1} = F(x_{k-1|k-1}, dt),
        P_{k|k-1} = jF * P_{k-1|k-1} * jF^T + Q.

        where jF is the Jacobian matrix of F at x_{k-1|k-1}.
        """
        dt = (t1 - self.t) / 1000  # time increment
        jF = self.FJacobian(self.x, dt)  # Jacobian matrix
        self.P = jF @ self.P @ jF.T + self.Q * dt  # updated covariance matrix
        self.x = self.F(self.x, dt)  # updated state vector
        self.t = t1
        return self.x.copy(), self.P.copy()

    def update(self, t1, pixel, camera):
        if pixel is not None:
            H = self.HJacobian(self.x, camera)
            PHT = self.P @ H.T
            S = H @ PHT + self.R
            S_inv = np.linalg.inv(S)
            K = PHT @ S_inv
            hx = self.H(self.x, camera)
            y = np.subtract(column_vector(pixel), column_vector(hx))
            self.x += K @ y
            I_KH = self._I - K @ H
            KRK = K @ self.R @ K.T
            self.P = I_KH @ self.P @ I_KH.T + KRK

        return self.x.copy(), self.P.copy()

    def predict_and_update(self, t1, pixel, camera):
        """
        Combine predict and update in one step and save filtered history.
        """
        xhat_, Phat_ = self.predict(t1)
        xhat, Phat = self.update(t1, pixel, camera)
        self.filter_history.append(FilterState(t1, xhat_, Phat_, xhat, Phat))

    def rts_interval_smoothing(self):
        """
        Rauch-Tung-Striebel fixed interval smoothing.

        xN: next state in the history sequence. Initially it's the last state in filter history.
        xP: previous state in the history sequence.

        Assume `x_smooth`, `P_smooth` hold the smoothed state at time xN, we now compute the
        smoothed state at time xP.
        """
        xN = self.filter_history[-1]
        self.x_smooth, self.P_smooth = xN.xhat, xN.Phat
        self.smooth_history.append((xN.t, self.x_smooth, self.P_smooth))

        for xP in self.filter_history[-1::-1]:
            dt = (xN.t - xP.t) / 1000.0
            A = self.FJacobian(xP.xhat, dt)
            K = xP.Phat @ A.T @ np.linalg.inv(xN.Phat_)
            self.x_smooth = xP.xhat + K @ (self.x_smooth - xN.xhat_)
            self.P_smooth = xP.Phat + K @ (self.P_smooth - xN.Phat_) @ K.T
            self.smooth_history.append((xP.t, self.x_smooth, self.P_smooth))
            xN = xP
        # reverse the list since we smoothed the states backwards.
        self.smooth_history = self.smooth_history[::-1]


class BicycleModel(_BaseKalman):

    """
    Vehicle bicycle model.
    """

    LR = 2.0

    def __init__(self, x0=None, P0=None,
                 Q=None, R=None, timestamp=None):
        super().__init__(5, 2, x0, P0, Q, R, timestamp)

    def F(self, x, dt):
        _, _, v, phi, beta = x[:, 0]  # note x is a column vector
        dphi = v / self.LR * np.sin(beta)
        dx = column_vector([
            v*np.cos(phi + beta),
            v*np.sin(phi + beta),
            0,
            dphi,
            0])
        return x + dx * dt

    def FJacobian(self, x, dt):
        lr = self.LR
        _, _, v, phi, beta = x[:, 0]
        cb, sb = np.cos(beta), np.sin(beta)
        alpha = phi + beta
        ca, sa = np.cos(alpha), np.sin(alpha)
        A = np.array([[0, 0,    ca, -v*sa,   -v*sa],
                      [0, 0,    sa,  v*ca,    v*ca],
                      [0, 0,     0,     0,       0],
                      [0, 0, sb/lr,     0, v*cb/lr],
                      [0, 0,     0,     0,       0]])
        return self._I + A * dt


class CVModel(_BaseKalman):

    """
    Vehicle constant velocity model.
    """

    def __init__(self, x0=None, P0=None,
                 Q=None, R=None, timestamp=None):
        super().__init__(4, 2, x0, P0, Q, R, timestamp)

    def F(self, x, dt):
        A = self.FJacobian(x, dt)
        return np.dot(A, x)

    def FJacobian(self, x, dt):
        return np.array([[1, 0, dt,  0],
                         [0, 1,  0, dt],
                         [0, 0,  1,  0],
                         [0, 0,  0,  1]])

    @staticmethod
    def cv2bicycle(state):
        x, y, vx, vy = state[:, 0]
        v = np.sqrt(vx*vx + vy*vy)
        phi = np.arctan2(vy, vx)
        return [x, y, v, phi, 0]
