import numpy as np
from math import exp, sin, cos

"""
    Markley, Landis & Crassidis, John. (2014). Fundamentals of Spacecraft Attitude Determination and Control.
"""


def q_to_a(q):
    """
        quaternion to rotation matrix  
        @param q: quaternion  
        @return A: rotation matrix 
    """

    # normalize quaternion 
    q = q * (1/np.linalg.norm(q))

    q_1 = q[0]
    q_2 = q[1]
    q_3 = q[2]
    q_4 = q[3]
    
    #----------------#
    # YOUR CODE HERE #
    #----------------#
    A = np.array([  [q_1**2 - q_2**2 - q_3**2 + q_4**2, 2*(q_1*q_2 + q_3*q_4), 2*(q_1*q_3 - q_2*q_4)],
                    [2*(q_2*q_1 - q_3*q_4), -q_1**2 + q_2**2 - q_3**2 + q_4**2, 2*(q_2*q_3 + q_1*q_4)],
                    [2*(q_3*q_1 + q_2*q_4), 2*(q_3*q_2 - q_1*q_4), -q_1**2 -q_2**2 + q_3**2 + q_4**2] ])

    return A


def euler_123_to_a(euler):
    """
        1-2-3 Euler angles to rotation matrix     
        @param euler: 1-2-3 euler angles    
        @return A: rotation matrix  
    """
    phi = euler[0]
    theta = euler[1]
    psi = euler[2]

    #----------------#
    # YOUR CODE HERE #
    #----------------#
    A_1 = np.array([    [1, 0, 0],
                        [0, cos(phi), sin(phi)],
                        [0, -sin(phi), cos(phi)] ])

    A_2 = np.array([    [cos(theta), 0, -sin(theta)],
                        [0, 1, 0],
                        [sin(theta), 0, cos(theta)] ])
    
    A_3 = np.array([    [cos(psi), sin(psi), 0],
                        [-sin(psi), cos(psi), 0],
                        [0, 0, 1] ])

    A = A_3 @ A_2 @ A_1

    return A


def a_dot(w,A):
    """
        derivative of a rotation matrix
        @param w: angular velocity
        @param A: a rotation matrix
        @return A_dot: derivate of the rotation matrix
    """

    w_skew = np.array([ [0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0] ]) # get skew-symmetric matrix from w

    # w represents angular velocity w_{B}^{BI}
    # A represents rotation matrix A_IB
    A_dot = A @ w_skew

    return A_dot


def euler_123_dot(w,euler):
    """
        derivative of 1-2-3 Euler angle sequence
        @param w: angular velocity
        @param euler: 1-2-3 euler angles
        @return euler_dot: derivate of the 1-2-3 euler angles 
    """
    
    theta = euler[1]
    psi = euler[2]

    # 1-2-3 Euler kinematics #
    # w represents angular velocity w_{B}^{BI}
    # euler angles prepresent rotation matrix A_BI
    euler_dot = (1/cos(theta)) * np.array([ [cos(psi), -sin(psi), 0], 
                                            [cos(theta)*sin(psi), cos(theta)*cos(psi), 0], 
                                            [-sin(theta)*cos(psi), sin(theta)*sin(psi), cos(theta)] ]) @ w

    return euler_dot 




def q_dot(w,q):
    """
        derivative of a quaternion
        @param w: angular velocity. w_B^BI
        @param q: a quaternion, representing rotation from inertial to B (A_BI)
        @return q_dot: derivate of the quaternion 
    """

    q_1 = q[0]
    q_2 = q[1]
    q_3 = q[2]
    q_4 = q[3]
    
    # quaternion kinematics #
    # w represents angular velocity w_{B}^{BI}
    # quaternion represent a rotation matrix A_BI
    q_dot = (1/2) * np.array([  [q_4, -q_3, q_2 ], 
                                [q_3, q_4, -q_1], 
                                [-q_2, q_1, q_4], 
                                [-q_1, -q_2, -q_3]  ]) @ w

    return q_dot




def rk4_kin(dt,w,x, flag):
    """
        integrate kinematics using fourth order Runge-Kutta method  
        @param dt: time step. seconds  
        @param w: angular velocity, rad/s  
        @param flag: 'rot' for rotation matrix. 'euler' for euler angles. 'quat' for quaternion.  
        @param x: previous attitude representation  
        @return y: current attitude representation
    """
    if flag == 'rot':
        k_1 = dt*a_dot(w, x)
        k_2 = dt*a_dot(w, x + 0.5*k_1)
        k_3 = dt*a_dot(w, x + 0.5*k_2)
        k_4 = dt*a_dot(w, x + k_3)
    elif flag == 'euler':
        k_1 = dt*euler_123_dot(w, x)
        k_2 = dt*euler_123_dot(w, x + 0.5*k_1)
        k_3 = dt*euler_123_dot(w, x + 0.5*k_2)
        k_4 = dt*euler_123_dot(w, x + k_3)
    elif flag == 'quat':
        k_1 = dt*q_dot(w, x)
        k_2 = dt*q_dot(w, x + 0.5*k_1)
        k_3 = dt*q_dot(w, x + 0.5*k_2)
        k_4 = dt*q_dot(w, x + k_3)
    else:
        print("wrong flag")


    y = x + 1/6.0 * (k_1 + 2*k_2 + 2*k_3 + k_4)

    return y




def circle_cross(q,p):
    """
        @param q: quaternion, rotation from A to B. fourth component scalar 
        @param p: quaternion, rotation from C to B. fourth component scalr
        @return circle_cross: q circile cross p = a quaternion rotation C to A
    """
    
    q_1_3 = q[0:3]
    q_4 = q[3]
    
    p_1_3 = p[0:3]
    p_4 = p[3]

    circle_cross_1_3 = q_4*p_1_3 + p_4*q_1_3 - np.cross(q_1_3, p_1_3)
    circle_crpss_4 = q_4*p_4 - np.dot(q_1_3, p_1_3)

    circle_cross = np.array([ circle_cross_1_3[0], circle_cross_1_3[1], circle_cross_1_3[2], circle_crpss_4 ])

    return circle_cross





def euler_from_quaternion(x, y, z, w):
    """
    https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians