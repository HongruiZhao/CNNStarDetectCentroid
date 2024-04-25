from cProfile import label
import numpy as np
from math import cos, sqrt, acos, radians, degrees, atan2, tan
import matplotlib.pyplot as plt
from requests import get


def get_errors(W, H, Res_W, Res_H, f_c, alpha_offb_seq):
    """
        get errors of star tracker test system.\\
        @param W: active panel width, mm.\\
        @param H: active panel height, mm.\\
        @param Res_W: width resolution, pixels.\\
        @param Res_H: height resolution, pixels.\\
        @param f_c: focal length of the collimating lens, mm.\\
        @param alpha_offb_seq: a sequence of off-boresight angles. deg.\\
        @return pos_max_error: star angular position max error (also single star roll or pitch attitude max error) at different off-boresight angles. arcsec.\\
        @return roll_max_error: ingle star roll max error at different off-boresight angles. arcsec.
    """
    delta_pixW = W / Res_W # pixel width, mm
    delta_pixH = H / Res_H # pixel height, mm
    a = sqrt( delta_pixW**2 + delta_pixH**2) / 2 

    position_max_error = [] 
    roll_max_error = [] 

    for alpha_offb in alpha_offb_seq:
        b = f_c * tan( radians(alpha_offb) )
        delta_pix_d2 = degrees(atan2(a+b, f_c)) - alpha_offb 
        position_max_error.append( 3600 * delta_pix_d2 ) # in arcsec

        delta_pixr_d2 = degrees( atan2( delta_pixW/2, b ) )
        roll_max_error.append( 3600 * delta_pixr_d2 ) # in arcsec

    return position_max_error, roll_max_error




if __name__ == '__main__':
    # # EliteDisplay E221li
    # W = 47.66 * 10 # active panel width, mm
    # H = 26.81 * 10 # active panel height, mm
    # Res_W = 1920 # width resolution, pixels
    # Res_H = 1080 # height resolution, pixels 
    # f_c = 1.1 * 1000 # focal length of the collimating lens, mm

    # # FEELWORLD FW568 V2
    # W = 147 # active panel width, mm
    # H = 87.2 # active panel height, mm
    # Res_W = 1920 # width resolution, pixels
    # Res_H = 1152 # height resolution, pixels 
    # f_c = 250 # focal length of the collimating lens, mm

    # FEELWORLD F5
    W = 110.69009321481776 # active panel width, mm
    H = 62.263177433334995 # active panel height, mm
    Res_W = 1920 # width resolution, pixels
    Res_H = 1080 # height resolution, pixels 
    f_c = 350 # focal length of the collimating lens, mm

    FOV_h = degrees( atan2( H/2, f_c) )
    FOV_w = degrees( atan2( W/2, f_c) )

    alpha_offb_seq = np.arange(0.05, FOV_h, 0.05) # off-boresight angle, 0~16 deg

    position_max_error, roll_max_error = get_errors(W, H, Res_W, Res_H, f_c, alpha_offb_seq)

    # University of Naples
    pos_ref_naples, roll_ref_naples = get_errors(0.641 * 1000, 0.401 * 1000, 2560, 1600, 1.3 * 1000, alpha_offb_seq)

    # JPL SFS
    pos_ref_jpl, roll_ref_jpl = get_errors(14.7 * 10, 7.4 * 10, 1920, 1080, 30 * 10, alpha_offb_seq)


    plt.figure(1)

    plt.subplot(211)
    plt.title('Half-cone FOV (height) = {:.2f} deg, FOV (width) = {:.2f} deg'.format(FOV_h, FOV_w), fontsize = 20, weight = 'bold')

    plt.plot(alpha_offb_seq, position_max_error, 'r', label = 'CUA')
    plt.plot(alpha_offb_seq, pos_ref_naples, 'c--', label = 'Naples')
    plt.plot(alpha_offb_seq, pos_ref_jpl, 'b-.', label = 'JPL')
    plt.xlabel('off-boresight angle (deg)', fontsize = 15)
    plt.ylabel('Pos / P or Y Max Error (arcsec)', fontsize = 15)
    plt.legend(loc='upper right', fontsize = 15)
    plt.grid(True)
    plt.subplot(212)

    plt.plot(alpha_offb_seq[20:-1], roll_max_error[20:-1], 'r', label = 'CUA')
    plt.plot(alpha_offb_seq[20:-1], roll_ref_naples[20:-1], 'c--', label = 'Naples')
    plt.plot(alpha_offb_seq[20:-1], roll_ref_jpl[20:-1], 'b-.', label = 'JPL')
    plt.xlabel('off-boresight angle (deg)', fontsize = 15)
    plt.ylabel('Roll Max Error (arcsec)', fontsize = 15)
    plt.legend(loc='upper right', fontsize = 15)
    plt.grid(True)

    plt.show()
