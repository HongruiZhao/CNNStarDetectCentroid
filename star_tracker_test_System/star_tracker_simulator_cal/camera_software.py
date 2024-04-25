import time
import numpy 
import json
from ImageConvert import *
import ArducamSDK
import os

class camera_software:
    """
        use to get images from cameras and save them into numpy array
    """
    absolute_path = os.path.dirname(__file__)
    config_file_name = os.path.join(absolute_path, "MT9V022_VGA.json")

    def __init__(self, int_time=100):
        # scan for cameras
        self.devices_num, self.index, serials = ArducamSDK.Py_ArduCam_scan()
        time.sleep(2) # need 2s delay between scan and open
        # handles for cameras
        self.handles = []
        # configuration for cameras
        self.cfg = []
        # other camera parameters
        self.Width = None
        self.Height = None
        self.color_mode = None

        # integration time in ms 
        self.int_time = int_time

        # store image numpy arraies, 3d
        self.images = None


    def get_value_for_integration_time(self):
        master_clock_period = (1 / (26.6E6))*1E3 # ms
        R0x04 =  640 # default decimal
        R0x05 = 94 # default decimal
        overhead = (R0x04 + R0x05 - 255) * master_clock_period
        row_time = (R0x04 + R0x05) * master_clock_period
        val_R0x0B = (self.int_time - overhead) / row_time

        return val_R0x0B

    def __del__(self):
        """
            close camera
        """
        for i in range( len(self.handles) ):
            handle = self.handles[i]
            error_code = ArducamSDK.Py_ArduCam_close(handle)
            if error_code == 0:
                print( "camera {} closed!".format(i) )


    def configBoard(self, handle,fileNodes):
        for i in range(0,len(fileNodes)):
            fileNode = fileNodes[i]
            buffs = []
            command = fileNode[0]
            value = fileNode[1]
            index = fileNode[2]
            buffsize = fileNode[3]
            for j in range(0,len(fileNode[4])):
                buffs.append(int(fileNode[4][j],16))
            ArducamSDK.Py_ArduCam_setboardConfig(handle,int(command,16),int(value,16),int(index,16),int(buffsize,16),buffs)


    def writeSensorRegs(self, handle,fileNodes):
        for i in range(0,len(fileNodes)):
            fileNode = fileNodes[i]      
            if fileNode[0] == "DELAY":
                time.sleep(float(fileNode[1])/1000)
                continue
            regAddr = int(fileNode[0],16)
            val = int(fileNode[1],16)
            ArducamSDK.Py_ArduCam_writeSensorReg(handle,regAddr,val)


    def camera_initFromFile(self, fialeName, index):
        """
            initialize camera and set configuration file
        """
        #load config file
        config = json.load(open(fialeName,"r"))

        camera_parameter = config["camera_parameter"]
        self.Width = int(camera_parameter["SIZE"][0])
        self.Height = int(camera_parameter["SIZE"][1])

        BitWidth = camera_parameter["BIT_WIDTH"]
        ByteLength = 1
        if BitWidth > 8 and BitWidth <= 16:
            ByteLength = 2
        FmtMode = int(camera_parameter["FORMAT"][0])

        self.color_mode = (int)(camera_parameter["FORMAT"][1])

        I2CMode = camera_parameter["I2C_MODE"]
        I2cAddr = int(camera_parameter["I2C_ADDR"],16)
        TransLvl = int(camera_parameter["TRANS_LVL"])

        self.cfg = {"u32CameraType":0x4D091031,
                "u32Width":self.Width,"u32Height":self.Height,
                "usbType":0,
                "u8PixelBytes":ByteLength,
                "u16Vid":0,
                "u32Size":0,
                "u8PixelBits":BitWidth,
                "u32I2cAddr":I2cAddr,
                "emI2cMode":I2CMode,
                "emImageFmtMode":FmtMode,
                "u32TransLvl":TransLvl }

        # ArducamSDK.
        ret,handle,rtn_cfg = ArducamSDK.Py_ArduCam_open(self.cfg,index)

        # if open successfully
        if ret == 0:
        
            #ArducamSDK.Py_ArduCam_writeReg_8_8(handle,0x46,3,0x00)
            usb_version = rtn_cfg['usbType']
            #print("USB VERSION:",usb_version)
            #config board param
            self.configBoard(handle,config["board_parameter"])

            if usb_version == ArducamSDK.USB_1 or usb_version == ArducamSDK.USB_2:
                self.configBoard(handle,config["board_parameter_dev2"])
            if usb_version == ArducamSDK.USB_3:
                self.configBoard(handle,config["board_parameter_dev3_inf3"])
            if usb_version == ArducamSDK.USB_3_2:
                self.configBoard(handle,config["board_parameter_dev3_inf2"])
            
            self.writeSensorRegs(handle,config["register_parameter"])
            
            if usb_version == ArducamSDK.USB_3:
                self.writeSensorRegs(handle,config["register_parameter_dev3_inf3"])
            if usb_version == ArducamSDK.USB_3_2:
                self.writeSensorRegs(handle,config["register_parameter_dev3_inf2"])

            ## camera tuning ##
            int_val = self.get_value_for_integration_time()
            ArducamSDK.Py_ArduCam_writeSensorReg( handle, 175, 0 ) # turn off AEC and AGC
            ArducamSDK.Py_ArduCam_writeSensorReg( handle, 11, int_val ) # set integration time
            ArducamSDK.Py_ArduCam_writeSensorReg( handle, 53, 16 ) # set analog gain

            rtn_val,datas = ArducamSDK.Py_ArduCam_readUserData(handle,0x400-16, 16)
            print("Serial: %c%c%c%c-%c%c%c%c-%c%c%c%c"%(datas[0],datas[1],datas[2],datas[3],
                                                        datas[4],datas[5],datas[6],datas[7],
                                                        datas[8],datas[9],datas[10],datas[11]))

            return handle
        else:
            print("open fail,ret_val = ",ret)
            return None

    def open_cameras(self):
        """
            open each camera
        """
        for i in range(self.devices_num):
            handle = self.camera_initFromFile(self.config_file_name, i)
            if handle != None:
                ret_val = ArducamSDK.Py_ArduCam_setMode(handle,ArducamSDK.EXTERNAL_TRIGGER_MODE)
                if(ret_val == ArducamSDK.USB_BOARD_FW_VERSION_NOT_SUPPORT_ERROR):
                    print("USB_BOARD_FW_VERSION_NOT_SUPPORT_ERROR")
                    exit(0)
                # store handels for cameras
                self.handles.append(handle)
    

    def get_images(self):
        """
            get single frame images from cameras via trigger
        """

        self.images = numpy.zeros(( len(self.handles), self.Height, self.Width ))

        for i in range( len(self.handles) ):
            handle = self.handles[i]
            # This function is used to trigger the camera to take image by software
            ArducamSDK.Py_ArduCam_softTrigger(handle)
            # get image
            rtn_val,data,rtn_cfg = ArducamSDK.Py_ArduCam_getSingleFrame(handle)
            image = convert_image(data,rtn_cfg, self.color_mode)
            image = image[:, :, 0]

            # save image
            self.images[i, :, :] = image


    def get_temperature(self):
        tmp_list = []
        for i in range( len(self.handles) ):
            handle = self.handles[i]
            regAddr= 193 # temperature information register
            # https://docs.arducam.com/USB-Industrial-Camera/Quick-Start-Guide/Software-SDK-and-API-for-Windows/Software-SDK-and-API-for-Python/#4242-py_arducam_readsensorreghandle-regaddr
            error_code, tmp = ArducamSDK.Py_ArduCam_readSensorReg(handle, regAddr)
            tmp_list.append(tmp)

        return tmp_list



                





    