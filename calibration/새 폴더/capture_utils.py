import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

import PySpin
import pygame
from pygame.locals import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Total number of buffers
NUM_BUFFERS = 3
SAVE_MODE = False

serial_main = '20356250'
serial_side = '24049908'
# serial_side = '22371101'
serial_list = [serial_main, serial_side]

# Set camera parameters
GAIN = 0
roi = [612, 512, 1224, 1024]
pixel_format = 'BayerRGPolarized8'


def configure_buffer(cam):
    # Retrieve Stream Parameters device nodemap
    s_node_map = cam.GetTLStreamNodeMap()

    # Retrieve Buffer Handling Mode Information
    handling_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferHandlingMode'))
    if not PySpin.IsAvailable(handling_mode) or not PySpin.IsWritable(handling_mode):
        print('Unable to set Buffer Handling mode (node retrieval). Aborting...\n')
        return False

    handling_mode_entry = PySpin.CEnumEntryPtr(handling_mode.GetCurrentEntry())
    if not PySpin.IsAvailable(handling_mode_entry) or not PySpin.IsReadable(handling_mode_entry):
        print('Unable to set Buffer Handling mode (Entry retrieval). Aborting...\n')
        return False

    # Set stream buffer Count Mode to manual
    stream_buffer_count_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferCountMode'))
    if not PySpin.IsAvailable(stream_buffer_count_mode) or not PySpin.IsWritable(stream_buffer_count_mode):
        print('Unable to set Buffer Count Mode (node retrieval). Aborting...\n')
        return False

    stream_buffer_count_mode_manual = PySpin.CEnumEntryPtr(stream_buffer_count_mode.GetEntryByName('Manual'))
    if not PySpin.IsAvailable(stream_buffer_count_mode_manual) or not PySpin.IsReadable(stream_buffer_count_mode_manual):
        print('Unable to set Buffer Count Mode entry (Entry retrieval). Aborting...\n')
        return False

    stream_buffer_count_mode.SetIntValue(stream_buffer_count_mode_manual.GetValue())
    print('Stream Buffer Count Mode set to manual...')

    # Retrieve and modify Stream Buffer Count
    buffer_count = PySpin.CIntegerPtr(s_node_map.GetNode('StreamBufferCountManual'))
    if not PySpin.IsAvailable(buffer_count) or not PySpin.IsWritable(buffer_count):
        print('Unable to set Buffer Count (Integer node retrieval). Aborting...\n')
        return False

    # Display Buffer Info
    print('\nDefault Buffer Handling Mode: %s' % handling_mode_entry.GetDisplayName())
    print('Default Buffer Count: %d' % buffer_count.GetValue())
    print('Maximum Buffer Count: %d' % buffer_count.GetMax())

    buffer_count.SetValue(NUM_BUFFERS)

    print('Buffer count now set to: %d' % buffer_count.GetValue())

    handling_mode_entry = handling_mode.GetEntryByName('OldestFirst')
    handling_mode.SetIntValue(handling_mode_entry.GetValue())
    print('\n\nBuffer Handling Mode has been set to %s' % handling_mode_entry.GetDisplayName())
    
def configure_cam(cam, pixel_format, exposure_time_to_set_us, roi=None):
    # pixel_format: 'Mono16' <- maybe not correct, color pixel format is expected
    GAIN = 0
    
    # # roi [x, w, y, h] -> [x, y, w, h]
    # if roi is not None:
    #     w = roi[1]
    #     y = roi[2]
    #     roi[1] = y
    #     roi[2] = w
        
    # Retrieve GenICam nodemap
    nodemap = cam.GetNodeMap()
    
    # Image format
    # if cam.PixelFormat.GetAccessMode() == PySpin.RW:
    #     cam.PixelFormat.SetValue(pixel_format)
    #     print('Pixel format set to %s...' % cam.PixelFormat.GetCurrentEntry().GetSymbolic())
    node_pixel_format = PySpin.CEnumEntryPtr(nodemap.GetNode('PixelFormat'))
    if PySpin.IsAvailable(node_pixel_format) and PySpin.IsWritable(node_pixel_format):
        # Retrieve the desired entry node from the enumeration node
        node_pixel_format_cur = PySpin.CEnumEntryPtr(node_pixel_format.GetEntryByName(pixel_format))  
        print(pixel_format_cur)
        if PySpin.IsAvailable(node_pixel_format_cur) and PySpin.IsReadable(node_pixel_format_cur):
            # Retrieve the integer value from the entry node
            pixel_format_cur = node_pixel_format_cur.GetValue()
            # Set integer as new value for enumeration node
            node_pixel_format.SetIntValue(pixel_format_cur)
            print('Pixel format set to %s...' % node_pixel_format.GetCurrentEntry().GetSymbolic())
        else:
            print('Pixel format %s not available...'%pixel_format)
    else:
        print('Pixel format not available...')
        
    # Set ROI first if applicable (framerate limits depend on it)
    try:
        # Note set width/height before x/y offset because upon
        # initialization max offset is 0 because full frame size is assumed
        if roi is not None:
            for i, (s, o) in enumerate(zip(["Width", "Height"], ["OffsetX", "OffsetY"])):

                roi_node = PySpin.CIntegerPtr(nodemap.GetNode(s))
                # If no ROI is specified, use full frame:
                if roi[2 + i] == -1:
                    value_to_set = roi_node.GetMax()
                else:
                    value_to_set = roi[2 + i]
                inc = roi_node.GetInc()
                if np.mod(value_to_set, inc) != 0:
                    value_to_set = (value_to_set // inc) * inc
                roi_node.SetValue(value_to_set)

                # offset
                offset_node = PySpin.CIntegerPtr(nodemap.GetNode(o))
                if roi[0 + i] == -1:
                    off_to_set = offset_node.GetMin()
                else:
                    off_to_set = roi[0 + i]
                offset_node.SetValue(off_to_set)
                
    except Exception as ex:
        print("E:Could not set ROI. Exception: {0}.".format(ex))

    
    # White balancing    
    # Auto off
    cam.BalanceWhiteAuto.SetValue(PySpin.BalanceWhiteAuto_Off)
    
    node_white_balance_format = PySpin.CEnumerationPtr(nodemap.GetNode('BalanceRatioSelector'))
    balance_ratio_node = PySpin.CFloatPtr(nodemap.GetNode('BalanceRatio'))
    
    # Blue
    if PySpin.IsAvailable(node_white_balance_format) and PySpin.IsWritable(node_white_balance_format):
        # Retrieve the desired entry node from the enumeration node
        node_white_balance_format_cur = PySpin.CEnumEntryPtr(node_white_balance_format.GetEntryByName("Blue")) 
        if PySpin.IsAvailable(node_white_balance_format_cur) and PySpin.IsReadable(node_white_balance_format_cur):
            # Retrieve the integer value from the entry node
            # Set integer as new value for enumeration node
            cam.BalanceRatioSelector.SetValue(PySpin.BalanceRatioSelector_Blue)
            balance_ratio_node.SetValue(1.0)
            print('WB blue set to %s...' % balance_ratio_node.GetValue())
        else:
            print('WB %s not available...'%("blue"))
            
        node_white_balance_format_cur = PySpin.CEnumEntryPtr(node_white_balance_format.GetEntryByName("Red"))  
        if PySpin.IsAvailable(node_white_balance_format_cur) and PySpin.IsReadable(node_white_balance_format_cur):
            # Retrieve the integer value from the entry node
            # Set integer as new value for enumeration node
            cam.BalanceRatioSelector.SetValue(PySpin.BalanceRatioSelector_Red)
            balance_ratio_node.SetValue(1.0)
            print('WB red set to %s...' % balance_ratio_node.GetValue())
        else:
            print('WB %s not available...'%("red"))

    else:
        print('WB not available...')

    # Turn off Auto Gain
    node_gainauto_mode = PySpin.CEnumerationPtr(nodemap.GetNode("GainAuto"))
    node_gainauto_mode_off = node_gainauto_mode.GetEntryByName("Off")
    node_gainauto_mode.SetIntValue(node_gainauto_mode_off.GetValue())

    # Set gain to 0 dB
    node_iGain_float = PySpin.CFloatPtr(nodemap.GetNode("Gain"))
    node_iGain_float.SetValue(GAIN)

    # Turn on Gamma
    node_gammaenable_mode = PySpin.CBooleanPtr(nodemap.GetNode("GammaEnable"))
    node_gammaenable_mode.SetValue(True)

    # Set Gamma as 1
    node_Gamma_float = PySpin.CFloatPtr(nodemap.GetNode("Gamma"))
    node_Gamma_float.SetValue(1)
    
    # Configure exposure
    if cam.ExposureAuto.GetAccessMode() != PySpin.RW:
        print('Unable to disable automatic exposure. Aborting...')
        return False

    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
    # Set exposure time manually; exposure time recorded in microseconds
    if cam.ExposureTime.GetAccessMode() != PySpin.RW:
        print('Unable to set exposure time. Aborting...')
        return False

    # Ensure desired exposure time does not exceed the maximum
    exposure_time_to_set = min(cam.ExposureTime.GetMax(), exposure_time_to_set_us)
    cam.ExposureTime.SetValue(exposure_time_to_set)
    print('Shutter time set to %s ms...\n' % (exposure_time_to_set/1e3))
    
    # Ensure trigger mode off
    # The trigger must be disabled in order to configure whether the source
    # is software or hardware.
    nodemap = cam.GetNodeMap()
    node_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
    if not PySpin.IsAvailable(node_trigger_mode) or not PySpin.IsReadable(node_trigger_mode):
        print('Unable to disable trigger mode (node retrieval). Aborting...')
        return False

    node_trigger_mode_off = node_trigger_mode.GetEntryByName('Off')
    if not PySpin.IsAvailable(node_trigger_mode_off) or not PySpin.IsReadable(node_trigger_mode_off):
        print('Unable to disable trigger mode (enum entry retrieval). Aborting...')
        return False

    node_trigger_mode.SetIntValue(node_trigger_mode_off.GetValue())

    print('Trigger mode disabled...')
    
    # Set TriggerSelector to FrameStart
    # For this example, the trigger selector should be set to frame start.
    # This is the default for most cameras.
    node_trigger_selector= PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSelector'))
    if not PySpin.IsAvailable(node_trigger_selector) or not PySpin.IsWritable(node_trigger_selector):
        print('Unable to get trigger selector (node retrieval). Aborting...')
        return False

    node_trigger_selector_framestart = node_trigger_selector.GetEntryByName('FrameStart')
    if not PySpin.IsAvailable(node_trigger_selector_framestart) or not PySpin.IsReadable(
            node_trigger_selector_framestart):
        print('Unable to set trigger selector (enum entry retrieval). Aborting...')
        return False
    node_trigger_selector.SetIntValue(node_trigger_selector_framestart.GetValue())
    
    print('Trigger selector set to frame start...')
    
    # Select trigger source
    # The trigger source must be set to hardware or software while trigger
    # mode is off.
    node_trigger_source = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSource'))
    if not PySpin.IsAvailable(node_trigger_source) or not PySpin.IsWritable(node_trigger_source):
        print('Unable to get trigger source (node retrieval). Aborting...')
        return False

    node_trigger_source_software = node_trigger_source.GetEntryByName('Software')
    if not PySpin.IsAvailable(node_trigger_source_software) or not PySpin.IsReadable(
            node_trigger_source_software):
        print('Unable to set trigger source (enum entry retrieval). Aborting...')
        return False
    node_trigger_source.SetIntValue(node_trigger_source_software.GetValue())
    print('Trigger source set to software...')
    
    # Turn trigger mode on
    # Once the appropriate trigger source has been set, turn trigger mode
    # on in order to retrieve images using the trigger.
    node_trigger_mode_on = node_trigger_mode.GetEntryByName('On')
    if not PySpin.IsAvailable(node_trigger_mode_on) or not PySpin.IsReadable(node_trigger_mode_on):
        print('Unable to enable trigger mode (enum entry retrieval). Aborting...')
        return False

    node_trigger_mode.SetIntValue(node_trigger_mode_on.GetValue())
    print('Trigger mode turned back on...')
    
    # Set acquisition mode to continuous
    # In order to access the node entries, they have to be casted to a pointer type (CEnumerationPtr here)
    node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
    if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
        print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
        return False

    # Retrieve entry node from enumeration node
    node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
    if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
            node_acquisition_mode_continuous):
        print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
        return False

    # Retrieve integer value from entry node
    acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

    # Set integer value from entry node as new value of enumeration node
    node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

    print('Acquisition mode set to continuous...')
    print('balance ratio:', balance_ratio_node.GetValue(), 'exposure time',cam.ExposureTime.GetValue())

def trigger_on(cam):
    nodemap = cam.GetNodeMap()
    # Execute software trigger
    node_softwaretrigger_cmd = PySpin.CCommandPtr(nodemap.GetNode('TriggerSoftware'))
    if not PySpin.IsAvailable(node_softwaretrigger_cmd) or not PySpin.IsWritable(node_softwaretrigger_cmd):
        print('Unable to execute trigger. Aborting...')

    node_softwaretrigger_cmd.Execute()
    # TODO: Blackfly and Flea3 GEV cameras need 2 second delay after software trigger
    
def reset_trigger(cam):
    """
    This function returns the camera to a normal state by turning off trigger mode.

    :param nodemap: Device nodemap to retrieve images from.
    :type nodemap: INodeMap
    :return: True if successful, False otherwise
    :rtype: bool
    """
    try:
        result = True
        nodemap = cam.GetNodeMap()

        # Turn trigger mode back off
        #
        # *** NOTES ***
        # Once all images have been captured, turn trigger mode back off to
        # restore the camera to a clean state.
        trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        if not PySpin.IsAvailable(trigger_mode) or not PySpin.IsWritable(trigger_mode):
            print('Unable to disable trigger mode (node retrieval). Non-fatal error...\n')
            return False

        trigger_mode_off = PySpin.CEnumEntryPtr(trigger_mode.GetEntryByName('Off'))
        if not PySpin.IsAvailable(trigger_mode_off) or not PySpin.IsReadable(trigger_mode_off):
            print('Unable to disable trigger mode (enum entry retrieval). Non-fatal error...\n')
            return False

        trigger_mode.SetIntValue(trigger_mode_off.GetValue())
        print('Trigger mode disabled...\n')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result

def capture_one_camera(cam, dir, file_name):
    
    trigger_on(cam)
    im_raw = cam.GetNextImage()
    im_raw_con = im_raw#.Convert(PySpin.PixelFormat_BayerRGPolarized8, PySpin.HQ_LINEAR) # PySpin.HQ_LINEAR

    # RGB numpy array
    im_raw_dat = im_raw_con.GetData()

    width = im_raw.GetWidth()
    height = im_raw.GetHeight()
    im_raw_dat = im_raw_dat.reshape((height, width, -1))
    cv2.imwrite(os.path.join(dir, (file_name)), im_raw_dat[:,:,::-1])

    im_raw.Release()
    
    return

def capture_multiple_cameras(cam_list, dir, file_name):
    time.sleep(0.1)

    for i, cam in enumerate(cam_list):
            
        trigger_on(cam)
        
        im_raw = cam.GetNextImage()
        im_raw_con = im_raw#.Convert(PySpin.PixelFormat_BayerRGPolarized8, PySpin.HQ_LINEAR) # PySpin.HQ_LINEAR

        # RGB numpy array
        im_raw_dat = im_raw_con.GetData()

        width = im_raw.GetWidth()
        height = im_raw.GetHeight()
        im_raw_dat = im_raw_dat.reshape((height, width, -1))
        
        if serial_list[i] == serial_main:
            cv2.imwrite(os.path.join(dir, 'main', (file_name)), im_raw_dat[:,:,::-1])
        else:
            cv2.imwrite(os.path.join(dir, 'side', (file_name)), im_raw_dat[:,:,::-1])

        im_raw.Release()

    time.sleep(0.1)
    return

def initialize_cam(pixel_format, SHUTTER_TIME, roi):
    
    # Initialize cameras
    system = PySpin.System.GetInstance()

    # Get current library version
    version = system.GetLibraryVersion()
    print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

    cam_list = system.GetCameras()
    print(cam_list)
    num_cameras = cam_list.GetSize()
    print('Number of cameras detected: %d' % num_cameras)
    # Finish if there are no cameras
    if num_cameras == 0:
        # Clear camera list before releasing system
        cam_list.Clear()
        # Release system instance
        system.ReleaseInstance()
        print('Not enough cameras!')
        input('Done! Press Enter to exit...')
        sys.exit(1)

    serial_list = []
    for i, cam in enumerate(cam_list):
        cam.Init()
        print(cam.DeviceModelName())
        
        nodemap_tldevice = cam.GetTLDeviceNodeMap()
        device_serial_number = ''
        node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
        if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
            device_serial_number = node_device_serial_number.GetValue()
            print(device_serial_number)
            serial_list.append(device_serial_number)

        configure_cam(cam, pixel_format, SHUTTER_TIME, roi=roi)
        configure_buffer(cam)
        cam.BeginAcquisition()
        
    return system, cam, cam_list