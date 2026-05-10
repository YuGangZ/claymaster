#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import serial
import time
import sys
import os
import threading
import copy
import win32ui
import re
import struct

global connect_status
connect_status = 0
global timer
timer = None
global flag
flag = 0
global len_num
len_num = 0
global rxpack
rxpack = []
global compare
compare = 0
global movetypes
movetypes = 0
global pathtype
global info_flag
info_flag = 0
enclosure_id_list = []
INST_HEADER0 = 0
INST_HEADER1 = 1
INST_LENGTH = 2
INST_FUNCID = 3
INST_CTRL = 4
TXPACKET_MAX_LEN = 250
RXPACKET_MAX_LEN = 48
FBPA_LENGTH = 2
sers = None
path_type = None
repeat_num = 3
over_time = 0.5
global funcid
funcid = 0
global accessid
ctrlid = 0
global receive
receive = False
global callback
callback = None
robotsta_dic = {17: "呼吸黄色灯 未连接机械臂", 1: "呼吸蓝色灯 空闲连接 PC ", 7: "呼吸蓝色灯 由 PowerBtn 造成的空闲",
                2: "常亮蓝色灯 正在执行任务状态 ", 14: "常亮蓝色灯 正在执行任务状态, 由取消当前动作造成",
                20: "常亮蓝色灯 校准状态", 32: "常亮蓝色灯 固件升级状态", 3: "常亮黄色灯 暂停状态 ", 4: "常亮红色灯 首次检测到障碍物 ",
                16: "常亮红色灯 紧急停止/再次检测到障碍物", 22: "常亮红色灯 开机初始化失败", 28: "常亮红色灯 开机检测到刻度跳变",
                34: "常亮红色灯 机械臂无法控制执行器", 5: "逆时针蓝色灯 关机状态"}
RealInfo_hight_dic = {0: "无队列指令执行/队列指令正常执行", 16: "指令执行成功未抵达目标点;", 32: "当前路径越界无法执行",
                      48: "程序处于非空闲/运行状态 无法执行当前指令", 64: "未知队列指令 无法执行",
                      80: "未连接机械臂无法执行"}
RealInfo_low_dic = {0: "无按键操作", 1: " PowerBtn长按 2s关机", 2: " PowerBtn 短按结束任务", 3: "RunBtn 按键一次暂停任务",
                    4: "RunBtn 按键一次继续任务", 5: "LockBtn 按键一直按下机械臂松掉刚度", 6: "LockBtn 按键松开"}

accessid_dic = {1: "旋转气嘴", 2: "气三爪", 3: "单笔", 4: "四芯转笔", 5: "激光头", 6: "电两爪", 7: "单吸盘", 8: "IO板",
                9: "传送带", 10: "滑动台", 11: "颜色传感器", 12: "电磁铁"}
read_accessid_dic = {1: "旋转气嘴", 2: "IO板", 3: "传送带", 4: "滑动台", 5: "颜色传感器"}

read_funcid_dic = {1: {1: "读取旋转气嘴的ID ", 2: "设置旋转气嘴使能", 3: "读取舵机的当前角度 ", 4: "写入舵机的目标角度"},
                   2: {1: "读取气三爪的 ID "}, 3: {1: "读取单笔的ID "}, 4: {1: "读取四芯转笔的 ID",
                                                                 2: "设置四芯转笔使能", 3: "选择笔的编号"},
                   5: {1: "读取激光头的 ID", 2: "设置使能 ", 3: "设定激光头的功率/脉冲百分比"},
                   6: {1: "读取电两爪的 ID ", 2: "设施电两爪使能", 3: "控制电两爪的张开幅度"}, 7: {1: "读取单吸盘的 ID "},
                   8: {1: "读取 ID", 2: "读取 IO 输入", 3: "读取 IO 输入锁存置 ", 4: "读取 IO 输出 ", 5: "清空 IO 输入锁存/输出值"},
                   9: {1: "读取 ID", 2: "设置使能", 3: "读取速度", 4: "读取正反转"}, 10: {1: "读写 ID", 2: "设置速度",
                                                                           3: "设置使能", 4: "回零功能", 5: "检测找零完成",
                                                                           6: "设置目标位置", 7: "检测是否到达目标位置"},
                   11: {1: "读取 ID", 2: "设置使能", 3: "读颜色校准结果", 4: "设置白平衡/颜色识别操作", 5: "读取颜色识别结果",
                        6: "获取颜色的 RGB 值"}}

set_funcid_dic = {1: {1: "写入旋转气嘴的ID ", 2: "设置旋转气嘴使能", 3: "读取舵机的当前角度 ", 4: "写入舵机的目标角度"},
                  2: {1: "写入气三爪的 ID "}, 3: {1: "写入单笔的ID "}, 4: {1: "写入四芯转笔的 ID",
                                                                2: "设置四芯转笔使能", 3: "选择笔的编号"},
                  5: {1: "写入激光头的 ID", 2: "设置使能 ", 3: "设定激光头的功率/脉冲百分比"},
                  6: {1: "写入电两爪的 ID ", 2: "设施电两爪使能", 3: "控制电两爪的张开幅度"}, 7: {1: "写入单吸盘的 ID "},
                  8: {1: "写入 ID", 2: "读取 IO 输入", 3: "读取 IO 输入锁存置 ", 4: "设置 IO 输出 ", 5: "清空 IO 输入锁存/输出值"},
                  9: {1: "写入 ID", 2: "设置使能", 3: "读取/设置速度", 4: "设置正反转"}, 10: {1: "读写 ID", 2: "设置速度",
                                                                             3: "设置使能", 4: "回零功能", 5: "检测找零完成",
                                                                             6: "设置目标位置", 7: "检测是否到达目标位置"},
                  11: {1: "写入 ID", 2: "设置使能", 3: "读颜色校准结果", 4: "设置白平衡/颜色识别操作", 5: "读取颜色识别结果",
                       6: "获取颜色的 RGB 值"},
                  12: {2: "设置使能", 3: "设置磁力"}}

accessid_dic2 = {0: "无", 8: "旋转气嘴", 16: "气三爪", 24: "单笔", 32: "四芯转笔", 40: "激光头", 48: "电两爪", 56: "单吸盘", 80: "IO板",
                 128: "传送带", 136: "滑动台", 192: "颜色传感器", 64: "电磁铁"}

value_dict1 = {0: "禁止", 1: "使能"}
value_dict2 = {0: "关闭", 1: "功率模式", 2: "脉冲模式"}
value_dict3 = {1: "开启", 2: "关闭", 3: "暂停"}
value_dict4 = {1: "正转", 2: "反转"}
value_dict5 = {0: "关闭", 1: "开启"}
value_dict6 = {0: "找零未完成", 1: "找零完成"}
value_dict7 = {1: "表示到达目标位置", 0: "表示未到达目标位置"}
value_dict8 = {0: "校准失败", 1: "校准成功"}
value_dict9 = {1: "白平衡调整", 2: "识别颜色"}
value_dict10 = {1: "红色", 2: "黄色", 3: "绿色", 4: "蓝色", 5: "紫色", 6: "白色", 7: "粉红", 9: "橙色", 10: "黑色"}

error_dict = {0x00: "指令执行/接收成功", 0x02: "队列指令仓库已满, 无法响应", 0x01: "接收指令校验码错误, 无法响应",
              0x03: "未知指令, 无法响应", 0x04: "指令执行失败", 0x05: "请先连接机械臂"}
# 重发次数
def repeat_send(nums):
    global repeat_num
    repeat_num = nums


# 超时设置
def overtime(times):
    global over_time
    over_time = times


# 发送数据包
def send_packet(txPacket):
    try:
        global repeat_num
        global receive
        for i in range(repeat_num):
            # 如果没有接收到数据，继续发送数据包
            if not receive:
                result = TxPacket(txPacket)
                # 如果是校准操作（功能码为0x7D，且状态为开始），等待3秒
                if txPacket[1] == 0x7D and txPacket[2] == 0x03 and txPacket[3] == 2:
                    time.sleep(3)
                elif txPacket[1] == 131 and txPacket[2] == 2:
                    time.sleep(3)
                # 否则使用默认的超时时间
                else:
                    time.sleep(over_time)
            # 如果接收到数据，重置receive并退出循环
            elif receive:
                receive = False
                break
            # 如果达到最大重试次数，重置receive
            elif i == repeat_num - 1:
                receive = False
    except Exception as e:
        print(e)    # 捕获并打印异常


def deal_params(num):
    try:
        lists = []
        # 如果 num 是负数，往 lists 中添加一个值 0x01，表示负号
        if float(num) < 0:
            lists.append(0x01)
        # 如果 num 是正数，往 lists 中添加一个值 0x00，表示正号
        elif float(num) > 0:
            lists.append(0x00)
        # 如果 num 是 0，往 lists 中添加四个值 0x00，表示 0
        elif float(num) == 0:
            lists.extend([0x00, 0x00, 0x00, 0x00])
            return lists
        # 将 num 转换为字符串，并去除字符串中的负号
        num_list = str(num).strip("-").split(".")
        # 如果 num_list 的长度为 1，往 num_list 中添加一个值 "0"，表示小数点后为 0
        if len(num_list) == 1:
            num_list.append("0")
        # 将 num_list 中的第一个元素转换为 16 进制字符串，并截取前两位和后两位，分别转换为整数，添加到 lists 中
        num_str = '{:04X}'.format(int(num_list[0]))
        # 将该字符串的前两位和后两位分别转化为整数，然后按 16 进制格式保存为 num1 和 num2。这些数值将分别作为整数部分的高位和低位字节。
        num1, num2 = int((num_str[:2]), 16), int((num_str[2:4]), 16)
        # 小数部分的处理只取前两位（num_list[1][:2]），如果小数部分长度只有 1 位，将其乘以 10 以确保保留两位数字，然后转化为 16 进制格式。如果小数部分已经有两位，直接将其转化为 16 进制数。
        # 最终，这个十六进制数会被作为小数部分的数据保存在 num_str2 中。
        if len(num_list[1][:2]) == 1:
            num_str2 = int('{:02X}'.format((int(num_list[1][:2]) * 10)), 16)
        elif len(num_list[1][:2]) == 2:
            num_str2 = int('{:02X}'.format((int(num_list[1][:2]))), 16)
        lists.append(num1)
        lists.append(num2)
        lists.append(num_str2)
        return lists
    except:
        if timer:
            timer.cancel()
        print("参数有误")


# 发送信息
# 组装数据包，添加校验码和包头，并通过串口发送出去。函数对各种异常情况进行了处理，确保在串口未打开或发送失败时能及时处理错误。
def TxPacket(txPacket):
    # 深拷贝一份到 txPacket2，避免对原始数据进行修改。
    txPacket2 = copy.deepcopy(txPacket)
    # print(txPacket2)
    # 全局变量，串口对象
    global sers
    if sers:
        checkSum = 0    # 检查指令长度
        written_packet_length = 0    # 写入指令长度
        # 添加指令头
        # 在 txPacket2 的开始位置添加两个字节 0xBB，这是数据包的指令头，表示数据传输的开始。INST_HEADER0 和 INST_HEADER1 代表插入的位置
        txPacket2.insert(INST_HEADER0, 0xBB)
        txPacket2.insert(INST_HEADER1, 0xBB)
        # 检测指令长度
        # 从指令包的指定位置 INST_LENGTH 读取包长度，并加上包头等固定字节（共4个字节）。总包长不能超过 TXPACKET_MAX_LEN，否则返回 False，表示失败
        total_packet_length = txPacket2[INST_LENGTH] + 4
        if total_packet_length > TXPACKET_MAX_LEN:
            return False
        # 添加校验码到指令尾
        # 遍历 txPacket2 的第3个字节到最后一个字节，将每个字节的值累加到 checkSum 变量中。
        # 取校验和的补码 ~checkSum & 0xFF（即取反后只保留最后8位）再加1，并将这个校验码追加到指令包末尾，保证数据的完整性
        for idx in range(3, len(txPacket2)):
            checkSum += txPacket2[idx]
        txPacket2.append((~checkSum & 0xFF) + 1)
        try:
            # 如果最后一个字节是 256，则将其改为 0
            # bytes(txPacket2) 将 txPacket2 转换为字节流
            if txPacket2[-1] == 256:
                txPacket2[-1] = 0
            txPacket2 = bytes(txPacket2)
        except:
            if timer:
                timer.cancel()
            print("参数范围在0-255")
        try:
            # 写入指令
            written_packet_length = sers.write(txPacket2)
        except Exception as e:
            print(e)
            sers.close()
            sers = None
        if total_packet_length != written_packet_length:
            return False
        return True
    else:
        print("串口未打开")


# 连接开启端口
def open_port(port_name, baud_rate, callbacks=None):
    global callback
    callback = callbacks
    global sers
    try:
        sers = serial.Serial(port_name, baud_rate, bytesize=serial.EIGHTBITS, timeout=0)
        sers.flushInput()
        global timer
        timer = threading.Timer(0.0001, data_receive)
        timer.start()
    except Exception as e:
        if timer:
            timer.cancel()
        print(e)
        print('端口开启失败')
        os.system('taskkill /f /im python.exe')
        # if sers:
        #     sers.close()
        # sys.exit(1)  # 正常退出当前程序


# 发送ping    1：表示连接机械臂;      2：表示断开连接
def ping(state):
    txPacket = [0x03, 0x00, 0x01, state]
    global funcid
    funcid = 0x00
    global ctrlid
    ctrlid = 0x01
    send_packet(txPacket)


# # 固件升级
# def update_firmware(vcode, pvsion):
#     """
#     :param vcode:版本标志码;例如：115
#     :param pvsion:PC端版本编号：用三个字节表示：例如：1.1.5
#     :return:无
#     """
#     try:
#         txPacket = [0x01, 0x03]
#         dlg = win32ui.CreateFileDialog(1)  # 1表示打开文件对话框
#         dlg.DoModal()
#         filename = dlg.GetPathName()  # 获取选择的文件名称
#         size = os.path.getsize(filename)
#         avsion = int(re.findall(r'\d+', filename)[0])
#         avsion_str = "{:08X}".format(avsion)
#         size_str = "{:08X}".format(size)
#         vcode1 = "{:04X}".format(int(vcode))
#         pvsion_list = str(pvsion).split(".")
#         params_list = [int(avsion_str[:2], 16), int(avsion_str[2:4], 16), int(avsion_str[4:6], 16),
#                        int(avsion_str[6:8], 16), int(size_str[:2], 16)
#             , int(size_str[2:4], 16), int(size_str[4:6], 16), int(size_str[6:8], 16),
#                        int(vcode1[:2], 16),
#                        int(vcode1[2:4], 16),
#                        int(pvsion_list[0], 16), int(pvsion_list[1], 16), int(pvsion_list[2], 16)]
#         txPacket.extend(params_list)
#         txPacket.insert(0, len(txPacket))
#         global funcid
#         funcid = 0x01
#         global ctrlid
#         ctrlid = 0x03
#         send_packet(txPacket)
#         funcid = 0x02
#         readfilethread = ReadFileThread(filename)
#         readfilethread.start()
#         readfilethread.join()
#         txPacket = [0x03, 0x03, 0x03, 0x00]
#         funcid = 0x03
#         send_packet(txPacket)
#     except Exception as e:
#         print(e)


# # 读取版本
# def get_version():
#     txPacket = [2, 2, 2]
#     global funcid
#     funcid = 0x02
#     global ctrlid
#     ctrlid = 0x02
#     global repeat_num
#     global receive
#     send_packet(txPacket)

# 软件关机
def shutdown_software():
    txPacket = [2, 4, 3]
    global funcid
    funcid = 0x04
    global ctrlid
    ctrlid = 0x03
    send_packet(txPacket)


# 读取机械臂种类
def get_arm_dev():
    txPacket = [2, 5, 2]
    global funcid
    funcid = 0x05
    global ctrlid
    ctrlid = 0x02
    send_packet(txPacket)


# 设置电流阈值
def set_electric_current(value):
    try:
        if float(value) < 0 or float(value) > 5500:
            print('电流阈值为0-5500')
        else:
            txPacket = [0x3c, 0x03]
            params_list = deal_params(value)
            txPacket.extend(params_list)
            leng = len(txPacket)
            txPacket.insert(0, leng)
            global funcid
            funcid = 0x3c
            global ctrlid
            ctrlid = 0x03
            send_packet(txPacket)
    except:
        if timer:
            timer.cancel()
        print("电流阈值参数有误")


# 设置回零位置
def set_zero_position(x, y, z):
    txPacket = [0x78, 0x03]
    try:
        x_list = deal_params(x)
        y_list = deal_params(y)
        z_list = deal_params(z)
        txPacket.extend(x_list)
        txPacket.extend(y_list)
        txPacket.extend(z_list)
    except:
        if timer:
            timer.cancel()
        print('参数有误')
    if len(txPacket) > 2:
        leng = len(txPacket)
        txPacket.insert(0, leng)
        global funcid
        funcid = 0x78
        global ctrlid
        ctrlid = 0x03
        send_packet(txPacket)


# 执行回零
def move_zero():
    txPacket = [0x02, 0x79, 0x04]
    global funcid
    funcid = 0x79
    global ctrlid
    ctrlid = 0x04
    send_packet(txPacket)


# 紧急停止
def stop():
    txPacket = [0x02, 0x79, 0x03]
    global funcid
    funcid = 0x79
    global ctrlid
    ctrlid = 0x03
    send_packet(txPacket)


# 附件回零
def enclosure_zero():
    txPacket = [0x02, 0x7A, 0x04]
    global funcid
    funcid = 0x7A
    global ctrlid
    ctrlid = 0x04
    send_packet(txPacket)


# 校准操作
def calibration(status):
    """
    :param status: 校准状态，0x01进入校准，0x02开始校准，0x03结束校准
    :return: None
    """
    txPacket = [0x03, 0x7D, 0x03, status]
    global funcid
    funcid = 0x7D
    global ctrlid
    ctrlid = 0x03
    send_packet(txPacket)


# 扫描附件
def get_enclosure():
    # 声明 enclosure_id_list 为全局变量，确保在函数内操作的 enclosure_id_list 与外部的全局变量是同一个列表
    global enclosure_id_list
    enclosure_id_list = []
    txPacket = [0x02, 0x83, 0x02]
    global funcid
    funcid = 0x83
    global ctrlid
    ctrlid = 0x02
    send_packet(txPacket)
    for i in range(100):
        if enclosure_id_list:
            return enclosure_id_list
        else:
            time.sleep(0.1)



# 读取附件char/folat参数
def get_enclosure_params(access_id, fun_id):
    """
    :param access_id: 附件id
    :param fun_id: 功能id
    :return:
    """
    txPacket = [0x04, 0xBE, 0x02, access_id, fun_id]
    global funcid
    funcid = 0xBE
    global ctrlid
    ctrlid = 0x02

    send_packet(txPacket)


# 设置附件char/folat参数
def set_enclosure(*args):
    """
    :param args: 第一个参数是accessid ,第二个参数是funid,其余参数为设置值
    :return:
    """
    try:
        global funcid
        funcid = 0xBE
        global ctrlid
        ctrlid = 0x03
        accessid = args[0]
        funid = args[1]
        params_list = list(args)[2:]
        txPacket = [0xBE, 0x03, accessid, funid]
        if accessid == 8:
            if funid == 2:
                if int(params_list[0]) not in [0, 1]:
                    print("参数为0，1")
                    return
                else:
                    txPacket.append(int(params_list[0]))
            elif funid == 4:
                GAngle = float(params_list[0])
                if GAngle > 100 or GAngle < -100:
                    print("参数在-100~100之内")
                    return
                else:
                    cangle_list = deal_params(params_list[0])
                    txPacket.extend(cangle_list)

        elif accessid == 32:
            if funid == 2:
                if int(params_list[0]) in [0, 1]:
                    txPacket.append(int(params_list[0]))
                else:
                    print("参数为0或者1")
                    return
            elif funid == 3:
                num = 0
                if int(params_list[0]) in [0, 1, 2, 3, 4]:
                    num = int(params_list[0])
                    txPacket.append(num)
                else:
                    print("参数为0或者1或者2或者3或者4")
                    return

        elif accessid == 40:
            if funid == 2:
                if int(params_list[0]) in [0, 1, 2]:
                    txPacket.append(int(params_list[0]))
                else:
                    print("参数为0或者1或者2")
                    return
            elif funid == 3:

                if 0 <= int(params_list[0]) <= 100:
                    txPacket.append(int(params_list[0]))
                else:
                    print("参数为0-100")
                    return

        elif accessid == 48:
            if funid == 2:
                if int(params_list[0]) in [0, 1]:
                    txPacket.append(int(params_list[0]))
                else:
                    print("参数为0或者1")
                    return
            elif funid == 3:
                if 0 <= int(params_list[0]) <= 100:
                    txPacket.append(int(params_list[0]))
                else:
                    print("参数为0-100")
                    return
        elif accessid == 64:
            if funid == 2:
                if int(params_list[0]) in [0, 1]:
                    txPacket.append(int(params_list[0]))
                else:
                    print("参数为0或者1")
                    return
            elif funid == 3:
                if int(params_list[0]) in [0, 1]:
                    txPacket.append(int(params_list[0]))
                else:
                    print("参数为0或者1")
                    return
        elif accessid == 80:
            if funid in [2, 3, 4]:
                txPacket.append(int(params_list[0]))
            if funcid == 5:
                pass
        elif accessid == 128:
            if funid == 2:
                if int(params_list[0]) in [1, 2, 3]:
                    txPacket.append(int(params_list[0]))
                else:
                    print("参数为1或者2或者3")
                    return
            elif funid == 3:
                speed = params_list[0]
                if float(speed) < 16 or float(speed) > 306:
                    print("速度范围在16-306")
                    return
                speed_list = deal_params(speed)
                txPacket.extend(speed_list)
            elif funid == 4:
                if int(params_list[0]) in [1, 2]:
                    txPacket.append(int(params_list[0]))
                else:
                    print("参数为1或者2")
                    return
        elif accessid == 136:
            if funid == 3:
                if int(params_list[0]) in [0, 1]:
                    txPacket.append(int(params_list[0]))
                else:
                    print("参数为0或者1")
            elif funid == 4:
                if int(params_list[0]) == 1:
                    txPacket.append(int(params_list[0]))
                else:
                    print("参数为1")
                    return
            elif funid == 2:
                speed = params_list[0]
                if float(speed) < 10 or float(speed) > 153:
                    print("速度范围在10-153")
                    return
                speed_list = deal_params(speed)
                txPacket.extend(speed_list)
            elif funid == 6:
                position = params_list[0]
                if float(position) < 0 or float(position) > 1200:
                    print("范围在0-1200")
                    return
                position_list = deal_params(position)
                txPacket.extend(position_list)
        elif accessid == 192:
            if funid == 2:
                if int(params_list[0]) in [0, 1]:
                    txPacket.append(int(params_list[0]))
                else:
                    print("参数为0或者1")
                    return
            if funid == 4:
                if int(params_list[0]) in [1, 2]:
                    txPacket.append(int(params_list[0]))
                else:
                    print("参数为1或者2")
                    return
        leng = len(txPacket)
        txPacket.insert(0, leng)
        send_packet(txPacket)
    except Exception as e:
        print(e)


# 设置控制信号
def set_control_signal(actsig):
    """
    :param actsig: 控制信号：0x01 开始，0x02 暂停，0x03 继续，0x04 取消，0x05 结束
    :return:None
    """
    txPacket = [0x03, 0x41, 0x03, actsig]
    global funcid
    funcid = 0x41
    global ctrlid
    ctrlid = 0x03
    send_packet(txPacket)


# 设置运动速率
def set_speed_rate(speed_rate):
    spe = -1
    try:
        spe = float(speed_rate)
        if spe < 0 or spe > 1:
            if timer:
                timer.cancel()
                print( '速度倍率范围为0-1')
        else:
            txPacket = [0x42, 0x03]
            params_list = deal_params(speed_rate)
            txPacket.extend(params_list)
            leng = len(txPacket)
            txPacket.insert(0, leng)
            global funcid
            funcid = 0x42
            global ctrlid
            ctrlid = 0x03
            send_packet(txPacket)
    except:
        if timer:
            timer.cancel()
        print('参数有误')


# 单点运动
def point_movement(movetype, x1, y1, z1, x2, y2, z2):
    txPacket = [0x44, 0x04, movetype]
    global funcid
    funcid = 0x44
    global ctrlid
    ctrlid = 0x04
    try:
        x1_list = deal_params(x1)
        y1_list = deal_params(y1)
        z1_list = deal_params(z1)
        txPacket.extend(x1_list)
        txPacket.extend(y1_list)
        txPacket.extend(z1_list)
        x2_list = deal_params(x2)
        y2_list = deal_params(y2)
        z2_list = deal_params(z2)
        txPacket.extend(x2_list)
        txPacket.extend(y2_list)
        txPacket.extend(z2_list)
        leng = len(txPacket)
        txPacket.insert(0, leng)
        send_packet(txPacket)

    except:
        if timer:
            timer.cancel()
        print('参数有误')


# 多点运动
def ponits_movement(*args):
    try:
        params_list = list(args)
        pointnum = int(params_list[0])
        txPacket = [0x45, 0x04, pointnum]
        points_list = []
        if 0 < pointnum <= 10:
            for i in range(1, len(params_list), 8):
                points_list.append(params_list[i:i + 8])
            for point in points_list:
                spe1 = deal_params(point[0])
                movetype1 = int(point[1])
                x1 = deal_params(point[2])
                y1 = deal_params(point[3])
                z1 = deal_params(point[4])
                x2 = deal_params(point[5])
                y2 = deal_params(point[6])
                z2 = deal_params(point[7])
                txPacket.extend(spe1)
                txPacket.append(movetype1)
                txPacket.extend(x1)
                txPacket.extend(y1)
                txPacket.extend(z1)
                txPacket.extend(x2)
                txPacket.extend(y2)
                txPacket.extend(z2)
            leng = len(txPacket)
            txPacket.insert(0, leng)
            global funcid
            funcid = 0x45
            global ctrlid
            ctrlid = 0x04
            send_packet(txPacket)
        else:
            print("过程点不能超过10个")
    except:
        print("参数有误")


# 关节运动
def joint_movement(x1, y1, z1):
    point_movement(0x01, x1, y1, z1, 0, 0, 0)


# 自由运动
def free_movement(x1, y1, z1):
    point_movement(0x02, x1, y1, z1, 0, 0, 0)


# 直线运动
def linear_movement(x1, y1, z1):
    point_movement(0x03, x1, y1, z1, 0, 0, 0)


# 相对运动
def relative_movement(x1, y1, z1):
    point_movement(0x04, x1, y1, z1, 0, 0, 0)


# 门型运动
def portal_movement(x1, y1, z1, x2):
    point_movement(0x05, x1, y1, z1, x2, 0, 0)


# 圆弧运动
def circular_movement(x1, y1, z1, x2, y2, z2):
    point_movement(0x06, x1, y1, z1, x2, y2, z2)


# 自由坐标运动
def free_coordinate_movement(x1, y1, z1):
    point_movement(0x07, x1, y1, z1, 0, 0, 0)


# 绘制单条路径
def single_path(*args):
    pathflag = args[0]
    params_list = list(args)[1:]
    txPacket = [0x46, 0x04, pathflag]
    global funcid
    funcid = 0x46
    global ctrlid
    ctrlid = 0x04
    if params_list == [""] and pathflag != 3:
        print("缺少参数")
    else:
        try:
            if pathflag == 1:
                global pathtype
                pathtype = int(params_list[0])
                pathsn = '{:04X}'.format(int(params_list[1]))
                x = deal_params(params_list[2])
                y = deal_params(params_list[3])
                z = deal_params(params_list[4])
                txPacket.extend([pathtype, int(pathsn[:2]), int(pathsn[2:4])])
                txPacket.extend(x)
                txPacket.extend(y)
                txPacket.extend(z)
                leng = len(txPacket)
                txPacket.insert(0, leng)
                send_packet(txPacket)
            if pathflag == 2:
                pointnum = int(params_list[0])
                if 0 <= pointnum <= 19:
                    points_list = []
                    if 0 < pointnum < 10:
                        # print(params_list)
                        for i in range(1, len(params_list), 4):
                            points_list.append(params_list[i:i + 4])
                    txPacket.append(pointnum)
                    for point in points_list:
                        # print(point)
                        x = deal_params(point[0])
                        y = deal_params(point[1])
                        z = deal_params(point[2])
                        pointattr = int(point[3])
                        if pathtype == 0 or pathtype == 2:
                            if pointattr != 0:
                                print("pointattr值应该为0")
                            else:
                                txPacket.extend(x)
                                txPacket.extend(y)
                                txPacket.extend(z)
                                txPacket.append(pointattr)
                        elif pathtype == 1:
                            if pointattr < 1 or pointattr > 5:
                                print("pointattr范围在1-5")
                            else:
                                txPacket.extend(x)
                                txPacket.extend(y)
                                txPacket.extend(z)
                                txPacket.append(pointattr)
                        elif pathtype == 3 or pathtype == 4:
                            if pointattr < 1 or pointattr > 100:
                                print("pointattr范围在1-100")
                            else:
                                txPacket.extend(x)
                                txPacket.extend(y)
                                txPacket.extend(z)
                                txPacket.append(pointattr)
                        elif pathtype == None:
                            print("没有设置起始点")
                            return
                    leng = len(txPacket)
                    txPacket.insert(0, leng)
                    send_packet(txPacket)
                else:
                    print("过程点不能超过19个")
            if pathflag == 3:
                txPacket = [0x03, 0x46, 0x04, pathflag]
                send_packet(txPacket)
        except:
            print("参数有误")


# 控制面板
def control_panel(moveflag, movetype):
    try:
        txPacket = [0x89, 0x03, moveflag, movetype]
        global funcid
        funcid = 0x89
        global ctrlid
        ctrlid = 0x03
        leng = len(txPacket)
        txPacket.insert(0, leng)
        send_packet(txPacket)
    except:
        if timer:
            timer.cancel()
        print("参数有误")


# 断开连接
def close_port():
    ping(0x02)
    global timer
    if timer:
        timer.cancel()
    if sers:
        sers.close()


# 解析反馈包
def deal_fbk(rxpacket):
    # ping 反馈
    if rxpacket[INST_FUNCID] == 0x00 and rxpacket[INST_CTRL] == 0x01:
        strs = error_dict.get(rxpacket[5])
        return strs

    # 固件升级
    elif rxpacket[INST_FUNCID] == 0x01 and rxpacket[INST_CTRL] == 0x03:
        if rxpacket[5] == 0x00:
            return "指令正常执行"
        elif rxpacket[5] == 0x30:
            return "程序处于非空闲状态无法进行固件升级"
        elif rxpacket[5] == 0x31:
            return "建立固件文件失败, 无法进行固件升级"

    elif rxpacket[INST_FUNCID] == 0x02 and rxpacket[INST_CTRL] == 0x03:
        if rxpacket[5] == 0x00:
            return "成功执行"
        elif rxpacket[5] == 0x32:
            return "程序运行状态改变,无法继续进行固件更新"
        elif rxpacket[5] == 0x33:
            return "数据序号不匹配, 无法继续进行固件更新"
        elif rxpacket[5] == 0x34:
            return "固件写入失败"
        elif rxpacket[5] == 0x35:
            return "固件更名失败"
    elif rxpacket[INST_FUNCID] == 0x03 and rxpacket[INST_CTRL] == 0x03:
        if rxpacket[5] == 0x00:
            return "指令正常执行"

    # 读取当前版本
    elif rxpacket[INST_FUNCID] == 0x02 and rxpacket[INST_CTRL] == 0x02:
        vcode_str = '{:02X}'.format(rxpacket[6]) + '{:02X}'.format(rxpacket[7])
        vscode = int(vcode_str, 16)
        prevsion_str = str(rxpacket[8]) + '.' + str(rxpacket[9]) + '.' + str(rxpacket[10])
        return "当前版本编码{0}，当前与 PC 端对应的版本编号{1}".format(vscode, prevsion_str)

    # 软件关机
    elif rxpacket[INST_FUNCID] == 0x04 and rxpacket[INST_CTRL] == 0x03:
        strs = error_dict.get(rxpacket[5])
        return strs

    # 读取机械臂种类
    elif rxpacket[INST_FUNCID] == 0x05 and rxpacket[INST_CTRL] == 0x02:
        if rxpacket[5] != 0:
            return error_dict.get(rxpacket[5])
        else:
            if rxpacket[6] == 0x01:
                return "机械臂种类为 Brobot Delta"
            elif rxpacket[6] == 0x10:
                return "机械臂种类为 Brobot 四轴 Smart"
            elif rxpacket[6] == 0x11:
                return "机械臂种类为 Brobot 拼装四轴 Alpha"
            elif rxpacket[6] == 0x20:
                return "机械臂种类为 Scarat"

    # 设置电流阈值
    elif rxpacket[INST_FUNCID] == 0x3c and rxpacket[INST_CTRL] == 0x03:
        strs = error_dict.get(rxpacket[5])
        return strs

    # 设置回零
    elif rxpacket[INST_FUNCID] == 0x78 and rxpacket[INST_CTRL] == 0x03:
        if rxpacket[5] == 0x00:
            return "设置成功"
        elif rxpacket[5] == 0x31:
            return "设置的回零位置越界, 无法写入"

    # 执行回零
    elif rxpacket[INST_FUNCID] == 0x79 and rxpacket[INST_CTRL] == 0x04:
        strs = error_dict.get(rxpacket[5])
        return strs

    # 紧急停止
    elif rxpacket[INST_FUNCID] == 0x79 and rxpacket[INST_CTRL] == 0x03:
        strs = error_dict.get(rxpacket[5])
        return strs

    # 附件回零
    elif rxpacket[INST_FUNCID] == 0x7A and rxpacket[INST_CTRL] == 0x04:
        strs = error_dict.get(rxpacket[5])
        return strs

    # 校准操作
    elif rxpacket[INST_FUNCID] == 0x7D and rxpacket[INST_CTRL] == 0x03:
        if rxpacket[5] == 0x00:
            return "指令执行成功"
        elif rxpacket[5] == 0x30:
            return "机械臂处于非空闲状态,无法进入校准状态"
        elif rxpacket[5] == 0x31:
            return "机械臂处于非校准状态,无法校准"
        elif rxpacket[5] == 0x32:
            return "机械臂校准失败"
        elif rxpacket[5] == 0x33:
            return "机械臂校准指令有误"

    # 扫描附件ID
    elif rxpacket[INST_FUNCID] == 0x83 and rxpacket[INST_CTRL] == 0x02:
        if rxpacket[5] == 0x00:
            num = rxpacket[6]
            id1 = rxpacket[7]
            id2 = rxpacket[8]
            id3 = rxpacket[9]
            global enclosure_id_list
            enclosure_id_list = [id1, id2, id3]
            return "附件有{0}个，附件为：{1}({2}),{3}({4}),{5}({6})".format(num, id1, accessid_dic2.get(id1), id2,
                                                                   accessid_dic2.get(id2), id3,
                                                                   accessid_dic2.get(id3))

    # 设置或读取附件 char/float 型参数
    elif rxpacket[INST_FUNCID] == 0xBE:
        num = 0
        try:
            error_num = rxpacket[5]
            accessid = rxpacket[6]
            funcid = rxpacket[7]
        except:
            return "不存在该附件"
        if error_num == 4:
            return "指令执行失败"
        elif error_num == 3:
            return "表示未知指令, 无法响应"
        elif error_num == 2:
            return "队列指令仓库已满, 无法响应"
        else:
            if 8 <= accessid <= 15:
                num = 1
            elif 16 <= accessid <= 23:
                num = 2
            elif 24 <= accessid <= 31:
                num = 3
            elif 32 <= accessid <= 39:
                num = 4
            elif 40 <= accessid <= 47:
                num = 5
            elif 48 <= accessid <= 55:
                num = 6
            elif 56 <= accessid <= 63:
                num = 7
            elif 80 <= accessid <= 87:
                num = 8
            elif 128 <= accessid <= 135:
                num = 9
            elif 136 <= accessid <= 147:
                num = 10
            elif 192 <= accessid <= 199:
                num = 11
            elif 64 <= accessid <= 71:
                num = 12
            if rxpacket[INST_CTRL] == 0x03:
                return "AccessID：{0}，FuncID：{1}".format(accessid_dic.get(num),
                                                        set_funcid_dic.get(num).get(funcid))
            elif rxpacket[INST_CTRL] == 0x02:
                if num == 1:
                    value = rxpacket[8]
                    if funcid == 2:
                        mess = return_message2(num, funcid, value, value_dict1)
                        return mess

                    elif funcid == 3:
                        strs = get_pre2(8, rxpacket)
                        return "AccessID：{0}，FuncID：{1},Value:{2}".format(accessid_dic.get(num),
                                                                          read_funcid_dic.get(num).get(funcid),
                                                                          strs)
                elif num == 4:
                    value = rxpacket[8]
                    if funcid == 2:
                        mess = return_message2(num, funcid, value, value_dict1)
                        return mess
                    elif funcid == 3:
                        return "AccessID：{0}，FuncID：{1},Value:{2}号笔".format(accessid_dic.get(num),
                                                                            read_funcid_dic.get(num).get(
                                                                                funcid),
                                                                            value)
                elif num == 5:
                    value = rxpacket[8]
                    if funcid == 1 or funcid == 3:
                        mess = return_message1(num, funcid, value)
                        return mess
                    elif funcid == 2:
                        mess = return_message2(num, funcid, value, value_dict2)
                        return mess
                elif num == 6:
                    value = rxpacket[8]
                    if funcid == 1 or funcid == 3:
                        mess = return_message1(num, funcid, value)
                        return mess
                    elif funcid == 2:
                        mess = return_message2(num, funcid, value, value_dict1)
                        return mess
                elif num == 8:
                    if funcid == 2 or funcid == 3 or funcid == 4:
                        value = rxpacket[8]
                        mess = return_message1(num, funcid, value)
                        return mess
                    elif funcid == 5:
                        return "AccessID：{0}，FuncID：{1}".format(accessid_dic.get(num),
                                                                read_funcid_dic.get(num).get(funcid))
                elif num == 9:
                    if funcid == 2:
                        value = rxpacket[8]
                        mess = return_message2(num, funcid, value, value_dict3)
                        return mess
                    elif funcid == 3:
                        strs = get_pre2(8, rxpacket)
                        return "AccessID：{0}，FuncID：{1},Value:{2}".format(accessid_dic.get(num),
                                                                          read_funcid_dic.get(num).get(funcid),
                                                                          strs)
                    elif funcid == 4:
                        value = rxpacket[8]
                        mess = return_message2(num, funcid, value, value_dict4)
                        return mess
                elif num == 10:
                    if funcid == 2:
                        strs = get_pre2(8, rxpacket)
                        return "AccessID：{0}，FuncID：{1},Value:{2}".format(accessid_dic.get(num),
                                                                          read_funcid_dic.get(num).get(funcid),
                                                                          strs)
                    elif funcid == 3:
                        value = rxpacket[8]
                        mess = return_message2(num, funcid, value, value_dict5)
                        return mess
                    elif funcid == 4:
                        value = rxpacket[8]
                        if value == 1:
                            return "AccessID：{0}，FuncID：{1},Value:回到零点".format(accessid_dic.get(num),
                                                                               read_funcid_dic.get(num).get(
                                                                                   funcid))
                    elif funcid == 5:
                        value = rxpacket[8]
                        mess = return_message2(num, funcid, value, value_dict6)
                        return mess
                    elif funcid == 6:
                        strs = get_pre2(8, rxpacket)
                        return "AccessID：{0}，FuncID：{1},Value:{2}mm".format(accessid_dic.get(num),
                                                                            read_funcid_dic.get(num).get(
                                                                                funcid),
                                                                            strs)
                    elif funcid == 7:
                        value = rxpacket[8]
                        mess = return_message2(num, funcid, value, value_dict7)
                        return mess

                elif num == 11:
                    if funcid == 2:
                        value = rxpacket[8]
                        mess = return_message2(num, funcid, value, value_dict1)
                        return mess
                    elif funcid == 3:
                        value = rxpacket[8]
                        mess = return_message2(num, funcid, value, value_dict8)
                        return mess
                    elif funcid == 4:
                        value = rxpacket[8]
                        mess = return_message2(num, funcid, value, value_dict9)
                        return mess
                    elif funcid == 5:
                        value = rxpacket[8]
                        mess = return_message2(num, funcid, value, value_dict10)
                        return mess
                    elif funcid == 6:
                        r_num = rxpacket[8]
                        g_num = rxpacket[9]
                        b_num = rxpacket[10]
                        return "AccessID：{0}，FuncID：{1},Value:{2},{3},{4}".format(accessid_dic.get(num),
                                                                                  read_funcid_dic.get(num).get(
                                                                                      funcid),
                                                                                  r_num, g_num,
                                                                                  b_num)

    # 控制气泵吸吹停
    elif rxpacket[INST_FUNCID] == 0xBF and rxpacket[INST_CTRL] == 0x03:
        if rxpacket[5] == 0x00:
            return "执行成功"
        elif rxpacket[5] == 0x04:
            return "执行失败"

    # 实时反馈协议
    elif rxpacket[INST_FUNCID] == 0x0A and rxpacket[INST_CTRL] == 0x50:
        RobotSta = rxpacket[5]
        Reuse = int('{:02X}'.format(rxpacket[30]) + '{:02X}'.format(rxpacket[31]), 16)
        preCur = int('{:02X}'.format(rxpacket[32]) + '{:02X}'.format(rxpacket[33]), 16)
        preTemp = int('{:02X}'.format(rxpacket[34]), 16)
        RealInfo = int('{:02X}'.format(rxpacket[35]), 16)
        pre_list = []
        li = [6, 10, 14, 18, 22, 26]
        for i in li:
            strs = get_pre2(i, rxpacket)
            pre_list.append(strs)
        robotsta_str = robotsta_dic.get(RobotSta)
        hight = RealInfo & 0xf0
        low = RealInfo & 0x0f
        hight_str = RealInfo_hight_dic.get(hight)
        low_str = RealInfo_low_dic.get(low)
        global callback
        if callback:
            callback(pre_list[0], pre_list[1], pre_list[2],pre_list[3], pre_list[4], pre_list[5],preCur,
                   preTemp)
        return "{0}, X：{1}  Y：{2}  Z：{3}  S1：{4}  S2：{5}  S3：{6}, 当前最大电流：{7}mA, 当前最大温度：{8}℃, {9}, {10}".\
            format(robotsta_str, pre_list[0], pre_list[1], pre_list[2],pre_list[3], pre_list[4], pre_list[5],preCur,
                   preTemp, hight_str, low_str)

    # 控制信号
    elif rxpacket[INST_FUNCID] == 0x41 and rxpacket[INST_CTRL] == 0x03:
        if rxpacket[5] == 0x00:
            return "指令接收成功"
        elif rxpacket[5] == 0x31:
            return "发送重复指令"
        elif rxpacket[5] == 0x32:
            return "指令逻辑错误, 无法执行该指令"
        elif rxpacket[5] == 0x33:
            return "程序目前处于异常状态, 无法执行该指令"
        elif rxpacket[5] == 0x34:
            return "不存在该条指令"

    # 设置运动速率
    elif rxpacket[INST_FUNCID] == 0x42 and rxpacket[INST_CTRL] == 0x03:
        if rxpacket[5] == 0x00:
            return "执行成功"
        elif rxpacket[5] == 0x31:
            return "数据越界,不在范围[0.01, 1.00]之间"

    # 设置抬笔高度
    elif rxpacket[INST_FUNCID] == 0x43 and rxpacket[INST_CTRL] == 0x03:
        if rxpacket[5] == 0x00:
            return "指令接收成功"
        elif rxpacket[5] == 0x31:
            return "数据越界, 设置失败"

    # 控制单点运动
    elif rxpacket[INST_FUNCID] == 0x44 and rxpacket[INST_CTRL] == 0x04:
        strs = error_dict.get(rxpacket[5])
        return strs

    # 控制多点运动
    elif rxpacket[INST_FUNCID] == 0x45 and rxpacket[INST_CTRL] == 0x04:
        strs = error_dict.get(rxpacket[5])
        return strs

    # 绘制单条路径
    elif rxpacket[INST_FUNCID] == 0x46 and rxpacket[INST_CTRL] == 0x04:
        strs = error_dict.get(rxpacket[5])
        return strs

    # 控制面板
    elif rxpacket[INST_FUNCID] == 0x89 and rxpacket[INST_CTRL] == 0x03:
        if rxpacket[5] == 0x00:
            return "设置成功"
        elif rxpacket[5] == 0x30:
            return "CP_MoveFlag 参数不对"
        elif rxpacket[5] == 0x31:
            return "CP_MoveType 高四位参数不对"
        elif rxpacket[5] == 0x32:
            return "CP_MoveType 低四位参数不对"


#  组装返回信息
def return_message1(num, funcid, value):
    return "AccessID：{0}，FuncID：{1},Value:{2}".format(accessid_dic.get(num),
                                                      read_funcid_dic.get(num).get(funcid), value)


def return_message2(num, funcid, value, dict):
    return "AccessID：{0}，FuncID：{1},Value:{2}".format(accessid_dic.get(num),
                                                      read_funcid_dic.get(num).get(funcid), dict.get(value))


def get_pre2(i, rxpacket):
    s = '{:02X}'.format(rxpacket[i]) + '{:02X}'.format(rxpacket[i + 1]) + '{:02X}'.format(rxpacket[i + 2]) + \
        '{:02X}'.format(rxpacket[i + 3])
    if int(s[:2], 16) == 0:
        f = ""
    else:
        f = "-"
    num1 = int(s[2:6], 16)
    num2 = int(s[6:8], 16)
    return "{0}{1}.{2}".format(f, num1, num2)


def get_current_info(state):
    global info_flag
    if state == 1:
        info_flag = 1
    elif state == 0:
        info_flag = 0


# 接收数据
def data_receive():
    global sers
    try:
        num = sers.inWaiting()
    except:
        return
    if num > 0:
        global flag
        global len_num
        if flag == 0:
            len_num = 3
        if sers:
            try:
                data = sers.read(len_num)
                num = len(data)
                global rxpack
                for i in range(0, len(data)):
                    rxpack.append(data[i])
                if len(rxpack) >= 3:
                    for idx in range(0, len(rxpack) - 2):
                        if (rxpack[idx] == 0xBB) and (rxpack[idx + 1] == 0xBB):
                            if idx == 0:
                                flag = 1
                                len_num = rxpack[2] + 4 - len(rxpack)
                                if len(rxpack) == rxpack[2] + 4:
                                    # 计算校验码
                                    checkSum = 0
                                    for i in range(3, len(rxpack) - 1):
                                        checkSum += rxpack[i]
                                    checkSum = (~checkSum & 0xFF) + 1
                                    if rxpack[-1] == checkSum:
                                        if rxpack[3] == 0x0A and rxpack[4] == 0x50:
                                            strs =deal_fbk(rxpack)
                                            global callback

                                            global info_flag
                                            if info_flag:
                                                print(strs)
                                        else:
                                            strs = deal_fbk(rxpack)
                                            print('{0}  {1}'.format(strs,rxpack))
                                            global funcid
                                            global ctrlid
                                            if rxpack[3] == funcid and rxpack[4] == ctrlid:
                                                global receive
                                                receive = True
                                        flag = 0
                                        rxpack = []
                                        break
                                    else:
                                        flag = 0
                                        rxpack = []
                                        break
                            elif idx > 0:
                                # 移除不需要的数据
                                del rxpack[0:idx]
                                flag = 0
                                break
                            else:
                                flag = 0
                                rxpack = []
                                break
            except Exception as e:
                print(e)
                if sers:
                    sers.close()

    global timer
    timer = threading.Timer(0.005, data_receive)
    timer.start()


class ReadFileThread(threading.Thread):

    def __init__(self, filename):
        super(ReadFileThread, self).__init__()
        global sers
        self.ser = sers
        self.filename = filename
        self.stop_th = 0

    def run(self):
        sn = 1
        try:
            with open(self.filename, "rb") as f:
                while True:
                    if self.stop_th:
                        print("机械臂断开连接")
                        break
                    txPacket = [0x02, 0x03]
                    data = f.read(241)
                    if not data:
                        break
                    data_len = len(data)
                    data_list = list(struct.unpack("{}B".format(data_len), data))
                    txPacket.extend(data_list)
                    txPacket.insert(2, data_len)
                    sn_str = '{:04X}'.format(sn)
                    txPacket.insert(3, int(sn_str[:2], 16))
                    txPacket.insert(4, int(sn_str[2:], 16))
                    txPacket.insert(0, len(txPacket))
                    global repeat_num
                    global receive
                    for i in range(repeat_num):
                        if not receive:
                            result = TxPacket(txPacket)
                            time.sleep(over_time)
                            if i == repeat_num - 1:
                                self.stop_th = 1
                        elif receive:
                            receive = False
                            break
                    sn += 1
                    time.sleep(0.001)
            txPacket = [0x05, 0x02, 0x03, 0x00, 0x00, 0x00]
            for i in range(repeat_num):
                if not receive:
                    result = TxPacket(txPacket)
                    time.sleep(over_time)
                elif receive:
                    receive = False
                    break
        except Exception as e:
            print(e)



# # 气泵动作
# def air_pump_action(action):
#     """
#     :param action: 气泵动作：0x00 停， 0x01 吸，0x02 吹
#     :return:
#     """
#     txPacket = [0x03, 0xBF, 0x03, action]
#     global funcid
#     funcid = 0xBF
#     global ctrlid
#     ctrlid = 0x03
#     send_packet(txPacket)
#
#
# # 设置抬笔高度
# def set_pen_height(height):
#     try:
#         if height < 0 or height > 30:
#             if timer:
#                 timer.cancel()
#             print('抬笔高度范围为0-30')
#         else:
#             txPacket = [0x03, 0x43, 0x03, height]
#             global funcid
#             funcid = 0x43
#             global ctrlid
#             ctrlid = 0x03
#             send_packet(txPacket)
#     except:
#         if timer:
#             timer.cancel()
#         print('参数有误')


