# Public Lib
import os
import sys
import time

from typing import Optional

text_colors = {"logs": "\033[34m", "info": "\033[32m", "warning": "\033[33m", "debug": "\033[93m", "error": "\033[31m", "bold": "\033[1m", "end_color": "\033[0m", "light_red": "\033[36m"}


def get_curr_time_stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")

'''
- python function (->) 함수의 리턴 값에 대한 주석을 의미한다.
- python function (message: str) 함수의 매개 변수 타입에 대한 주석을 의미한다.
'''

def error(message: str):
    time_stamp = get_curr_time_stamp()
    error_str = (text_colors["error"] + text_colors["bold"] + "ERROR " + text_colors["end_color"])
    print("{} - {} - {}".format(time_stamp, error_str, message), flush=True)
    print("{} - {} - {}".format(time_stamp, error_str, "Exiting!!!"), flush=True)
    exit(-1)

'''
- python print() 함수의 flush 인수를 True 설정하여 함수가 출력 데이터를 버퍼링 하는 것을 중지하고 강제로 플러시 할 수 있다.
- exit(-1) 프로그램이 오류로 인해 정상 종료 되지 않았음을 뜻하고 그 값은 -1 반환한다.
'''

def color_text(in_text: str):
    return text_colors["light_red"] + in_text + text_colors["end_color"]


def log(message: str):
    time_stamp = get_curr_time_stamp()
    log_str = (text_colors["logs"] + text_colors["bold"] + "LOGS " + text_colors["end_color"])
    print("{} - {} - {}".format(time_stamp, log_str, message))


def warning(message: str):
    time_stamp = get_curr_time_stamp()
    warn_str = (text_colors["warning"] + text_colors["bold"] + "WARNING" + text_colors["end_color"])
    print("{} - {} - {}".format(time_stamp, warn_str, message))


def info(message: str, print_line: Optional[bool] = False):
    time_stamp = get_curr_time_stamp()
    info_str = (text_colors["info"] + text_colors["bold"] + "INFO " + text_colors["end_color"])
    print("{} - {} - {}".format(time_stamp, info_str, message))
    if print_line:
        double_dash_line(dashes=150)

'''
- python typing 모듈의 Optional은 None이 허용되는 함수의 매개 변수에 대한 타입을 명시할 때 유용하다.
'''

def debug(message: str):
    time_stamp = get_curr_time_stamp()
    log_str = (text_colors["debug"] + text_colors["bold"] + "DEBUG " + text_colors["end_color"])
    print("{} - {} - {}".format(time_stamp, log_str, message))


def double_dash_line(dashes: Optional[int] = 75):
    print(text_colors["error"] + "=" * dashes + text_colors["end_color"])


def singe_dash_line(dashes: Optional[int] = 67):
    print("-" * dashes)


def print_header(header: str):
    double_dash_line()
    print(text_colors["info"] + text_colors["bold"] + "=" * 50 + str(header) + text_colors["end_color"])
    double_dash_line()


def disable_printing():
    sys.stdout = open(os.devnull, "w")


def enable_printing():
    sys.stdout = sys.__stdout__

    

