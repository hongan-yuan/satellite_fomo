# -*- coding: utf-8 -*-
"""
@Time : 
@Author: Honggang Yuan
@Email: hn_yuanhg@163.com
Description:
    
"""


def count_visible_soldiers(n, heights):
    # 初始化可见士兵的数量和目前为止见到的最高士兵的身高
    visible_count = 0
    max_height_so_far = 0
    for height in heights:
        # 如果当前士兵的身高大于之前所有士兵的最高身高，则将军能看到他
        if height > max_height_so_far:
            visible_count += 1
            max_height_so_far = height
    return visible_count


T = int(input().strip())  # 读取测试数据组数
for _ in range(T):
    s_n = int(input().strip())  # 每组数据中的士兵数
    s_heights = list(map(int, input().strip().split()))  # 士兵的身高列表
    print(count_visible_soldiers(s_n, s_heights))
