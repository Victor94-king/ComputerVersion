import pandas as pd
import numpy as np  
import os
from easygui import *
import re


# thershold = integerbox('请输入您要修改的zos范围，超过多少百分比需要修改,建议60') / 100
thershold = 0.6
choices_list = ('40', '50', '90', '210', '250')
sheet_name  = choicebox(msg='请选择需要工作的OP', title= '工作OP', choices=choices_list)
zos_file = fileopenbox(msg='请选择你要修改的ZOS文件', title='excel', default='*', filetypes=None, multiple=False)

def deal_df(x):
    '''有的数值只是用来reference'''
    x = 0 if x > 10 else x
    return x

def find_psd(x):
    if 'KOX' in x:
        return True
    elif 'KOY' in x:
        return True
    elif 'KOZ' in x:
        return True
    elif 'PSD' in x:
        return True
    return True if "PSD" in x  or 'PSY' in x else False

def find_element(ele_list , zuobiaoxi):
    zidian = {}
    for axis in 'XYZ': ## 三坐标遍历
        axis_list = []
        for ele in ele_list: #元素遍历
            pattern = f"{ele}\d*_KO" + axis
            measure_list = [col for col in df['Characteristic'] if re.match(pattern, col)]
            if measure_list:
                axis_list.append(np.median([float(df[df['Characteristic'] == element]['val']) for element in measure_list]))
        zidian[axis] = round(float(np.mean(axis_list)),3)

    return zidian
df = pd.read_excel(zos_file)
def main():
    refer = pd.read_excel('MOPF Cfine调整计算表.xlsx', sheet_name=f'OP{sheet_name}')
    df['element'] = df['Characteristic'].map(lambda x:x[:4])
    df['val'] = df.iloc[:,7].abs() - df.iloc[:,4].abs()
    df['val'] = df['val'].map(lambda x :deal_df(x))
    refer = refer.drop(0 , axis=0 )
    refer['元素号'] = refer['元素号'].astype(str)
    df_PSD = df[df['Characteristic'].map(lambda x:find_psd(x))].sort_values('%', ascending=False) ##找到位置度相关的指标
    df_zos = df_PSD[df_PSD['%'] > thershold].groupby('element').mean().index.to_list() #找到超过阈值的指标
    visited_zuobiaoxi = []

    if len(df_zos) == 0:
        msgbox(f"没有需要更改的元素")

    id = 0
    for i ,ele in enumerate(df_zos) :
        zuobiaoxi  = refer[refer['元素号'] == ele]['坐标系'].values[0]
        if zuobiaoxi in visited_zuobiaoxi:
            continue
        visited_zuobiaoxi.append(zuobiaoxi)
        ele_list = refer[refer['坐标系']  == zuobiaoxi]['元素号'].to_list() #在这个坐标系下的所有元素
        zidian = find_element(ele , zuobiaoxi)
        id += 1
        msgbox(f" \n 需要修改的第{id}元素: {ele} \n 修改的坐标系为:    {zuobiaoxi} \n 对应修改的值为{zidian}")
        print('\n')


if __name__ == '__main__':
    main()