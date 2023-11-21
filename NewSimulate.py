import functools
import os
import pickle
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from statistics import mean

import numpy
import pandas
from matplotlib import pyplot as plt
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from DataBase import MySQLInterface
from FactorLibrary import price_slope_factor, reverse_factor, T_BS_link_factor, Y_premium_rate_factor, \
    reverse_factor_2


def try_except(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            exc_type, exc_instance, exc_traceback = sys.exc_info()
            formatted_traceback = ''.join(traceback.format_tb(exc_traceback))
            message = '\n{0} raise {1}:{2}'.format(
                formatted_traceback,
                exc_type.__name__,
                exc_instance
            )
            # raise exc_type(message)
            print(message)
            return None

    return wrapper


class HistoryDataProvider:
    def __init__(self):
        self.mysql_interface = MySQLInterface()

    def __get_start_date_exist_pair_dict(self, start_date):
        '''
        从开盘获得的比价表中得到目前正在交易的债券，结合上市日期筛选债券
        :return:
        '''
        bond_info_df = pandas.read_csv('D:\\bs_pair_info\%s_pair_info.csv' % (start_date[0:6]))
        # 筛选上市时间
        bond_info_df = bond_info_df[(bond_info_df['上市日期'] <= start_date)]
        return bond_info_df.set_index('转债代码')['正股代码'].to_dict()

    def load_cache(self, start_date, end_date):
        if (os.path.exists('D:\\bs_pair_info\\bs_cache_%s_%s.pickle' % (start_date, end_date))):
            with open('D:\\bs_pair_info\\bs_cache_%s_%s.pickle' % (start_date, end_date), 'rb') as f:
                package_list = pickle.load(f)
            return [True, package_list[0], package_list[1], package_list[2]]
        else:
            return [False]

    def store_cache(self, start_date, end_date, output_code_dict, output_bond_df_dict, output_stock_df_dict):
        with open('D:\\bs_pair_info\\bs_cache_%s_%s.pickle' % (start_date, end_date), 'wb') as f:
            pickle.dump([output_code_dict, output_bond_df_dict, output_stock_df_dict], f)
        print('成功缓存')
        return

    def get_database_df_dict(self, start_date, end_date, process_num=1):
        '''
        获取数据生成df_dict {code:[date1_df,date2_df}
        '''
        cache = self.load_cache(start_date, end_date)

        if (cache[0]):
            pair_info_df = pandas.read_csv('D:\\bs_pair_info\%s_pair_info.csv' % (start_date[0:6]))
            return cache[1], cache[2], cache[3],pair_info_df
        else:
            # 获取代码dict
            raw_code_dict = self.__get_start_date_exist_pair_dict(start_date)
            output_raw_code_dict = raw_code_dict.copy()

            # 初始化df_dict
            bond_df_dict = {i: [] for i in raw_code_dict.keys()}
            stock_df_dict = {i: [] for i in raw_code_dict.values()}
            # 验证长度
            df_length_dict = {i: 0 for i in (list(raw_code_dict.keys()) + list(raw_code_dict.values()))}
            pool = ThreadPoolExecutor(max_workers=process_num)

            def __get_mysql_bond_data(code_list):
                for code in code_list:
                    try:
                        temp_df = self.mysql_interface.interface_get_data(code, start_date, end_date)
                        df_length_dict[code] = len(temp_df)
                        temp_df.index = pandas.to_datetime(temp_df.index)
                        temp_inday_df_list = [group[1] for group in temp_df.groupby(temp_df.index.day)]
                        temp_inday_df_list.reverse()
                        bond_df_dict[code] = temp_inday_df_list
                    except Exception as e:
                        print(e)
                        del output_raw_code_dict[code]
                        print('无法找到', code, ' ', e)
                return

            def __get_mysql_stock_data(code_list):
                for code in code_list:
                    try:
                        temp_df = self.mysql_interface.interface_get_data(code, start_date, end_date)
                        df_length_dict[code] = len(temp_df)
                        temp_df.index = pandas.to_datetime(temp_df.index)
                        temp_inday_df_list = [group[1] for group in temp_df.groupby(temp_df.index.day)]
                        temp_inday_df_list.reverse()
                        stock_df_dict[code] = temp_inday_df_list
                    except Exception as e:
                        print(e)
                        del output_raw_code_dict[code]
                        print('无法找到', code, ' ', e)
                return

            for b_code_list in [list(raw_code_dict.keys())[i:i + 100] for i in
                                range(0, len(raw_code_dict.keys()), 100)]:
                pool.submit(__get_mysql_bond_data, (b_code_list))

            for s_code_list in [list(raw_code_dict.values())[i:i + 100] for i in
                                range(0, len(raw_code_dict.values()), 100)]:
                pool.submit(__get_mysql_stock_data, (s_code_list))

            pool.shutdown()

            majority_length = numpy.argmax(numpy.bincount(list(df_length_dict.values())))
            print(majority_length)
            output_code_dict = {k: v for k, v in output_raw_code_dict.items() if
                                (df_length_dict[k] == majority_length and df_length_dict[v] == majority_length)}
            output_bond_df_dict = {k: v for k, v in bond_df_dict.items() if k in output_code_dict.keys()}
            output_stock_df_dict = {k: v for k, v in stock_df_dict.items() if k in output_code_dict.values()}
            self.store_cache(start_date, end_date, output_code_dict, output_bond_df_dict, output_stock_df_dict)
            pair_info_df = pandas.read_csv('D:\\bs_pair_info\%s_pair_info.csv'%(start_date[0:6]))
            return output_code_dict, output_bond_df_dict, output_stock_df_dict, pair_info_df


class FactorValueProvider:
    '''
    输入：output_code_dict,output_bond_df_dict,output_stock_df_dict
    输出：预处理后的factor_df_list
    '''

    def __init__(self):
        pass

    def __get_factor_df_list(self, raw_code_dict, bond_df_dict, stock_df_dict,pair_info_df,factor_func, factor_N):
        factor_df_list = []
        if ((factor_func.__name__)[0:1] == 'T'):
            for i in range(0, len(bond_df_dict.get(next(iter(bond_df_dict))))):
                single_day_bond_df_dict = {k: v[i] for k, v in bond_df_dict.items()}
                single_day_stock_df_dict = {k: v[i] for k, v in stock_df_dict.items()}

                factor_df = factor_func(raw_code_dict, single_day_bond_df_dict, single_day_stock_df_dict, factor_N)
                factor_df_list.append(factor_df)
        elif((factor_func.__name__)[0:1] == 'Y'):
            for i in range(0, len(bond_df_dict.get(next(iter(bond_df_dict))))):
                single_day_bond_df_dict = {k: v[i] for k, v in bond_df_dict.items()}
                single_day_stock_df_dict = {k: v[i] for k, v in stock_df_dict.items()}

                factor_df = factor_func(raw_code_dict, single_day_bond_df_dict, single_day_stock_df_dict,pair_info_df,factor_N)
                factor_df_list.append(factor_df)
        else:
            for i in range(0, len(bond_df_dict.get(next(iter(bond_df_dict))))):
                single_day_df_dict = {k: v[i] for k, v in bond_df_dict.items()}

                factor_df = factor_func(single_day_df_dict, factor_N)
                factor_df_list.append(factor_df)
        return factor_df_list

    def __pre_handle_factor_data(self, factor_df):
        '''
        一级函数，预处理核心函数
        :return:
        '''

        def __del_and_replace(factor_df):
            '''
            二级函数，删除全空值，填充-in)f值
            :param factor_df:
            :return:
            '''
            factor_df = factor_df.dropna(how='all')
            factor_df = factor_df.replace(numpy.inf, 0)  # 替换正inf为0
            factor_df = factor_df.replace(-numpy.inf, 0)  # 替换-inf为0

            return factor_df

        def __replace_extreme_factor_value(factor_df):
            def __cal_new_value(row):
                median_value = row.median()
                mad = abs(row - median_value).median()
                upper_bound = median_value + 5 * mad
                lower_bound = median_value - 5 * mad
                row = row.where(row <= lower_bound, lower_bound)
                row = row.where(row >= upper_bound, upper_bound)
                std = row.std()
                if std == 0:
                    row[:] = 0
                else:
                    row = (row - row.mean()) / std
                return row

            factor_df.apply(__cal_new_value, axis=1)
            print('因子去极值，标准化完成')
            return factor_df

        factor_df = __del_and_replace(factor_df)
        factor_df = __replace_extreme_factor_value(factor_df)
        return factor_df

    def input_bond_factor_lib(self, raw_code_dict, bond_df_dict, stock_df_dict,pair_info_df,factor_func, factor_N):
        '''
        一级函数，输入因子
        :param start_date:
        :param end_date:
        :param factor_func:
        :return:
        '''

        '''
        获取数据生成因子
        '''
        raw_factor_df_list = self.__get_factor_df_list(raw_code_dict, bond_df_dict, stock_df_dict, pair_info_df, factor_func,
                                                       factor_N)
        '''
        预处理
        '''
        factor_df_list = []
        for raw_factor_df in raw_factor_df_list:
            factor_df = self.__pre_handle_factor_data(raw_factor_df)

            factor_df_list.append(factor_df)
        return factor_df_list


class SingleFactorTest:
    '''
    类作用：单因子测试
    输入：factor_df_list,bond_df_dict,pos_N
    输出：

    '''

    def __init__(self):
        pass

    def __single_day_single_factor_test(self, df_dict, factor_df, pos_N):
        def __get_close_df(df_dict):
            # 计算close_df
            close_df = pandas.DataFrame()
            for code, df in df_dict.items():
                temp_series = df['收盘']
                temp_series.rename(code, inplace=True)
                close_df = pandas.concat([close_df, temp_series], axis=1)
            close_df = close_df.dropna(how='all')

            return close_df

        def __get_only_bond_IC_df(close_df, factor_df, pos_N):
            '''
            计算IC,分组IC,超额收益
            :param close_df:
            :param factor_df:
            :param pos_N:
            :return:
            '''




            def cal_group_IC_profit(factor_series,in_transfer_close_df,n):

                factor_sorted_code_list = factor_series.sort_values().index.tolist()
                layer_code_list = [factor_sorted_code_list[i:i + n] for i in range(0, len(factor_sorted_code_list), n)]
                #获取在in_transfer_close_df中的index:num_index
                index = factor_series.name
                index_list = in_transfer_close_df.index.tolist()
                num_index = index_list.index(index)
                #初始化填充的表
                factor_group_array_list = []
                pctChg_group_array_list = []
                t0_series = in_transfer_close_df.loc[index,:]
                t1_series = in_transfer_close_df.iloc[num_index+1,:]
                pctchg_series = (t1_series - t0_series)/t0_series

                for s_list in layer_code_list:
                    factor_value = numpy.array(factor_series[s_list].values)
                    pctchg_value = numpy.array(pctchg_series[s_list].values)
                    pctChg_group_array_list.append(pctchg_value)
                    factor_group_array_list.append(factor_value)

                factor_group_array_1_9_list = factor_group_array_list[:-1]
                factor_group_array_10_list = factor_group_array_list[-1]
                pctChg_group_array_1_9_list = pctChg_group_array_list[:-1]
                pctChg_group_array_10_list = pctChg_group_array_list[-1]
                #升维
                factor_group_array_1_9_list = numpy.array(factor_group_array_1_9_list)
                factor_group_array_10_list = numpy.array(factor_group_array_10_list)
                pctChg_group_array_1_9_list = numpy.array(pctChg_group_array_1_9_list)
                pctChg_group_array_10_list = numpy.array(pctChg_group_array_10_list)



                #排序
                argsort_factor_array = numpy.argsort(factor_group_array_1_9_list, axis=1)  # 按行排序
                argsort_pctchg_array = numpy.argsort(pctChg_group_array_1_9_list, axis=1)  # 按行排序

                argsort_factor_10_array = numpy.argsort(factor_group_array_10_list)
                argsort_pctchg_10_array = numpy.argsort(pctChg_group_array_10_list)  # 按行排序
                argsort_base_factor_array = numpy.argsort(numpy.array(factor_series[factor_sorted_code_list].values))
                argsort_base_pctchg_array = numpy.argsort(numpy.array(pctchg_series[factor_sorted_code_list].values))
                #bug记录：只有在factor_series和pctchg_series后加上同一个顺序才会获得正确结果

                corr_pearson_array = numpy.diagonal(numpy.corrcoef(factor_group_array_1_9_list, pctChg_group_array_1_9_list)[:9, 9:])
                corr_spearman_array = numpy.diagonal(
                    numpy.corrcoef(argsort_factor_array, argsort_pctchg_array)[:9, 9:])
                result_dict = {'IC%s'%(k):corr_pearson_array[k] for k in range(0,9)}
                result_dict.update({'RANK_IC%s'%(k):corr_spearman_array[k] for k in range(0,9)})
                result_dict.update({'IC9':numpy.corrcoef(factor_group_array_list[-1],pctChg_group_array_list[-1])[0,1],
                                    'RANK_IC9':numpy.corrcoef(argsort_factor_10_array,argsort_pctchg_10_array)[0,1]})
                result_dict.update({'profit%s'%(k):numpy.sum(pctChg_group_array_list[k])/len(pctChg_group_array_list[k])-0.0001 for k in range(0,10)})

                result_dict.update({'IC_C':numpy.corrcoef(numpy.array(factor_series.values),numpy.array(pctchg_series[factor_series.index].values))[0,1],
                                    'RANK_IC_C': numpy.corrcoef(argsort_base_factor_array,argsort_base_pctchg_array)[0,1],
                                    'profit_C':pctchg_series.sum()/len(pctchg_series)-0.0001})

                return pandas.Series(result_dict,name=index)

            def cal_divide_part_num(in_transfer_factor_df):
                length = len(in_transfer_factor_df.columns)
                if (length % 10 == 0):
                    n = int(length / 10)
                else:
                    n = int(length / 10) + 1
                return n

            # 调整close_df的index和factor_df相同
            close_df = close_df.loc[factor_df.index[0]:, :]
            in_transfer_factor_df = factor_df.iloc[range(0,len(factor_df)-pos_N,pos_N),:]
            in_transfer_close_df = close_df.iloc[range(0,len(factor_df),pos_N),:]
            #计算分层每层需要的数量
            divide_part_num = cal_divide_part_num(in_transfer_factor_df)
            #计算分层IC/收益

            IC_profit_df = in_transfer_factor_df.apply(lambda x:cal_group_IC_profit(x,in_transfer_close_df,divide_part_num),axis=1)
            IC_df = IC_profit_df[['IC%s'%(i) for i in range(0,10)]+['RANK_IC%s'%(i) for i in range(0,10)]+['IC_C','RANK_IC_C']]
            profit_df = IC_profit_df[['profit%s'%(i) for i in range(0,10)]+['profit_C']]
            return IC_df, profit_df

        close_df = __get_close_df(df_dict)
        IC_df, profit_df = __get_only_bond_IC_df(close_df, factor_df, pos_N)
        IC_df = IC_df.astype('float')
        profit_df = profit_df.astype('float')
        return IC_df, profit_df

    def multiday_single_factor_result_output(self, df_dict, factor_df_list, pos_N):
        '''

        :param test_result_dict: {code:[[IC_df,acc_profit_df]}
        :return:
        '''

        def __get_IC_and_profit_concat_df(df_dict, factor_df_list, pos_N):
            IC_df = pandas.DataFrame()
            profit_df = pandas.DataFrame()
            for i in range(0, len(factor_df_list)):
                single_day_df_dict = {k: v[i] for k, v in df_dict.items()}
                inday_IC_df, inday_profit_df = self.__single_day_single_factor_test(single_day_df_dict,
                                                                                    factor_df_list[i], pos_N)
                IC_df = pandas.concat([IC_df, inday_IC_df], axis=0)
                # 获得profit_df值
                profit_df = pandas.concat([profit_df, inday_profit_df], axis=0)
            return IC_df, profit_df

        def __get_accumulated_IC_and_profit_df(IC_df, profit_df):
            for col in IC_df.columns:
                IC_df['acc' + col] = IC_df[col].cumsum()
            acc_IC_df = IC_df.loc[:, IC_df.columns.str.contains('acc')]

            for col in profit_df.columns:
                profit_df['acc' + col] = profit_df[col].cumsum()
            acc_profit_df = profit_df.loc[:, profit_df.columns.str.contains('acc')]
            return acc_IC_df, acc_profit_df

        def __get_end_acc_profit(acc_profit_df):
            end_acc_profit = {}
            for col in acc_profit_df.columns:
                end_acc_profit['Group' + col[-1]] = acc_profit_df[col].values[-1]
            profit_df = pandas.DataFrame(end_acc_profit, index=['profit'])
            return profit_df.T

        def __get_sharp_rate(acc_profit_df):
            sharp_dict = {}
            for col in acc_profit_df.columns:
                sharp_dict['Group' + col[-1]] = acc_profit_df[col].values[-1] / acc_profit_df[col].std()
            sharp_df = pandas.DataFrame(sharp_dict, index=['sharp_rate'])
            return sharp_df.T

        def __get_group_IR(IC_df):
            IR_dict = {}
            for col in IC_df.columns[IC_df.columns.str.startswith('RANK')]:
                IR_dict['Group' + col[-1]] = IC_df[col].sum()/len(IC_df[col]) / IC_df[col].std()
            IR_df = pandas.DataFrame(IR_dict, index=['IR'])
            return IR_df.T

        def __get_group_IC_mean(IC_df):
            IC_mean_dict = {}

            for col in IC_df.columns[IC_df.columns.str.startswith('RANK')]:
                IC_mean_dict['Group' + col[-1]] = IC_df[col].sum()/len(IC_df[col])
            IC_mean_df = pandas.DataFrame(IC_mean_dict, index=['IC_mean'])
            return IC_mean_df.T

        def __get_group_IC_std(IC_df):
            IC_std_dict = {}
            for col in IC_df.columns[IC_df.columns.str.startswith('RANK')]:
                IC_std_dict['Group' + col[-1]] = IC_df[col].std()
            IC_std_df = pandas.DataFrame(IC_std_dict, index=['IC_std'])
            return IC_std_df.T

        IC_df, profit_df = __get_IC_and_profit_concat_df(df_dict, factor_df_list, pos_N)
        acc_IC_df, acc_profit_df = __get_accumulated_IC_and_profit_df(IC_df, profit_df)
        # 计算IC特征值表，包含分组IC均值，std，IR
        IC_mean_df = __get_group_IC_mean(IC_df)
        IC_std_df = __get_group_IC_std(IC_df)
        IC_IR_df = __get_group_IR(IC_df)
        # 计算profit特征值表，包含end_profit，sharp_rate
        profit_end_df = __get_end_acc_profit(acc_profit_df)
        profit_sharp_df = __get_sharp_rate(acc_profit_df)
        # 聚合所有df为一张表.01
        factor_test_df = pandas.concat([IC_mean_df, IC_std_df, IC_IR_df, profit_end_df, profit_sharp_df], axis=1)
        output_figure_acc_profit_df = acc_profit_df.drop(['accprofit_C'], axis=1)
        output_figure_acc_profit_df.index = output_figure_acc_profit_df.index.astype('str')
        output_figure_acc_profit_df.plot()
        plt.show()
        return acc_IC_df, acc_profit_df, factor_test_df


class AssessFactor:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.__get_history_data()

    def __get_history_data(self):
        self.output_code_dict, self.output_bond_df_dict, self.output_stock_df_dict,self.pair_info_df = HistoryDataProvider().get_database_df_dict(
            self.start_date, self.end_date)
        return self

    def get_factor_test_result(self, factor_func, factor_N, pos_N):
        factor_df_list = FactorValueProvider().input_bond_factor_lib(self.output_code_dict, self.output_bond_df_dict,
                                                                     self.output_stock_df_dict,self.pair_info_df, factor_func, factor_N)
        acc_IC_df, acc_profit_df, factor_test_df = SingleFactorTest().multiday_single_factor_result_output(
            self.output_bond_df_dict, factor_df_list, pos_N)
        print(factor_test_df)
        return factor_test_df

    def testify_param(self, factor_func):
        '''
        自动确定最优的参数，目标变量：累计收益率
        :return:
        '''
        factor_N_list = range(2, 21)
        for factor_N in factor_N_list:
            pos_N_list = range(1, factor_N)
            factor_df_list = FactorValueProvider().input_bond_factor_lib(self.output_code_dict,
                                                                         self.output_bond_df_dict,
                                                                         self.output_stock_df_dict,self.pair_info_df, factor_func,
                                                                         factor_N)
            for pos_N in pos_N_list:
                    acc_IC_df, acc_profit_df, factor_test_df = SingleFactorTest().multiday_single_factor_result_output(
                        self.output_bond_df_dict, factor_df_list, pos_N)
                    print(factor_test_df)
                    temp_df = pandas.DataFrame([{'IC_mean': factor_test_df.loc['GroupC', 'IC_mean'],
                                                 'IC_std': factor_test_df.loc['GroupC', 'IC_std'],
                                                 'IR': factor_test_df.loc['GroupC', 'IR'],
                                                 'Group9_profit': factor_test_df.loc['Group9', 'profit'],
                                                 'Group9_sharp_rate': factor_test_df.loc['Group9', 'sharp_rate'],
                                                 'L3group_ave_profit': mean([factor_test_df.loc['Group9', 'profit'],
                                                                             factor_test_df.loc['Group8', 'profit'],
                                                                             factor_test_df.loc['Group7', 'profit']]),
                                                 'L3group_ave_sharp_rate': mean(
                                                     [factor_test_df.loc['Group9', 'sharp_rate'],
                                                      factor_test_df.loc['Group8', 'sharp_rate'],
                                                      factor_test_df.loc['Group7', 'sharp_rate']])}],
                                               index=[factor_func.__name__ + '_' + str(factor_N) + '_' + str(pos_N)],
                                               columns=['IC_mean', 'IC_std', 'IR', 'Group9_profit', 'Group9_sharp_rate',
                                                        'L3group_ave_profit', 'L3group_ave_sharp_rate'])

                    temp_df.to_csv('D:\\bs_pair_info\AssessFactor\\%s\\%s\\%s_params_test.csv' % (
                    factor_func.__name__, self.start_date[0:6], factor_func.__name__), encoding='utf-8-sig', mode='a')
                    print('已完成一次')



# A1 = AssessFactor('20230801', '20230831')
# A1.testify_param(reverse_factor)
# A1.testify_param(price_slope_factor)
# A1.testify_param(T_BS_link_factor)
# A1 = AssessFactor('20230901', '20230930')
# A1.testify_param(reverse_factor)
# A1.testify_param(price_slope_factor)
# A1.testify_param(T_BS_link_factor)
# AssessFactor('20230801','20230831').get_factor_test_result(reverse_factor,13,2)

