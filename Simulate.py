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
from FactorLibrary import price_slope_factor, reverse_factor, T_BS_link_factor


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
        bond_info_df = pandas.read_csv('D:\\bs_pair_info\%s_pair_info.csv'%(start_date[0:6]))
        # 筛选上市时间
        bond_info_df = bond_info_df[(bond_info_df['上市日期'] <= start_date)]
        return bond_info_df.set_index('转债代码')['正股代码'].to_dict()

    def load_cache(self,start_date,end_date):
        if(os.path.exists('D:\\bs_pair_info\\bs_cache_%s_%s.pickle'%(start_date,end_date))):
            with open('D:\\bs_pair_info\\bs_cache_%s_%s.pickle'%(start_date,end_date), 'rb') as f:
                package_list = pickle.load(f)
            return [True,package_list[0],package_list[1],package_list[2]]
        else:
            return [False]

    def store_cache(self,start_date,end_date,output_code_dict,output_bond_df_dict,output_stock_df_dict):
        with open('D:\\bs_pair_info\\bs_cache_%s_%s.pickle'%(start_date,end_date), 'wb') as f:
            pickle.dump([output_code_dict,output_bond_df_dict,output_stock_df_dict], f)
        print('成功缓存')
        return
    def get_database_df_dict(self,start_date,end_date,process_num=1):
        '''
        获取数据生成df_dict {code:[date1_df,date2_df}
        '''
        cache = self.load_cache(start_date,end_date)
        if(cache[0]):
            return cache[1], cache[2], cache[3]
        else:
            # 获取代码dict
            raw_code_dict = self.__get_start_date_exist_pair_dict(start_date)
            output_raw_code_dict = raw_code_dict.copy()

            # 初始化df_dict
            bond_df_dict = {i: [] for i in raw_code_dict.keys()}
            stock_df_dict = {i: [] for i in raw_code_dict.values()}
            # 验证长度
            df_length_dict = {i:0 for i in (list(raw_code_dict.keys())+list(raw_code_dict.values()))}
            pool = ThreadPoolExecutor(max_workers=process_num)

            def __get_mysql_bond_data(code_list):
                for code in code_list:
                    try:
                        temp_df = self.mysql_interface.interface_get_data(code,start_date,end_date)
                        df_length_dict[code] = len(temp_df)
                        temp_df.index = pandas.to_datetime(temp_df.index)
                        temp_inday_df_list = [group[1] for group in temp_df.groupby(temp_df.index.day)]
                        temp_inday_df_list.reverse()
                        bond_df_dict[code] = temp_inday_df_list
                    except Exception as e:
                        print(e)
                        del output_raw_code_dict[code]
                        print('无法找到',code,' ',e)
                return
            def __get_mysql_stock_data(code_list):
                for code in code_list:
                    try:
                        temp_df = self.mysql_interface.interface_get_data(code,start_date,end_date)
                        df_length_dict[code] = len(temp_df)
                        temp_df.index = pandas.to_datetime(temp_df.index)
                        temp_inday_df_list = [group[1] for group in temp_df.groupby(temp_df.index.day)]
                        temp_inday_df_list.reverse()
                        stock_df_dict[code] = temp_inday_df_list
                    except Exception as e:
                        print(e)
                        del output_raw_code_dict[code]
                        print('无法找到',code,' ',e)
                return



            for b_code_list in [list(raw_code_dict.keys())[i:i+100] for i in range(0,len(raw_code_dict.keys()),100)]:
                pool.submit(__get_mysql_bond_data,(b_code_list))

            for s_code_list in [list(raw_code_dict.values())[i:i+100] for i in range(0,len(raw_code_dict.values()),100)]:

                pool.submit(__get_mysql_stock_data,(s_code_list))

            pool.shutdown()

            majority_length = numpy.argmax(numpy.bincount(list(df_length_dict.values())))
            print(majority_length)
            output_code_dict = {k:v for k,v in output_raw_code_dict.items() if (df_length_dict[k]==majority_length and df_length_dict[v]==majority_length)}
            output_bond_df_dict = {k:v for k,v in bond_df_dict.items() if k in output_code_dict.keys()}
            output_stock_df_dict = {k:v for k,v in stock_df_dict.items() if k in output_code_dict.values()}
            self.store_cache(start_date,end_date,output_code_dict,output_bond_df_dict,output_stock_df_dict)
            return output_code_dict,output_bond_df_dict,output_stock_df_dict



class FactorValueProvider:
    '''
    输入：output_code_dict,output_bond_df_dict,output_stock_df_dict
    输出：预处理后的factor_df_list
    '''
    def __init__(self):
        pass

    def __get_factor_df_list(self, raw_code_dict, bond_df_dict, stock_df_dict, factor_func, factor_N):
        factor_df_list = []
        if ((factor_func.__name__)[0:1] == 'T'):
            for i in range(0,len(bond_df_dict.get(next(iter(bond_df_dict))))):
                single_day_bond_df_dict = {k: v[i] for k,v in bond_df_dict.items()}
                single_day_stock_df_dict = {k: v[i] for k,v in stock_df_dict.items()}

                factor_df = factor_func(raw_code_dict,single_day_bond_df_dict,single_day_stock_df_dict,factor_N)
                factor_df_list.append(factor_df)
        else:
            for i in range(0,len(bond_df_dict.get(next(iter(bond_df_dict))))):
                single_day_df_dict = {k: v[i] for k,v in bond_df_dict.items()}

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
            factor_df = factor_df.replace(numpy.inf,  0)  # 替换正inf为0
            factor_df = factor_df.replace(-numpy.inf, 0)  # 替换-inf为0

            return factor_df

        def __replace_extreme_factor_value(factor_df):
            def __cal_new_value(row):
                median_value = row.median()
                mad = abs(row-median_value).median()
                upper_bound = median_value + 5 * mad
                lower_bound = median_value - 5 * mad
                row = row.where(row<=lower_bound,lower_bound)
                row = row.where(row>=upper_bound,upper_bound)
                std = row.std()
                if std == 0 :
                    row[:] = 0
                else:
                    row = (row - row.mean())/std
                return row

            factor_df.apply(__cal_new_value,axis=1)
            print('因子去极值，标准化完成')
            return factor_df
        factor_df = __del_and_replace(factor_df)
        factor_df = __replace_extreme_factor_value(factor_df)
        return factor_df

    def input_bond_factor_lib(self,raw_code_dict, bond_df_dict, stock_df_dict,factor_func, factor_N):
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
        raw_factor_df_list = self.__get_factor_df_list(raw_code_dict, bond_df_dict, stock_df_dict, factor_func, factor_N)
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

        def __get_only_bond_IC_df(close_df,factor_df,pos_N):
            '''
            计算IC,分组IC,超额收益
            :param close_df:
            :param factor_df:
            :param pos_N:
            :return:
            '''
            def __get_layer_index_list_dict(factor_index_list, factor_df, seq_index):
                '''
                默认分10层
                :param factor_index_list:
                :param factor_df:
                :param seq_index:
                :return:
                '''
                factor_sorted_series = (factor_df.loc[factor_index_list[seq_index], :]).sort_values()
                factor_sorted_code_list = factor_sorted_series.index.tolist()
                if(len(factor_sorted_code_list)%10==0):
                    n = int(len(factor_sorted_code_list)/10)
                else:
                    n = int(len(factor_sorted_code_list)/10) + 1
                # 改这两个10就可以改分割的层数
                layer_code_list = [factor_sorted_code_list[i:i + n] for i in range(0, len(factor_sorted_code_list), n)]

                layer_index_list_dict = {i: layer_code_list[i] for i in range(0, 10)}
                return layer_index_list_dict

            def __cal_group_IC_dict(factor_value_group_dict,pctchg_group_dict,IC_dict):
                for i in range(0,10):
                    IC_dict['IC%d'%(i)].append(
                        pandas.Series(factor_value_group_dict[i]).corr(pandas.Series(pctchg_group_dict[i]), method="pearson"))

                    IC_dict['RANK_IC%d' % (i)].append(
                        pandas.Series(factor_value_group_dict[i]).corr(pandas.Series(pctchg_group_dict[i]), method="spearman"))
                return IC_dict


            factor_index_list = factor_df.index.tolist()
            close_index_list = close_df.index.tolist()
            #包含分组IC和总IC
            IC_dict = {'IC%d'%(i):[] for i in range(0,10)}
            IC_dict.update({'RANK_IC%d'%(i):[] for i in range(0,10)})
            IC_dict.update({'IC':[],'RANK_IC':[]})
            #超额收益
            profit_dict = {'Profit%d'%(i):[0] for i in range(0,10)}
            profit_dict['base_profit_C'] = [0]
            IC_index_list = []
            for seq_index in range(0, len(factor_index_list) - pos_N, pos_N):
                '''
                分组IC，每组收益
                '''
                if(factor_index_list[seq_index+1] in close_index_list):
                    layer_index_list_dict = __get_layer_index_list_dict(factor_index_list,factor_df,seq_index)
                    factor_value_group_dict = {k:factor_df.loc[factor_index_list[seq_index], :][v].values for k,v in layer_index_list_dict.items()}

                    pctchg_group_dict = {k:(numpy.array(close_df.loc[factor_index_list[seq_index+pos_N],:][v].values)-numpy.array(close_df.loc[factor_index_list[seq_index],:][v].values))/numpy.array(close_df.loc[factor_index_list[seq_index],:][v].values) for k,v in layer_index_list_dict.items()}
                    IC_dict = __cal_group_IC_dict(factor_value_group_dict,pctchg_group_dict,IC_dict)
                    #计算收益率
                    for key, value in pctchg_group_dict.items():
                        # profit_dict['Profit%s' % (key)].append(sum(value)/len(value) - sum(numpy.array(close_df.loc[factor_index_list[seq_index+pos_N],:])-numpy.array(close_df.loc[factor_index_list[seq_index],:]))/len(numpy.array(close_df.loc[factor_index_list[seq_index],:])))
                        profit_dict['Profit%s' % (key)].append(sum(value)/len(value))

                    '''
                    整体IC
                    '''
                    factor_value_list = factor_df.loc[factor_index_list[seq_index], :].values
                    factor_code_list = factor_df.columns.tolist()
                    profit_pctChg_list = (numpy.array(close_df.loc[factor_index_list[seq_index + pos_N], :][factor_code_list].values) - numpy.array(close_df.loc[factor_index_list[seq_index], :][factor_code_list].values)) / numpy.array(close_df.loc[factor_index_list[seq_index], :][factor_code_list].values)
                    IC_dict['IC'].append(pandas.Series(factor_value_list).corr(pandas.Series(profit_pctChg_list), method="pearson"))
                    IC_dict['RANK_IC'].append(pandas.Series(factor_value_list).corr(pandas.Series(profit_pctChg_list), method="spearman"))
                    IC_index_list.append(factor_index_list[seq_index])
                    #计算基准收益
                    profit_dict['base_profit_C'].append(sum((numpy.array(close_df.loc[factor_index_list[seq_index + pos_N], :]) - numpy.array(close_df.loc[factor_index_list[seq_index], :]))/numpy.array(close_df.loc[factor_index_list[seq_index], :])) / len(numpy.array(close_df.loc[factor_index_list[seq_index], :])))

            IC_df = pandas.DataFrame(IC_dict,index=IC_index_list)
            #调整acc_profit_index，让其累计收益率从开盘时期开始
            profit_index = IC_index_list
            profit_index.insert(0,close_index_list[0])

            profit_df = pandas.DataFrame(profit_dict,index = profit_index)

            return IC_df,profit_df

        close_df = __get_close_df(df_dict)
        IC_df,profit_df = __get_only_bond_IC_df(close_df,factor_df,pos_N)

        return IC_df,profit_df

    def multiday_single_factor_result_output(self,df_dict,factor_df_list,pos_N):
        '''

        :param test_result_dict: {code:[[IC_df,acc_profit_df]}
        :return:
        '''

        def __get_IC_and_profit_concat_df(df_dict,factor_df_list,pos_N):
            IC_df = pandas.DataFrame()
            profit_df = pandas.DataFrame()
            for i in range(0, len(factor_df_list)):
                single_day_df_dict = {k: v[i] for k, v in df_dict.items()}
                inday_IC_df, inday_profit_df = self.__single_day_single_factor_test(single_day_df_dict,
                                                                                    factor_df_list[i], pos_N)
                IC_df = pandas.concat([IC_df, inday_IC_df], axis=0)
                # 获得profit_df值
                profit_df = pandas.concat([profit_df, inday_profit_df], axis=0)
            return IC_df,profit_df

        def __get_accumulated_IC_and_profit_df(IC_df,profit_df):
            for col in IC_df.columns:
                IC_df['acc'+col] = IC_df[col].cumsum()
            acc_IC_df = IC_df.loc[:,IC_df.columns.str.contains('acc')]

            for col in profit_df.columns:
                profit_df['acc' + col] = profit_df[col].cumsum()
            acc_profit_df = profit_df.loc[:, profit_df.columns.str.contains('acc')]
            return acc_IC_df,acc_profit_df


        def __get_end_acc_profit(acc_profit_df):
            end_acc_profit = {}
            for col in acc_profit_df.columns:
                end_acc_profit['Group' + col[-1]] = acc_profit_df[col].values[-1]
            profit_df = pandas.DataFrame(end_acc_profit, index=['profit'])
            return profit_df.T
        def __get_sharp_rate(acc_profit_df):
            sharp_dict = {}
            for col in acc_profit_df.columns:
                sharp_dict['Group' + col[-1]] = acc_profit_df[col].values[-1]/acc_profit_df[col].std()
            sharp_df = pandas.DataFrame(sharp_dict, index=['sharp_rate'])
            return sharp_df.T

        def __get_group_IR(IC_df):
            IR_dict = {}
            for col in IC_df.columns[IC_df.columns.str.startswith('RANK')]:
                IR_dict['Group'+col[-1]] = IC_df[col].mean()/IC_df[col].std()
            IR_df = pandas.DataFrame(IR_dict,index =['IR'])
            return IR_df.T

        def __get_group_IC_mean(IC_df):
            print(IC_df)
            IC_mean_dict = {}
            for col in IC_df.columns[IC_df.columns.str.startswith('RANK')]:
                IC_mean_dict['Group' + col[-1]] = IC_df[col].mean()
                print(IC_df[col].sum(),len(IC_df[col]))

            IC_mean_df = pandas.DataFrame(IC_mean_dict, index=['IC_mean'])
            return IC_mean_df.T

        def __get_group_IC_std(IC_df):
            IC_std_dict = {}
            for col in IC_df.columns[IC_df.columns.str.startswith('RANK')]:
                IC_std_dict['Group' + col[-1]] = IC_df[col].std()
            IC_std_df = pandas.DataFrame(IC_std_dict, index=['IC_std'])
            return IC_std_df.T

        IC_df,profit_df = __get_IC_and_profit_concat_df(df_dict, factor_df_list, pos_N)
        acc_IC_df,acc_profit_df = __get_accumulated_IC_and_profit_df(IC_df,profit_df)

        #计算IC特征值表，包含分组IC均值，std，IR
        IC_mean_df = __get_group_IC_mean(IC_df)
        IC_std_df = __get_group_IC_std(IC_df)
        IC_IR_df = __get_group_IR(IC_df)
        #计算profit特征值表，包含end_profit，sharp_rate
        profit_end_df = __get_end_acc_profit(acc_profit_df)
        profit_sharp_df = __get_sharp_rate(acc_profit_df)
        #聚合所有df为一张表.01
        factor_test_df = pandas.concat([IC_mean_df,IC_std_df,IC_IR_df,profit_end_df,profit_sharp_df],axis=1)
        # output_figure_acc_profit_df = acc_profit_df.drop(['accbase_profit_C'], axis=1)
        # output_figure_acc_profit_df.index = output_figure_acc_profit_df.index.astype('str')
        # output_figure_acc_profit_df.plot()
        # plt.show()



        return acc_IC_df,acc_profit_df,factor_test_df

class AssessFactor:
    def __init__(self,start_date,end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.__get_history_data()
    def __get_history_data(self):
        self.output_code_dict,self.output_bond_df_dict,self.output_stock_df_dict = HistoryDataProvider().get_database_df_dict(self.start_date,self.end_date)
        return self
    def get_factor_test_result(self,factor_func,factor_N,pos_N):
        factor_df_list = FactorValueProvider().input_bond_factor_lib(self.output_code_dict,self.output_bond_df_dict,self.output_stock_df_dict,factor_func,factor_N)
        acc_IC_df,acc_profit_df,factor_test_df = SingleFactorTest().multiday_single_factor_result_output(self.output_bond_df_dict,factor_df_list,pos_N)
        print(factor_test_df)
        return factor_test_df
    def testify_param(self,factor_func):
        '''
        自动确定最优的参数，目标变量：累计收益率
        :return:
        '''
        factor_N_list = range(2,21)
        for factor_N in factor_N_list:
            pos_N_list = range(1,factor_N)
            factor_df_list = FactorValueProvider().input_bond_factor_lib(self.output_code_dict,
                                                                         self.output_bond_df_dict,
                                                                         self.output_stock_df_dict, factor_func,
                                                                         factor_N)
            for pos_N in pos_N_list:
                try:
                    acc_IC_df, acc_profit_df, factor_test_df = SingleFactorTest().multiday_single_factor_result_output(
                        self.output_bond_df_dict, factor_df_list, pos_N)

                    temp_df = pandas.DataFrame([{'IC_mean':factor_test_df.loc['GroupC','IC_mean'],
                                                 'IC_std':factor_test_df.loc['GroupC','IC_std'],
                                                 'IR':factor_test_df.loc['GroupC','IR'],
                                                 'Group9_profit':factor_test_df.loc['Group9','profit'],
                                                 'Group9_sharp_rate':factor_test_df.loc['Group9','sharp_rate'],
                                                 'L3group_ave_profit':mean([factor_test_df.loc['Group9','profit'],factor_test_df.loc['Group8','profit'],factor_test_df.loc['Group7','profit']]),
                                                 'L3group_ave_sharp_rate':mean([factor_test_df.loc['Group9','sharp_rate'],factor_test_df.loc['Group8','sharp_rate'],factor_test_df.loc['Group7','sharp_rate']])}],
                                           index=[factor_func.__name__+'_'+str(factor_N)+'_'+str(pos_N)],columns=['IC_mean','IC_std','IR','Group9_profit','Group9_sharp_rate','L3group_ave_profit','L3group_ave_sharp_rate'])

                    temp_df.to_csv('D:\\bs_pair_info\AssessFactor\\%s\\%s\\%s_params_test.csv'%(factor_func.__name__,self.start_date[0:6],factor_func.__name__),encoding='utf-8-sig',mode='a')
                except Exception as e:
                    print(e)




# A1 = AssessFactor('20230801','20230831')
# A1.testify_param(reverse_factor)
# A1.testify_param(price_slope_factor)
# A1.testify_param(T_BS_link_factor)
# A1 = AssessFactor('20230901','20230930')
# A1.testify_param(reverse_factor)
# A1.testify_param(price_slope_factor)
# A1.testify_param(T_BS_link_factor)
AssessFactor('20230801','20230802').get_factor_test_result(reverse_factor,10,3)

