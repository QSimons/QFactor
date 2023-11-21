import numpy
import pandas
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def price_slope_factor(df_dict,N):

    def cal_slope(roll_data,N):
        if(len(roll_data)<N):
            return numpy.nan
        x = numpy.arange(1,N+1)
        y = roll_data.values / roll_data.values[0]
        result = OLS(exog=add_constant(x),endog=y).fit()
        r_square = result.rsquared
        k = result.params[1]
        return -10000*k*r_square
    # factor_df 为以time为index，以code为columns的df
    factor_df = pandas.DataFrame()
    for code,df in df_dict.items():
        temp_series = df['收盘'].rolling(N).apply(lambda x:cal_slope(x,N))
        temp_series.rename(code,inplace=True)
        factor_df = pandas.concat([factor_df,temp_series],axis=1)
    factor_df.index.name = 'price_slope_factor_%s'%(N)
    return factor_df

def reverse_factor(df_dict,N):
    def cal_pctchg(x):
        if(len(x)<2):
            return numpy.nan
        price_list  = x.values
        return (price_list[1] - price_list[0])/price_list[0]
    factor_df = pandas.DataFrame()
    for code,df in df_dict.items():
        df['涨跌幅'] = df['收盘'].rolling(2).apply(lambda x:cal_pctchg(x))
        df['动量值'] = -df['涨跌幅'] * (df['成交量']+1).apply(numpy.log)
        time_series = (df['动量值']).rolling(N).sum()
        time_series.rename(code,inplace=True)
        factor_df = pandas.concat([factor_df,time_series],axis=1)
    factor_df.index.name = 'reverse_factor_%s'%(N)
    return factor_df

def reverse_factor_2(single_day_bond_df_dict,factor_N):
    def cal_pctChg(x):
        pass
    for code,df in single_day_bond_df_dict.items():
        inday_open = df.loc[df.index[0],'开盘']
        pass

        print(df)
def T_BS_link_factor(raw_code_dict,single_day_bond_df_dict,single_day_stock_df_dict,factor_N):

    factor_df = pandas.DataFrame()
    for b_code, b_df in single_day_bond_df_dict.items():
        s_code = raw_code_dict[b_code]
        s_df = (single_day_stock_df_dict[s_code])
        s_df = s_df.rename(columns={k:('股票'+k) for k in s_df.columns})
        b_s_df = pandas.concat([b_df,s_df],axis=1)
        index = b_s_df.index.tolist()
        link_series = pandas.Series([], dtype='float64')

        for i in range(0,len(b_df)-factor_N,factor_N):
            temp_b_s_df = b_s_df.loc[index[i:i+factor_N],:]
            result = OLS(exog=add_constant(temp_b_s_df['股票收盘']), endog=temp_b_s_df[['收盘']]).fit()
            link_series[result.resid.tail(1).index[0]] = -result.resid.tail(1).values[0]/b_df.loc[result.resid.tail(1).index[0],'收盘'] * result.rsquared
        link_series.rename(b_code,inplace=True)
        factor_df = pandas.concat([factor_df,link_series],axis=1)
    factor_df.index.name = 'BS_link_factor%s' % (factor_N)
    return factor_df

    #     df['涨跌幅'] = df['收盘'].rolling(2).apply(lambda x: cal_pctchg(x))
    #     df['动量值'] = df['涨跌幅'] * (df['成交量'] + 1).apply(numpy.log)
    #     time_series = df['动量值'].rolling(N).sum()
    #     time_series.rename(code, inplace=True)
    #     factor_df = pandas.concat([factor_df, time_series], axis=1)
    # factor_df.index.name = 'reverse_factor_%s' % (N)

def Y_premium_rate_factor(raw_code_dict, single_day_bond_df_dict, single_day_stock_df_dict, pair_info_df, factor_N):
    transfer_price_dict = pair_info_df.set_index('转债代码')['转股价'].to_dict()
    factor_df = pandas.DataFrame()
    for b_code, b_df in single_day_bond_df_dict.items():
        s_code = raw_code_dict[b_code]
        b_df['正股价'] = single_day_stock_df_dict[s_code]['收盘']
        transfer_price = transfer_price_dict[b_code]
        time_series = (b_df['收盘'] - (100/transfer_price*b_df['正股价']))/(100/transfer_price*b_df['正股价'])
        time_series.rename(b_code,inplace=True)
        factor_df = pandas.concat([factor_df,time_series],axis=1)
    factor_df.index.name = 'premium_rate_factor'
    return factor_df

# def Y_premium_reverse_factor(raw_code_dict, single_day_bond_df_dict, single_day_stock_df_dict, pair_info_df, factor_N):
#     factor1 = reverse_factor(single_day_bond_df_dict,factor_N)
#     factor2 = Y_premium_rate_factor(raw_code_dict, single_day_bond_df_dict, single_day_stock_df_dict,  pair_info_df, factor_N)
#     factor2 = factor2/10
#     print(factor2)
#     return factor1-factor2

