import datetime
import sys
import time
import traceback
from xtquant import xtdata
import akshare
import pandas
import pymysql
from sqlalchemy import create_engine, text
from sqlalchemy.orm import session, sessionmaker

'''
数据库开始：230801
'''
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


class DataBase:
    def __init__(self):
        pass

    def get_tradeday(self):
        '''
        获得以今天作为最后一个元素的交易时间list
        :return: self
        '''
        trade_date_raw_list = xtdata.get_trading_dates('SZ', start_time='20230101', end_time='', count=-1)
        trade_date_list = [time.strftime("%Y%m%d", time.localtime(int(i / 1000))) for i in trade_date_raw_list]
        return trade_date_list

    def get_time_filter_df(self,start_date):
        '''
        从开盘获得的比价表中得到目前正在交易的债券，结合上市日期筛选债券
        :return:
        '''
        # 获得交易日
        trade_date_list = self.get_tradeday()
        bond_compare_df = akshare.bond_cov_comparison()
        # 调整上市日期
        bond_compare_df = bond_compare_df[bond_compare_df['上市日期'] != '-']

        bond_compare_df['上市日期'] = bond_compare_df['上市日期'].astype('str')
        bond_compare_df['上市日期'] = pandas.to_datetime(bond_compare_df['上市日期'])
        # 筛选上市时间
        bond_compare_df = bond_compare_df[(bond_compare_df['上市日期'] <= start_date)]
        return bond_compare_df


    def df_to_bond_stock_dict(self,df):
        '''
        将获得的比价表df转变为{bond_code:stock_code}
        :param df: 比价表
        :return:
        '''
        raw_qmt_code_dict = df.set_index('转债代码')['正股代码'].to_dict()
        qmt_code_dict = {}
        for b_code, s_code in raw_qmt_code_dict.items():
            if (b_code[0:2] == '12'):
                nb_code = b_code + '.SZ'
            else:
                nb_code = b_code + '.SH'
            if (s_code[0:1] == '6'):
                ns_code = s_code + '.SH'
            else:
                ns_code = s_code + '.SZ'
            qmt_code_dict.update({nb_code: ns_code})
        return qmt_code_dict


    def get_1min_df_dict(self,start_time,end_time):
        '''
        :param start_time:
        :param end_time:
        :return:
        '''
        start_time = start_time + '093000'
        end_time = end_time + '150000'
        qmt_code_dict = self.df_to_bond_stock_dict(self.get_time_filter_df(start_time))
        bond_code_list = list(qmt_code_dict.keys())
        stock_code_list = list(qmt_code_dict.values())
        bond_result_dict = {}
        stock_result_dict = {}

        for code in bond_code_list:
            seq = xtdata.subscribe_quote(code, period='1m', start_time=start_time, end_time=end_time,
                                         count=-1,
                                         callback=None)
            df = xtdata.get_market_data_ex(field_list=[], period='1m', stock_list=[code],
                                           start_time=start_time,
                                           end_time=end_time)[code]
            xtdata.unsubscribe_quote(seq)
            df.reset_index(inplace=True)
            df = df[['index','open','high','low','close','volume','amount']]
            df.rename(columns={'index':'时间','open':'开盘','high':'最高','low':'最低','close':'收盘','volume':'成交量','amount':'成交额'},inplace=True)
            bond_result_dict[start_time[0:6]+code[:-3]] = df

        for code in stock_code_list:
            seq = xtdata.subscribe_quote(code, period='1m', start_time=start_time, end_time=end_time,
                                         count=-1,
                                         callback=None)
            df = xtdata.get_market_data_ex(field_list=[], period='1m', stock_list=[code],
                                           start_time=start_time,
                                           end_time=end_time)[code]
            xtdata.unsubscribe_quote(seq)
            df.reset_index(inplace=True)
            df = df[['index','open', 'high', 'low', 'close', 'volume', 'amount']]
            df.rename(columns={'index':'时间','open': '开盘', 'high': '最高', 'low': '最低', 'close': '收盘', 'volume': '成交量', 'amount': '成交额'},inplace=True)
            stock_result_dict[start_time[0:6]+code[:-3]] = df
        return bond_result_dict,stock_result_dict








class MySQLInterface:
    def __init__(self):
        self.host = 'localhost'
        self.user = 'root'
        self.passwd = 'root'
        self.port = 3306
        self.__connect_mysql_database()

    @try_except
    def __connect_mysql_database(self):
        '''
        基础函数：连接mysql数据库
        :return:
        '''

        self.db = pymysql.connect(host=self.host, user=self.user, passwd=self.passwd,port=self.port)

        self.cursor = self.db.cursor()
        self.bonddb_engine = create_engine('mysql+pymysql://root:root@localhost:3306/bdb',pool_size=50)
        self.stockdb_engine = create_engine('mysql+pymysql://root:root@localhost:3306/sdb',pool_size=50)
        self.bonddb_engine_connect = self.bonddb_engine.connect()
        self.stockdb_engine_connect = self.stockdb_engine.connect()
        self.bdb_Session = sessionmaker(bind=self.bonddb_engine)
        self.sdb_Session = sessionmaker(bind=self.stockdb_engine)

        print('连接成功！')
        return

    def __is_exist_in_database(self, table_name, database):
        '''
        基础函数：判断数据库内表是否存在
        :param table_name:
        :param database:
        :return:
        '''
        self.cursor.execute('use {}'.format(database))
        sql = """
        SHOW TABLES LIKE '{}'
        """.format(table_name)
        self.cursor.execute(sql)
        result = self.cursor.fetchall()
        if(result==()):
            return False
        else:
            return True




    def store_month_1m_data(self, start_date, end_date):
        '''
        核心函数：储存1m数据，并给出
        :param start_date:
        :param end_date:
        :return:
        '''
        self.__store_bs_pair_info(start_date)

        bond_result_dict, stock_result_dict = DataBase().get_1min_df_dict(start_date,end_date)

        for table_name, df in bond_result_dict.items():
            if (self.__is_exist_in_database(table_name, 'bdb')):
                df.to_sql(table_name, self.bonddb_engine, index=True, if_exists='replace')
                print('表已存在，已替换')

            else:
                df.to_sql(table_name, self.bonddb_engine, index=True)

        for table_name, df in stock_result_dict.items():
            if (self.__is_exist_in_database(table_name, 'sdb')):
                df.to_sql(table_name, self.stockdb_engine, index=True, if_exists='replace')

                print('表已存在，已替换')

            else:
                df.to_sql(table_name, self.stockdb_engine, index=True)
        print('今日数据已全部录入数据库')
        return

    def __store_bs_pair_info(self,start_date):
        '''
        附加函数，给store_month_1m_data增加额外信息
        :return:
        '''
        def change_code_to_qmt_code(code):
            if code[0:2] == '11' or code[0:1] == '6':
                return code+'.SH'
            else:
                return code+'.SZ'
        bond_compare_df = akshare.bond_cov_comparison()
        # 调整上市日期
        bond_compare_df = bond_compare_df[bond_compare_df['上市日期'] != '-']

        bond_compare_df['上市日期'] = bond_compare_df['上市日期'].astype('str')
        bond_compare_df['上市日期'] = pandas.to_datetime(bond_compare_df['上市日期'])
        # 筛选上市时间
        bond_compare_df = bond_compare_df[(bond_compare_df['上市日期'] <= start_date)]
        df = bond_compare_df[['转债代码','转股价','正股代码','上市日期','回售触发价','强赎触发价','到期赎回价','开始转股日']].copy()
        df['转债代码'] = df['转债代码'].apply(lambda x:change_code_to_qmt_code(x))
        df['正股代码'] = df['正股代码'].apply(lambda x:change_code_to_qmt_code(x))
        df = df.set_index('转债代码')
        df.to_csv('D:\\bs_pair_info\%s_pair_info.csv'%(start_date[0:6]),encoding='utf-8-sig')
        print('%s股债对信息创建成功'%(start_date[0:6]))
        return

    def interface_get_data(self,code,start_date,end_date):
        '''
        :param code: 可为数字代码，也可为qmt代码
        :param start_date: 开始日期，例为'20230812'
        :param end_date: 同开始日期
        :return:
        '''

        bdb_session = self.bdb_Session()
        sdb_session = self.sdb_Session()
        if(code[0:2]=='11' or code[0:2]=='12'):
            database = 'bdb'
        else:
            database = 'sdb'
        table_name = start_date[0:6]+code[0:6]
        start_time = start_date + '093000'
        end_time = end_date + '150000'

        sql = "SELECT * FROM `{}` WHERE 时间  between '{}' and '{}'".format(table_name,start_time,end_time)
        if(database=='bdb'):
            df = pandas.read_sql_query(text(sql),bdb_session.connection())
        else:
            df = pandas.read_sql_query(text(sql),sdb_session.connection())
        df.set_index('时间',inplace=True)
        bdb_session.close()
        sdb_session.close()
        return df


if __name__ == '__main__':
    pass
    '''
    23 8月信息已全部录入
    '''
    # MySQLInterface().store_month_1m_data('20230801','20230830')
    # MySQLInterface().store_month_1m_data('20230801', '20230831')
    # MySQLInterface().store_month_1m_data('20230901', '20230930')


