# QFactor 1.0.0
## 1.介绍 
* QFactor是QSimons独立编写的因子分析、组合、回测工具
* 实现本地化数据的读取，本地存有转债、股票对的分钟数据，使用DataBase.py实现数据库的读取和存入
* 实现因子库搭建，FactorLibrary.py已存入几个示例因子，因子库按照已有模板搭建后可直接用于回测
* 实现因子的分层回测
* 实现绘制指定时间范围内的分层RANK-IC、IC、收益率曲线，并且已经部分实现自动变更参数范围自动寻优参数的功能
* NewSimulate.py在Simulate.py的基础上大量使用python向量化方法，因此大大提升了回测效率
* 本项目仅用于python金融分析学习交流使用，请勿用于投资等商业用途。



## 2.使用和回测：
1.NewSimulate.py或者Simulate.py调用AssessFactor().get_factor_test_result()可获得收益率的相关结果图，如需用RANK-IC和IC的结果图，可自定义get_factor_test_result()函数。

2.回测结果：
![反转因子分层收益（2023.08全月转债全市场，形成期13分钟，换仓周期30分钟，手续费万分之一)](https://github.com/QSimons/QFactor/image/反转因子分层收益（转债市场）.png)

## 3.维护
@QSimons
## 4.更新记录
1.0.0---上传所有本地代码
## 5.未来更新
1.未来会添加更多的因子库  
2.完善多因子组合和回测
