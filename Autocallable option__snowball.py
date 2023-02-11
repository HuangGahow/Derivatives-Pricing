# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:57:23 2022

@author: HP
"""


def simulation(S, r, T, sigma, I, dn = 0, steps = 1000, plotpath = False, plothist = False):
    """

    :param S: 初始价格
    :param r: 无风险收益率
    :param T: 到期期限（年）
    :param sigma: 波动率
    :param I: 模拟路径数量
    :param dn: 敲入点
    :param steps:
    :param plotpath:
    :param plothist:
    :return:
    """
    delta_t = float(T)/steps
    Spath = np.zeros((steps + 1, I))
    Spath[0] = S

    for t in range(1, steps + 1):
        z = np.random.standard_normal(I)
        middle1 = Spath[t-1, 0:I] * np.exp((r - 0.5 * sigma ** 2) * delta_t + sigma * np.sqrt(delta_t) * z)
        uplimit = Spath[t-1] * 1.1
        lowlimit = Spath[t-1] * 0.9
        temp = np.where(uplimit < middle1, uplimit, middle1)
        #当where内有三个参数时，第一个参数表示条件，当条件成立时where方法返回x，当条件不成立时where返回y
        temp = np.where(lowlimit > middle1, lowlimit, temp)
        Spath[t, 0:I] = temp

    if plotpath:
        plt.plot(Spath[:, :])
        plt.plot([dn]*len(Spath))
        plt.xlabel('time')
        plt.ylabel('price')
        plt.title('Price Simulation')
        plt.grid(True)
        plt.show()
        plt.close()

    if plothist:
        plt.hist(Spath[int(T*steps)], bins=50)
        plt.title('T moment price Histogram')
        plt.show()
        plt.close()

    return Spath



def snowball_cashflow(price_path, coupon, I):
    payoff = np.zeros(I)
    knock_out_times = 0
    knock_in_times = 0
    existence_times = 0
    for i in range(I):
        # 收盘价超过敲出线的交易日
        tmp_up_d = np.where(price_path[:, i] > k_out)
        #当where内只有一个参数时，那个参数表示条件，当条件成立时，where返回的是每个符合condition条件元素的坐标,返回的是以元组的形式
        # 收盘价超出敲出线的观察日
        tmp_up_m = tmp_up_d[0][tmp_up_d[0] % 21 == 0]
        # 收盘价超出敲出线的观察日（超过封闭期）
        tmp_up_m_md = tmp_up_m[tmp_up_m > lock_period]
        tmp_dn_d = np.where(price_path[:, i] < dn)
        # 根据合约条款判断现金流

        # 情景1：发生过向上敲出
        if len(tmp_up_m_md) > 0:
            t = tmp_up_m_md[0]
            payoff[i] = coupon * (t/252) * np.exp(-r * t/252)
            knock_out_times += 1

        # 情景2：未敲出且未敲入
        elif len(tmp_up_m_md) == 0 and len(tmp_dn_d[0]) == 0:
            payoff[i] = coupon * np.exp(-r * T)
            existence_times += 1

        # 情景3：只发生向下敲入，不发生向上敲出
        elif len(tmp_dn_d[0]) > 0 and len(tmp_up_m_md) == 0:
            # 只有向下敲入，没有向上敲出
            payoff[i] = 0 if price_path[len(price_path)-1][i] > 1 else (price_path[len(price_path)-1][i] - S) * np.exp(-r * T)
            knock_in_times += 1
        else:
            print(i)
    return payoff, knock_out_times, knock_in_times, existence_times



np.random.seed(0)
sigma = 0.13
T = 1
S = 1
r = 0.03
K = 1
I = 300000
dn = K * 0.85
k_out = S * (0.03 + 1)
lock_period = 0
coupon = 0.2
principal = 1

steps = 252 * T
price_path = simulation(S, r, T, sigma, I, dn=dn, steps=steps, plotpath=False, plothist=False)

payoff, knock_out_times, knock_in_times, existence_times = snowball_cashflow(price_path, coupon, I)
price = sum(payoff)/len(payoff)
print('snow_ball price: %f' % price)
print('knock_out_times: %f' % (knock_out_times/I))
print('knock_in_times: %f' % (knock_in_times/I))
print('existence_times: %f' % (existence_times/I))
np.random.standard_normal(10)
