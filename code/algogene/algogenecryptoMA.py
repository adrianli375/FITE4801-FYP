from AlgoAPI import AlgoAPIUtil, AlgoAPI_Backtest
from datetime import datetime, timedelta
import pandas as pd

class AlgoEvent:
    def __init__(self):
        pass
        
    def start(self, mEvt):
        self.evt = AlgoAPI_Backtest.AlgoEvtHandler(self, mEvt)
        self.instrument = "BTCUSD"
        self.strikethrough = ""
        
        ### Parameters
        self.n_days = 5 #MA
        self.takeprofit = False
        self.takeprofitpercentage = 0.4
        self.trailingstoploss = False
        self.trailingstoplosspercent = 0.07
        #self.percent_above = 0.03
        self.volatility_n_days = 10
        self.volatility_coefficient = 0.3
        ### End Parameters


        self.upperlinepos = ""
        self.lowerlinepos = ""
        self.cur_purchaseprice = 0
        self.cont_liquidate = False
        self.highwatermark = 0
        self.evt.start()

    def on_marketdatafeed(self, md, ab):
        res = self.evt.getHistoricalBar({"instrument":self.instrument},self.n_days, "D")
        df = pd.DataFrame(data=[[res[t]['o'],res[t]['h'],res[t]['l'],res[t]['c']] for t in res],columns=['Open', 'High', 'Low', 'Close'])
        #self.evt.consoleLog(res)
        MA = df["Close"].mean()
        #self.evt.consoleLog(MA)
        
        res = self.evt.getHistoricalBar({"instrument":self.instrument},self.volatility_n_days, "D")
        df2 = pd.DataFrame(data=[[res[t]['o'],res[t]['h'],res[t]['l'],res[t]['c']] for t in res],columns=['Open', 'High', 'Low', 'Close'])
        c = df2["Close"].pct_change().dropna().std()
        pos, osOrder, pendOrder = self.evt.getSystemOrders()
        self.percent_above = c * self.volatility_coefficient
        price = md.lastPrice
        if len(osOrder) > 0:
            self.highwatermark = max(price,self.highwatermark)
            self.lowwatermark = min(price,self.lowwatermark)
            close_trades = False
            if self.takeprofit and ((self.strikethrough == "Upper" and price/self.cur_purchaseprice - 1 > self.takeprofitpercentage) or (self.strikethrough == "Lower" and 1 - price/self.cur_purchaseprice > self.takeprofitpercentage)):
                self.evt.consoleLog("Take Profit {},{},{},{}".format(price,MA,self.strikethrough,quantity))
                close_trades = True
            elif self.trailingstoploss and ((self.strikethrough == "Upper" and price/self.highwatermark < 1 - self.trailingstoplosspercent) or (self.strikethrough == "Lower" and price/self.lowwatermark > 1 + self.trailingstoplosspercent)):
                self.evt.consoleLog("Trailing stop loss: price: {}, highwm:{}, lowwm:{},{}, q:{}".format(price,self.highwatermark,self.lowwatermark,self.strikethrough,quantity))
                close_trades = True
            elif self.strikethrough == "Upper":
                if price <= MA*(1+self.percent_above):
                    self.evt.consoleLog("Pass MA: sell")
                    close_trades = True
                if price >= MA/(1+self.percent_above):
                    self.evt.consoleLog("Pass MA: buy back")
                    close_trades = True
            if close_trades:
                for tradeID in osOrder:
                    self.close_order(tradeID)
        else:
            r = self.evt.getAccountBalance()
            if self.upperlinepos == "Lower" and price >= MA*(1+self.percent_above):
                q = r["availableBalance"]/price*0.8 
                self.open_order(1,q)
                # self.Log(f"Price: {price}, MA: {MA}, buy {q}")
                self.strikethrough = "Upper"
                self.highwatermark = price
                self.lowwatermark = price
                self.cur_purchaseprice = price
            if self.lowerlinepos == "Upper" and price <= MA/(1+self.percent_above):
                q = r["availableBalance"]/price*0.8
                self.open_order(-1,q)
                # self.Log(f"Price: {price}, MA: {MA}, sell {q}")
                self.strikethrough = "Lower"
                self.highwatermark = price
                self.lowwatermark = price
                self.cur_purchaseprice = price
        
        if price >= MA*(1+self.percent_above):
            self.upperlinepos = "Upper"
        else:
            self.upperlinepos = "Lower"
        
        if price <= MA/(1+self.percent_above):
            self.lowerlinepos = "Lower"
        else:
            self.lowerlinepos = "Upper"
            
        pass

    def on_openPositionfeed(self, op, oo, uo):
        pass
    
    def open_order(self, buysell,volume):
        order = AlgoAPIUtil.OrderObject()
        order.instrument = self.instrument
        order.openclose = 'open'
        order.buysell = buysell    #1=buy, -1=sell
        order.ordertype = 0  #0=market, 1=limit
        order.volume = volume
        # self.evt.consoleLog(buysell,orderref,Price)
        self.evt.sendOrder(order)
        
    def close_order(self,tradeID):
        self.evt.consoleLog("close",tradeID)
        order = AlgoAPIUtil.OrderObject()
        order.tradeID = int(tradeID)
        order.openclose = 'close'
        # order.orderRef = orderref
        self.evt.sendOrder(order)



