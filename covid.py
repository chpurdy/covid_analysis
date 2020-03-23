import matplotlib.pyplot as plt 
import csv 
import numpy
import math
from scipy.optimize import curve_fit
from datetime import datetime, timedelta


def exp_func(x, a, b):
    return a * numpy.exp(b*x)

class Covid:
    def __init__(self):
        with open('time_series_19-covid-Confirmed.csv') as f:
            data = csv.reader(f)
            headers = next(data)
            self.us_data = {}
            self.china_data = {}
            
            # initialize data
            for date in headers[4:]:
                self.us_data[date] = 0
                self.china_data[date] = 0

            # get data
            for row in data:
                if row[1] == 'US':
                    for i in range(4,len(row)):
                        self.us_data[headers[i]] += int(row[i])
                elif row[1] == 'China':
                    for i in range(4,len(row)):
                        self.china_data[headers[i]] += int(row[i])
        

    def display(self):
        print(self.us_data)


    def daily_inc(self):
        d = [(k,v) for k,v in self.us_data.items()]
        daily = {}
        for i in range(1,len(d)):
            daily[d[i][0]] = d[i][1] / d[i-1][1]
        
        return daily
        
        
    def daily_inc_graph(self):
        daily = self.daily_inc()
        x = [dates for dates,_ in daily.items()]
        y = [change for _, change in daily.items()]
        plt.xticks(rotation='vertical')
        plt.bar(x,y)
        plt.show()

    def predict_confirmed(self, start, end, days_out):
        if start < 0 or end > len(self.us_data.values()):
            print("INVALID START/END")
            return

        dates = list(self.us_data.keys())[start:end+1]
        print(f"From {dates[0]} to {dates[-1]}")

        x = numpy.array([i-start for i in range(start,end+1)])
        y = numpy.array(list(self.us_data.values())[start:end+1])

        model, model_cov = curve_fit(exp_func,x,y)
        
        print(f"Coefficients: {model}")
        def gen_model(x):
            return model[0]*numpy.exp(model[1]*x)

        model_y = numpy.array([gen_model(i) for i in x])

        for i in range(len(model_y)):
            print(f"{dates[i]:>8}: Model: {model_y[i]:6.0f}\tActual: {y[i]:6}\tDifference: {model_y[i] - y[i]:5.0f}") 
            # positive means OVERESTIMATE


        date = datetime.strptime(dates[-1],'%m/%d/%y')
        one_day = timedelta(days=1)
        future_dates = []
        for i in range(1,days_out+1):
            date = date + one_day
            future_dates.append(date.strftime("%m/%d/%y"))

        start_num = end+1
        num_days_out = days_out
        
        x = numpy.array([i-start for i in range(start_num,start_num+num_days_out+1)])
        y = numpy.array([gen_model(t) for t in x])
        # print(x)
        # print(y)

        print(f"Predictions from {future_dates[0]} to {future_dates[-1]}:")
        for i in range(len(future_dates)):
            print(f"{future_dates[i]:>8}: Prediction: {y[i]:6.0f}")


        

        # x = numpy.array([i for i in range(len(self.china_data.values()))])
        # y = numpy.array(list(self.china_data.values()))
        # #plt.xticks(list(self.us_data.keys()), rotation='vertical')
        # plt.scatter(x,y)


       


    def scatterplot(self,start,end,days_out):
        if start < 0 or end > len(self.us_data.values()):
            print("INVALID START/END")
            return

        dates = list(self.us_data.keys())[start:end+1]
        print(f"From {dates[0]} to {dates[-1]}")


        x = numpy.array([i-start for i in range(start,end+1)])
        
        y = numpy.array(list(self.us_data.values())[start:end+1])
        #plt.xticks(list(self.us_data.keys()), rotation='vertical')
        plt.scatter(x,y)
        print(x)
        print(y)



        print('\n\n')
        model, model_cov = curve_fit(exp_func,x,y)
        
        mar13_model = [0.01187034, 0.29641059]

        # print(f"Coefficients: {model}")
        # print(f"Model Covariance: {model_cov}")
        def gen_model(x):
            return model[0]*numpy.exp(model[1]*x)

        model_y = numpy.array([gen_model(i) for i in x])

        for i in range(len(model_y)):
            print(f"{dates[i]:>8}: Model: {model_y[i]:6.0f}\tActual: {y[i]:6}\tDifference: {model_y[i] - y[i]:5.0f}") 
            # positive means OVERESTIMATE

        plt.plot(x,model_y)

        # def gen_old_model(x):
        #     return mar13_model[0]*numpy.exp(mar13_model[1]*x)
        # y = numpy.array([gen_old_model(i) for i in x])
        # plt.plot(x,y)
        start_num = end+1
        num_days_out = days_out
        
        x = numpy.array([i-start for i in range(start_num,start_num+num_days_out+1)])
        y = numpy.array([gen_model(t) for t in x])
        print(x)
        print(y)


        plt.scatter(x,y)
        

        # x = numpy.array([i for i in range(len(self.china_data.values()))])
        # y = numpy.array(list(self.china_data.values()))
        # #plt.xticks(list(self.us_data.keys()), rotation='vertical')
        # plt.scatter(x,y)


        plt.show()

    






if __name__ == "__main__":
    c = Covid()
    c.display()
    c.predict_confirmed(30,59,14)
    #c.scatterplot(30,58,7)
