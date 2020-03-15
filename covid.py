import matplotlib.pyplot as plt 
import csv 
import numpy
import math
from scipy.optimize import curve_fit


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

    def scatterplot(self):
        x = numpy.array([i for i in range(40,len(self.us_data.values()))])
        y = numpy.array(list(self.us_data.values())[40:])
        #plt.xticks(list(self.us_data.keys()), rotation='vertical')
        plt.scatter(x,y)
        print(x)
        print(y)
        model, model_cov = curve_fit(exp_func,x,y)
        
        print(f"Coefficients: {model}")
        print(f"Model Covariance: {model_cov}")
        def gen_model(x):
            return model[0]*math.exp(model[1]*x)

        y = numpy.array([gen_model(i) for i in x])

        plt.plot(x,y)
        
        start_num = len(self.us_data.values())
        num_days_out = 25
        x = numpy.array([i for i in range(start_num,start_num+num_days_out)])
        y = numpy.array([gen_model(t) for t in x])
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
    c.scatterplot()