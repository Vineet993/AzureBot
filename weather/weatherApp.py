import pyowm
from config.config_reader import ConfigReader
import pandas as pd


class WeatherInformation():
    """
    def __init__(self):
        self.config_reader = ConfigReader()
        self.configuration = self.config_reader.read_config()
        self.owmapikey = self.configuration['WEATHER_API_KEY']
        self.owm = pyowm.OWM(self.owmapikey)
    """

    def get_weather_info(self, city):
        self.city = city
        temp_max_celsius = 'cels'
        humidity = 'humo'
        temp_min_celsius = 'max'

        # self.bot_says = "Today the weather in " + city + " is :\n Maximum Temperature :" + temp_max_celsius + " Degree Celsius" + ".\n Minimum Temperature :" + temp_min_celsius + " Degree Celsius" + ": \n" + "Humidity :" + humidity + "%"
        lst = ['VI', 'NE', 'ET', 'KU',
               'MA', 'R', 'ME', 'SSI']

        # Calling DataFrame constructor on list

        self.bot_says = pd.DataFrame(lst)
        return self.bot_says

    """
        observation = self.owm.weather_at_place(city)
        w = observation.get_weather()
        latlon_res = observation.get_location()
        lat = str(latlon_res.get_lat())
        lon = str(latlon_res.get_lon())

        wind_res = w.get_wind()
        wind_speed = str(wind_res.get('speed'))

        humidity = str(w.get_humidity())

        celsius_result = w.get_temperature('celsius')
        temp_min_celsius = str(celsius_result.get('temp_min'))
        temp_max_celsius = str(celsius_result.get('temp_max'))

        fahrenheit_result = w.get_temperature('fahrenheit')
        temp_min_fahrenheit = str(fahrenheit_result.get('temp_min'))
        temp_max_fahrenheit = str(fahrenheit_result.get('temp_max'))
    
        temp_max_celsius = 22
        humidity =55
        
        
    
        self.bot_says = "Today the weather in " + city +" is :\n Maximum Temperature :"+temp_max_celsius+ " Degree Celsius"+".\n Minimum Temperature :"+temp_min_celsius+ " Degree Celsius" +": \n" + "Humidity :" + humidity + "%"
        return self.bot_says
    """
