import re
from datetime import date

class DateParser: #the specificity goes as high as days
    FULL_DATE_PTRN = "^\d{4}-\d{2}-\d{2}$" # 2008-09-16\\1743.htm.txt, 0, DATE, 2008-09-16
    FULL_TIME_PTRN = "^\d{4}-\d{2}-\d{2}T(\w+|[a-z0-9:]$" # 2008-09-16\\1743.htm.txt, 0, TIME, 2008-09-16T16:30 | 2008-09-15\\103.htm.txt, 10, TIME, 2023-05-11TMO
    
    PRESENT_REF_PTRN = "^PRESENT_REF$" #2008-09-07\\1107.htm.txt, 54, DATE, PRESENT_REF
    PAST_REF_PTRN = "^PAST_REF$" #2008-09-17\\453.htm.txt, 21, DATE, PAST_REF

    YEAR_PTRN = "^\d{4}$" # 2008-09-16\\166.htm.txt, 0, DATE, 2021
    YEAR_MONTH_PTRN = "^\d{4}-\d{2}$" #2008-09-17\\1431.htm.txt, 1, DATE, 2023-05
    YEAR_WEEK_PTRN = "^\d{4}-W\d{1,2}$" #2008-09-16\\1743.htm.txt, 17, DATE, 2023-W18
    YEAR_WEEK_WEEKEND_PTRN = "^\d{4}-W\d{2}-WE$" #2008-09-15\\103.htm.txt, 0, DATE, 2023-W19-WE

    DECADE_PTRN = "^\d{3}X?$" #2008-09-15\\103.htm.txt, 25, DATE, 201 | 2008-09-12\\67.htm.txt, 6, DATE, 201X

    DEFAULT_MONTH = 6
    DEFAULT_DAY = 15



    def __init__(self, current_date:date):
        self.current_date = current_date

    def extract_date(self, date_str:str) -> date:
        res = self.current_date
        try:
            if re.search(DateParser.FULL_DATE_PTRN, date_str) is not None:
                values = date_str.split('-')
                res = date(int(values[0]), int(values[1]), int(values[2]))

            elif re.search(DateParser.FULL_TIME_PTRN, date_str) is not None:
                values = date_str[0:10].split('-')
                res = date(int(values[0]), int(values[1]), int(values[2]))

            elif re.search(DateParser.YEAR_PTRN, date_str) is not None:
                values = [date_str]
                res = date(int(values[0]), DateParser.DEFAULT_MONTH, DateParser.DEFAULT_DAY)
            
            elif re.search(DateParser.YEAR_MONTH_PTRN, date_str) is not None:
                values = date_str.split('-')
                res = date(int(values[0]), int(values[1]), DateParser.DEFAULT_DAY)

            elif re.search(DateParser.YEAR_WEEK_PTRN, date_str) is not None:
                values = date_str.split('-')
                res = date(int(values[0]), int(values[2][1:]) / 4, DateParser.DEFAULT_DAY)
            
            elif re.search(DateParser.YEAR_WEEK_WEEKEND_PTRN, date_str) is not None:
                values = date_str.split('-')
                res = date(int(values[0]), int(values[2][1:]) / 4, DateParser.DEFAULT_DAY)

            elif re.search(DateParser.DECADE_PTRN, date_str) is not None:
                values = [date_str[0:3] + '5']
                res = date(int(values[0], DateParser.DEFAULT_MONTH, DateParser.DEFAULT_DAY))
        except ValueError as err:
            print("Error parsing date value: { ", date_str, ' }')
        finally:
            return res
            
            