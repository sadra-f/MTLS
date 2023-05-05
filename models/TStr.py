from datetime import date

class TStr(str):
    def set_date(self, date):
        self.date = date


    def get_date(self)->date:
        return self.date