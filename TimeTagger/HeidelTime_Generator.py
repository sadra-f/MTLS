from TimeTagger import HTW
from models.TStr import TStr
from datetime import date
def ht(input, reformat:bool=True, date:date=None):
    '''
        transforms a text or a list of text into a time tagged version of it
    '''
    hw = HTW('english', reformat_output=reformat)
    res_list = []
    if type(input) == list:
        for i in range(len(input)):
            res_list.append(hw.parse(input[i], date_ref=date))
        return res_list
    elif type(input) == (str or TStr):
        return hw.parse(input)
    else:
        raise TypeError