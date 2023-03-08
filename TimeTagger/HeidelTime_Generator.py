from TimeTagger import HTW

def ht(input, reformat:bool=True):
    '''
        transforms a text or a list of text into a time tagged version of it
    '''
    hw = HTW('english', reformat_output=reformat)
    res_list = []
    if type(input) == list:
        for string in input:
            res_list.append(hw.parse(string))
        return res_list
    elif type(input) == str:
        return hw.parse(input)
    else:
        raise TypeError