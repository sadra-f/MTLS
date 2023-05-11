class DateParser:
    FULL_DATE_PTRN = "^\d{4}-\d{2}-\d{2}$" # 2008-09-16\\1743.htm.txt, 0, DATE, 2008-09-16
    PRESENT_REF_PTRN = "^PRESENT_REF$" #2008-09-07\\1107.htm.txt, 54, DATE, PRESENT_REF
    PAST_REF_PTRN = "^PAST_REF$" #2008-09-17\\453.htm.txt, 21, DATE, PAST_REF
    YEAR_PTRN = "^\d{4}$" # 2008-09-16\\166.htm.txt, 0, DATE, 2021
    YEAR_MONTH_PTRN = "^\d{4}-\d{2}$" #2008-09-17\\1431.htm.txt, 1, DATE, 2023-05
    YEAR_WEEK_PTRN = "^\d{4}-W\d{1,2}$" #2008-09-16\\1743.htm.txt, 17, DATE, 2023-W18
    YEAR_WEEK_WEEKEND_PTRN = "^\d{4}-W\d{2}-WE$" #2008-09-15\\103.htm.txt, 0, DATE, 2023-W19-WE
    DECADE_PTRN = "^\d{3}X?$" #2008-09-15\\103.htm.txt, 25, DATE, 201 | 2008-09-12\\67.htm.txt, 6, DATE, 201X
    FULL_TIME_PTRN = "^\d{4}-\d{2}-\d{2}T\w+" # 2008-09-16\\1743.htm.txt, 0, TIME, 2008-09-16T16:30 | 2008-09-15\\103.htm.txt, 10, TIME, 2023-05-11TMO

    def extract_date(date_str:str):
        pass