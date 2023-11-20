import win32com.client
#msword = win32.Dispatch('PythonDemos')
#print(msword.tests())
excel = win32com.client.Dispatch('Excel.Application')
workbook = excel.Workbooks.Open('C:\\inetpub\\aigordr\\Info.xlsx')