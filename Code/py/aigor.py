class aigor:
	_public_methods_ = [ 'tests' ]
	_reg_progid_ = "PythonDemos"
	_reg_clsid_ = "{3EE7C150-9A18-4B97-86FB-09EA2C569074}"
	def tests(self):
		return "Hello World！"
#import pythoncom
#print(pythoncom.CreateGuid())		
if __name__ == '__main__':
	import win32com.server.register
	win32com.server.register.UseCommandLine(aigor)