
import traceback
import sys


def logging_decorator(func):
	def wrapper(self, x):
		print("Calling function ", str(func))
		# traceback.print_stack(file=sys.stdout)
		with open("logs.txt", "a") as f:					
			for line in traceback.format_stack():
				text = line.strip() + '\n'
				f.write(text)
		return func(self, x)		
		
	return wrapper

