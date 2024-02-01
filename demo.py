from digit_interface import Digit
from digit_interface import DigitHandler
 
d = Digit("D20790") # Unique serial number
d.connect()
#d.set_intensity(intensity=15)
d.show_view()
d.disconnect()
digits = DigitHandler.list_digits()

