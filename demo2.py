from digit_interface import Digit
from digit_interface import DigitHandler
import cv2
 
d = Digit("D20782") # Unique serial number
d.connect()
d.show_view()
d.disconnect()
digits = DigitHandler.list_digits()