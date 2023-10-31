
import numpy as np
print(np.random.rand(1))

import matplotlib.font_manager as fm

fonts = fm.findSystemFonts()
for font in fonts:
    print(font)
# font_names = [fm.get_font(font).family_name for font in fonts]
#
# print(font_names)
