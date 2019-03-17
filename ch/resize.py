from os import listdir
from os.path import isfile, join


path = '/home/ochiroo/Desktop/last_data/'
files = [f for f in listdir(path) if isfile(join(path, f))]


import PIL
from PIL import Image


basewidth = [600]

for rwidth in basewidth:
    print('Resize: ' + str(rwidth) + 'px starting...')
    for f in files:
        img = Image.open(path + f)
        wpercent = (rwidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((rwidth, hsize), PIL.Image.ANTIALIAS)
        # img = img.rotate(180)
        img.save('images_' + str(rwidth) + '_b' + f)
    print('Resize: ' + str(rwidth) + 'px finished')
