#!/usr/bin/env python

import math
import moviepy
import moviepy.editor
import numpy
import PIL
import PIL.Image
import PIL.ImageFont
import PIL.ImageDraw

intX = 32
intY = 436 - 64

objImages = [ {
	'strFile': 'official - caffe.png',
	'strText': 'official - Caffe'
}, {
	'strFile': 'this - pytorch.png',
	'strText': 'this - PyTorch'
} ]

npyImages = []

for objImage in objImages:
	objOutput = PIL.Image.open(objImage['strFile']).convert('RGB')

	for intU in [ intShift - 10 for intShift in range(20) ]:
		for intV in [ intShift - 10 for intShift in range(20) ]:
			if math.sqrt(math.pow(intU, 2.0) + math.pow(intV, 2.0)) <= 5.0:
				PIL.ImageDraw.Draw(objOutput).text((intX + intU, intY + intV), objImage['strText'], (255, 255, 255), PIL.ImageFont.truetype('freefont/FreeSerifBold.ttf', 32))
			# end
		# end
	# end

	PIL.ImageDraw.Draw(objOutput).text((intX, intY), objImage['strText'], (0, 0, 0), PIL.ImageFont.truetype('freefont/FreeSerifBold.ttf', 32))

	npyImages.append(numpy.array(objOutput))
# end

moviepy.editor.ImageSequenceClip(sequence=npyImages, fps=1).write_gif(filename='comparison.gif', program='ImageMagick', opt='optimizeplus')