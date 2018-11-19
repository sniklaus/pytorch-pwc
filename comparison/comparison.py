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

objectImages = [ {
	'strFile': 'official - caffe.png',
	'strText': 'official - Caffe'
}, {
	'strFile': 'this - pytorch.png',
	'strText': 'this - PyTorch'
} ]

numpyImages = []

for objectImage in objectImages:
	objectOutput = PIL.Image.open(objectImage['strFile']).convert('RGB')

	for intU in [ intShift - 10 for intShift in range(20) ]:
		for intV in [ intShift - 10 for intShift in range(20) ]:
			if math.sqrt(math.pow(intU, 2.0) + math.pow(intV, 2.0)) <= 5.0:
				PIL.ImageDraw.Draw(objectOutput).text((intX + intU, intY + intV), objectImage['strText'], (255, 255, 255), PIL.ImageFont.truetype('freefont/FreeSerifBold.ttf', 32))
			# end
		# end
	# end

	PIL.ImageDraw.Draw(objectOutput).text((intX, intY), objectImage['strText'], (0, 0, 0), PIL.ImageFont.truetype('freefont/FreeSerifBold.ttf', 32))

	numpyImages.append(numpy.array(objectOutput))
# end

moviepy.editor.ImageSequenceClip(sequence=numpyImages, fps=1).write_gif(filename='comparison.gif', program='ImageMagick', opt='optimizeplus')