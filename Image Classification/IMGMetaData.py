# This is a simple example of image metadata extraction of images
from hachoir.parser import createParser
from hachoir.metadata import extractMetadata

filename = "IMG.jpg"
parser = createParser(filename)
metadata = extractMetadata(parser)
# Convert image metadata to text

for line in metadata.exportPlaintext():
    print(line)
