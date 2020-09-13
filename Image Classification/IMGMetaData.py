from hachoir.parser import createParser
from hachoir.metadata import extractMetadata

filename = "IMG.jpg"
parser = createParser(filename)
metadata = extractMetadata(parser)

for line in metadata.exportPlaintext():
    print(line)
