import sys
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.layout import LAParams
import io
import re

def pdfparser(data):

    fp = open(data, 'rb')
    rsrcmgr = PDFResourceManager()
    retstr = io.StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr,laparams=laparams)
    # Create a PDF interpreter object.
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    # Process each page contained in the document.
    data=""
    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
        data=retstr.getvalue()
        par=re.split('\s{4,}',data)
        print(par)

    # print(data)

fp=r"../../corpus/bpkhackathon/telkom/FS Q12018_INDONESIA.pdf"
pdfparser(fp)