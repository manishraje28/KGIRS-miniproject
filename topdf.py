import sys, os, win32com.client
w = win32com.client.DispatchEx("Word.Application"); w.Visible = False
for f in sys.argv[1:]:
    f = os.path.abspath(f); d = w.Documents.Open(f)
    for t in d.TablesOfContents: t.Update()            # fill the TOC page numbers
    d.Save(); d.ExportAsFixedFormat(f[:-5] + ".pdf", 17); d.Close(False)
w.Quit()
