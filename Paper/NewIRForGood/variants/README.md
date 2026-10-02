# Rendering test — pick the one that displays correctly

Every file below is the same 8-page document. They differ only in how the PDF is
built, each testing a different reason the in-browser viewer might not draw the
figures. Open each in the browser (do not download) and note which ones show the
confusion matrices.

| File | Technique | What it tests |
|:--|:--|:--|
| 01-vector-as-is | plain LaTeX output, figures included as PDF | baseline |
| 02-vector-classic-xref | same, classic cross-reference table, no object streams | parser chokes on PDF 1.5 xref/object streams |
| 03-vector-pdf14 | same, rewritten as PDF 1.4 | older, simpler PDF structure |
| 04-vector-pdf14-classic-xref | both of the above combined | |
| 05-pgf-as-is | figures drawn with native PGF commands in the page stream | viewer skips Form XObjects |
| 06-pgf-classic-xref | 05 + classic cross-reference table | |
| 07-pgf-pdf14 | 05 as PDF 1.4 | |
| 08-pgf-pdf14-classic-xref | 05 with both | |
| 09-flattened-type3 | whole document pushed through PostScript, Type 3 glyphs at 1200 dpi | no XObjects at all |
| 10-all-text-as-outlines | every glyph converted to vector paths, then flattened | viewer cannot handle embedded fonts |
| 11-pgf-flattened-type3 | the PGF build, flattened through PostScript | |
| 12-pages-as-images | each page rendered as one full-page image | viewer only draws images |
| 13-raster-inline-images | figures rasterised and written as **inline** images in the page stream | viewer skips XObjects but draws inline images |
| 14-pdfa | converted to PDF/A-1 | conservative archival profile |
| 15-pgf-pdf13 | the PGF build as PDF 1.3 | oldest structure |
| 16-pgf-linearized | the PGF build, linearised for fast web view | viewer needs progressive loading |
| 17-pgf-inline-images | the PGF build with its colour bars written inline | |

All 17 are 8 pages and render correctly when downloaded.
