"""将运行实验生成的 Markdown 报告转换为可编辑 Word 文档。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / '实验报告.md'
DEST = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / '超声图像增强与降噪实验报告.docx'


def set_run_font(run, name='Microsoft YaHei', size=10.5, bold=None, color='20272E'):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    if bold is not None:
        run.bold = bold
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.rFonts
    if fonts is None:
        fonts = OxmlElement('w:rFonts')
        rpr.insert(0, fonts)
    fonts.set(qn('w:eastAsia'), name)


def shade(cell, fill):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = tcpr.find(qn('w:shd'))
    if shd is None:
        shd = OxmlElement('w:shd')
        tcpr.append(shd)
    shd.set(qn('w:fill'), fill)


def table_borders(table):
    borders = OxmlElement('w:tblBorders')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        tag = OxmlElement('w:' + edge)
        tag.set(qn('w:val'), 'single')
        tag.set(qn('w:sz'), '5')
        tag.set(qn('w:space'), '0')
        tag.set(qn('w:color'), 'D9D9D9')
        borders.append(tag)
    table._tbl.tblPr.append(borders)


def set_cell_margin(cell, top=90, start=100, bottom=90, end=100):
    tc = cell._tc
    tcpr = tc.get_or_add_tcPr()
    margins = tcpr.first_child_found_in('w:tcMar')
    if margins is None:
        margins = OxmlElement('w:tcMar')
        tcpr.append(margins)
    for side, value in [('top', top), ('start', start), ('bottom', bottom), ('end', end)]:
        node = margins.find(qn('w:' + side))
        if node is None:
            node = OxmlElement('w:' + side)
            margins.append(node)
        node.set(qn('w:w'), str(value))
        node.set(qn('w:type'), 'dxa')


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    set_run_font(run, size=8, color='64717A')
    for kind, text in [('begin', None), ('instrText', ' PAGE '), ('end', None)]:
        if kind == 'instrText':
            node = OxmlElement('w:instrText')
            node.set(qn('xml:space'), 'preserve')
            node.text = text
        else:
            node = OxmlElement('w:fldChar')
            node.set(qn('w:fldCharType'), kind)
        run._r.append(node)


def add_table(doc, lines):
    rows = []
    for line in lines:
        cells = [x.strip() for x in line.strip().strip('|').split('|')]
        if cells and all(re.fullmatch(r':?-{3,}:?', x) for x in cells):
            continue
        rows.append(cells)
    if not rows:
        return
    width = max(map(len, rows))
    table = doc.add_table(rows=len(rows), cols=width)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    table_borders(table)
    table.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
    for r, row in enumerate(rows):
        for c in range(width):
            cell = table.cell(r, c)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margin(cell)
            if r == 0:
                shade(cell, 'DCE8EF')
            elif r % 2 == 0:
                shade(cell, 'F5F8FA')
            cell.text = row[c] if c < len(row) else ''
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if c in (0,1) and width > 4 else WD_ALIGN_PARAGRAPH.CENTER
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.line_spacing = 1.15
                if r < len(rows)-1:
                    paragraph.paragraph_format.keep_with_next = True
                for run in paragraph.runs:
                    set_run_font(run, size=8.5, bold=(r == 0), color='000000' if r == 0 else '20272E')
        trpr=table.rows[r]._tr.get_or_add_trPr()
        trpr.append(OxmlElement('w:cantSplit'))
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_paragraph(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(7)
    p.paragraph_format.line_spacing = 1.35
    p.paragraph_format.widow_control = True
    p.paragraph_format.keep_together = True
    run = p.add_run(text)
    set_run_font(run)
    return p


def add_figure(doc, alt, image_path):
    path = (ROOT / image_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f'Missing report image: {path}')
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_together = True
    p.paragraph_format.keep_with_next = True
    width = Inches(3.85 if path.name == 'spatial_split.png' else 6.1)
    inline = p.add_run().add_picture(str(path), width=width)
    inline._inline.docPr.set('descr', alt)
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(12)
    cap.paragraph_format.keep_with_next = False
    r = cap.add_run(alt)
    set_run_font(r, size=9, color='53616A')


def build():
    lines = SOURCE.read_text(encoding='utf-8').splitlines()
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.top_margin, sec.bottom_margin = Inches(.72), Inches(.7)
    sec.left_margin, sec.right_margin = Inches(.82), Inches(.82)
    sec.footer_distance = Inches(.35)
    add_page_number(sec.footer.paragraphs[0])
    styles = doc.styles
    normal = styles['Normal']
    normal.font.name = 'Microsoft YaHei'
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor.from_string('20272E')
    normal._element.rPr.rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')
    for name, size in [('Heading 1',16),('Heading 2',13),('Heading 3',11.5)]:
        style=styles[name]
        style.font.name='Microsoft YaHei';style.font.size=Pt(size);style.font.bold=True
        style.font.color.rgb=RGBColor(0,0,0)
        style._element.rPr.rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
        style.paragraph_format.space_before=Pt(16 if name=='Heading 1' else 12)
        style.paragraph_format.space_after=Pt(6)
        style.paragraph_format.keep_with_next=True
    title_style=styles['Title']
    title_style.font.name='Microsoft YaHei'
    title_style.font.size=Pt(22)
    title_style.font.bold=True
    title_style.font.underline=False
    title_style.font.color.rgb=RGBColor(0,0,0)
    title_style._element.rPr.rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
    title_style.paragraph_format.space_after=Pt(16)
    title_ppr=title_style._element.find(qn('w:pPr'))
    if title_ppr is not None:
        border=title_ppr.find(qn('w:pBdr'))
        if border is not None:
            title_ppr.remove(border)
    index=0
    while index < len(lines):
        line=lines[index].strip()
        if not line:
            index+=1
            continue
        if line.startswith('|'):
            group=[]
            while index < len(lines) and lines[index].strip().startswith('|'):
                group.append(lines[index].strip());index+=1
            add_table(doc,group)
            continue
        if line.startswith('!['):
            match=re.match(r'!\[(.*?)\]\((.*?)\)',line)
            if not match:
                raise ValueError(f'Invalid image markup: {line}')
            add_figure(doc,match.group(1),match.group(2))
            next_text=index+1
            while next_text<len(lines) and not lines[next_text].strip():
                next_text+=1
            if next_text<len(lines) and lines[next_text].strip()==match.group(1):
                index=next_text+1
            else:
                index+=1
            continue
        if line.startswith('# '):
            p=doc.add_paragraph(style='Title')
            p.paragraph_format.space_after=Pt(16)
            ppr=p._p.get_or_add_pPr()
            borders=OxmlElement('w:pBdr')
            for edge in ('top','left','bottom','right','between'):
                element=OxmlElement('w:'+edge)
                element.set(qn('w:val'),'nil')
                borders.append(element)
            ppr.append(borders)
            run=p.add_run(line[2:]);set_run_font(run,size=22,bold=True,color='000000')
            run.font.underline=False
            index+=1
            continue
        if line.startswith('## '):
            doc.add_paragraph(line[3:],style='Heading 1');index+=1;continue
        if line.startswith('### '):
            doc.add_paragraph(line[4:],style='Heading 2');index+=1;continue
        if re.match(r'^\d+\. ',line):
            p=doc.add_paragraph(style='List Number')
            p.paragraph_format.space_after=Pt(4)
            set_run_font(p.add_run(re.sub(r'^\d+\. ', '',line)))
            index+=1
            continue
        add_paragraph(doc,line)
        index+=1
    doc.core_properties.title='超声图像增强与降噪实验报告'
    doc.core_properties.subject='灰度直方图、图像增强与超声图像降噪'
    doc.core_properties.author=''
    doc.save(DEST)
    print(f'Wrote {DEST} ({DEST.stat().st_size:,} bytes)')


if __name__=='__main__':
    build()
