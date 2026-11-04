"""Small dependency-free terminal presentation layer for the guided review CLI.

All formatting is local. Data, account validation, source handling and PLN stay
in their existing services. NO_COLOR, pipes and narrow terminals are supported.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import textwrap
import unicodedata

ANSI=re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
PALETTE={'accent':'36','title':'1;36','bold':'1','muted':'90','success':'32','warning':'33','danger':'31'}
STATUS={'more_support':'success','needs_review':'warning','lower_support':'danger','unreviewed':'muted'}


def clean(value):
    value=ANSI.sub('',str(value))
    return re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ',value)


def visible_width(value):
    return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in ('W','F') else 1
               for c in ANSI.sub('',str(value)))


def clip(value,width):
    value=clean(value)
    if visible_width(value)<=width:return value
    out=''
    for char in value:
        if visible_width(out+char)>max(0,width-1):break
        out+=char
    return out+'…' if width else ''


def confidence(value):
    if value is None:return '—'
    rounded=f'{value:.3f}'
    return '<1.000' if value<1 and rounded=='1.000' else rounded


def strength(value):return '—' if value is None else f'{100*value:.1f}%'


class Terminal:
    def __init__(self,emit=None,*,interactive=None,color=None):
        self.emit=emit or (lambda value='':print(value,flush=True))
        self.interactive=sys.stdout.isatty() if interactive is None else interactive
        self.color=(self.interactive and 'NO_COLOR' not in os.environ and os.getenv('TERM')!='dumb') if color is None else color
        self.can_clear=self.interactive and os.getenv('TERM')!='dumb' and 'FEEDBACK_NO_CLEAR' not in os.environ

    @property
    def width(self):return max(40,min(112,shutil.get_terminal_size((100,32)).columns-2))

    @property
    def height(self):return max(20,shutil.get_terminal_size((100,32)).lines)

    def paint(self,value,kind='accent'):
        return '\x1b['+PALETTE.get(kind,'0')+'m'+str(value)+'\x1b[0m' if self.color else str(value)

    def raw(self,value=''):
        self.lines=getattr(self,'lines',0)+max(1,len(str(value).splitlines()))
        self.emit(value)

    def text(self,value='',indent=0,kind=None):
        if not value:self.raw();return
        prefix=' '*min(indent,8)
        lines=textwrap.wrap(clean(value),width=max(12,self.width-len(prefix)),break_long_words=True,break_on_hyphens=False) or ['']
        for line in lines:self.raw(prefix+(self.paint(line,kind) if kind else line))

    def begin(self,title,subtitle='',compact=False):
        self.lines=0
        prefix='\x1b[2J\x1b[H' if self.can_clear else ''
        self.raw(prefix+self.paint('ALIS + OMEGA','title')+'  '+self.paint('RESEARCH REVIEW','muted'))
        self.raw(self.paint('─'*self.width,'muted'))
        if compact:
            self.text(title+(' · '+subtitle if subtitle else ''),kind='bold')
        else:
            self.text(title,kind='bold')
            if subtitle:self.text(subtitle,kind='muted')
            self.raw()

    def section(self,title):self.raw();self.text(title,kind='accent')

    def menu(self,choices):
        self.raw()
        for key,label in choices:self.text('['+str(key)+']  '+label,2)

    def rule(self):self.raw(self.paint('─'*self.width,'muted'))

    def table(self,headers,rows,widths,kinds=None):
        def line(cells,header=False,style=None):
            parts=[]
            for index,(cell,width) in enumerate(zip(cells,widths)):
                value=clip(cell,width);value+=' '*max(0,width-visible_width(value))
                kind='muted' if header else (style or {}).get(index)
                parts.append(self.paint(value,kind) if kind else value)
            self.raw('  '.join(parts).rstrip())
        line(headers,header=True)
        for i,row in enumerate(rows):line(row,style=(kinds[i] if kinds else None))

    def entries(self,rows):
        if not rows:return
        if self.width>=70:
            widths=[3,self.width-53,15,8,10,7]
            self.table(['ID','Intervention','Status','Strength','Confidence','Reviews'],
                       [[r['entry_number'],r['name'],r['routing']['label'],strength(r['appraisal']['strength']),
                         confidence(r['appraisal']['confidence']),str(r['routing']['decisive'])+'/'+str(r['routing']['total'])] for r in rows],
                       widths,[{2:STATUS[r['routing']['status']]} for r in rows])
        else:
            for r in rows:
                t=r['routing'];self.text(str(r['entry_number'])+'. '+r['name'],kind='bold')
                self.text(t['label']+' · strength '+strength(r['appraisal']['strength'])+' · confidence '+confidence(r['appraisal']['confidence']),2,
                          kind=STATUS[t['status']])

    def metrics(self,a,t,compact=False):
        values='Strength '+strength((a or {}).get('strength'))+'    Count confidence '+confidence((a or {}).get('confidence'))
        if compact:self.text(t['label']+'  |  '+values,kind=STATUS[t['status']])
        else:
            self.text(t['label'],kind=STATUS[t['status']]);self.text(values)
        self.text(f"{t['support']} Support  ·  {t['oppose']} Oppose  ·  {t['uncertain']} Uncertain  |  {t['decisive']}/{t['total']} decisive",kind='muted')

    def delta(self,before,after):
        rows=[('Strength',strength((before or {}).get('strength')),strength((after or {}).get('strength'))),
              ('Count confidence',confidence((before or {}).get('confidence')),confidence((after or {}).get('confidence')))]
        self.table(['','Before:','After:'],rows,[min(21,self.width-24),10,10])

    def document(self,title,body,subtitle=''):
        """Readable source/trace pages. Noninteractive output keeps the full text."""
        lines=[]
        for paragraph in str(body).splitlines():
            lines.extend(textwrap.wrap(clean(paragraph),width=self.width,break_long_words=True,break_on_hyphens=False) if paragraph else [''])
        size=max(8,self.height-9)
        if not self.interactive:
            self.begin(title,subtitle)
            for line in lines:self.raw(line)
            return
        page=0;last=max(0,(len(lines)-1)//size)
        while True:
            self.begin(title,subtitle)
            for line in lines[page*size:(page+1)*size]:self.raw(line)
            self.raw()
            navigation='[Enter/n] Next  [p] Previous  [0] Back' if page<last else '[Enter/0] Back  [p] Previous'
            self.text('Page '+str(page+1)+'/'+str(last+1)+'  ·  '+navigation,kind='muted')
            answer=input('Read > ').strip().casefold()
            if answer in ('0','q','back') or (not answer and page==last):return
            if answer=='p':page=max(0,page-1)
            elif answer in ('','n'):page=min(last,page+1)
