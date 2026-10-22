"""Data-driven SVG/PNG figures and a short, explicitly non-live walkthrough.

SVG construction uses the standard library. PNG/GIF export additionally needs Pillow.
"""
from __future__ import annotations
import argparse
from html import escape
import json
import math
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BG='#f5f7fa'; INK='#142234'; MUTED='#566779'; GRID='#dbe3eb'
TEAL='#087f8c'; BLUE='#4656c9'; GREEN='#76b900'; WHITE='#ffffff'; ORANGE='#b66a18'


class Canvas:
    def __init__(self,title,w=1200,h=700,png=False,bg=BG):
        self.w,self.h,self.parts,self.png=w,h,[],png
        self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img"><title>{escape(title)}</title>']
        if png:
            from PIL import Image, ImageDraw
            self.image=Image.new('RGB',(w*2,h*2),bg); self.draw=ImageDraw.Draw(self.image)
        self.rect(0,0,w,h,bg)

    def rect(self,x,y,w,h,fill,rx=0):
        self.parts.append(f'<rect x="{x:.3f}" y="{y:.3f}" width="{w:.3f}" height="{h:.3f}" rx="{rx}" fill="{fill}"/>')
        if self.png:
            self.draw.rounded_rectangle((x*2,y*2,(x+w)*2,(y+h)*2),radius=rx*2,fill=fill)

    def line(self,x,y,x2,y2,color=GRID,width=1,dash=False):
        self.parts.append(f'<line x1="{x:.3f}" y1="{y:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" stroke="{color}" stroke-width="{width}"'+(' stroke-dasharray="7 6"' if dash else '')+'/>')
        if self.png:
            if dash:
                n=max(1,int(math.hypot(x2-x,y2-y)/10))
                for i in range(0,n,2):
                    a,b=i/n,min((i+1)/n,1)
                    self.draw.line(((x+(x2-x)*a)*2,(y+(y2-y)*a)*2,(x+(x2-x)*b)*2,(y+(y2-y)*b)*2),fill=color,width=max(1,int(width*2)))
            else: self.draw.line((x*2,y*2,x2*2,y2*2),fill=color,width=max(1,int(width*2)))

    def circle(self,x,y,r,fill):
        self.parts.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{r}" fill="{fill}"/>')
        if self.png: self.draw.ellipse(((x-r)*2,(y-r)*2,(x+r)*2,(y+r)*2),fill=fill)

    def text(self,x,y,s,size=20,color=INK,bold=False,anchor='start'):
        s=str(s)
        self.parts.append(f'<text x="{x:.3f}" y="{y:.3f}" font-family="Arial, DejaVu Sans, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}" text-anchor="{anchor}">{escape(s)}</text>')
        if self.png:
            from PIL import ImageFont
            candidates=[os.environ.get('SHOWCASE_FONT',''),
                'C:/Windows/Fonts/segoeuib.ttf' if bold else 'C:/Windows/Fonts/segoeui.ttf',
                '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
            f=next((p for p in candidates if p and Path(p).exists()),None)
            font=ImageFont.truetype(f or 'DejaVuSans.ttf',int(size*2))
            self.draw.text((x*2,y*2),s,font=font,fill=color,anchor={'start':'ls','middle':'ms','end':'rs'}[anchor])

    def svg(self): return '\n'.join(self.parts+['</svg>'])+'\n'

    def heading(self,section,title,subtitle):
        self.rect(42,30,32,5,GREEN)
        self.text(86,40,section.upper(),13,MUTED,True)
        self.text(42,94,title,36,INK,True)
        self.text(42,130,subtitle,18,MUTED)

    def footer(self,s):
        self.line(42,self.h-57,self.w-42,self.h-57)
        self.text(42,self.h-26,s,14,MUTED)


def hero(s,png=False):
    c=Canvas('Multi-GPU systems: measure, verify, explain',1200,570,png,bg='#102235')
    c.rect(44,38,38,6,GREEN)
    c.text(98,48,'MEHRAN KEIVANIMEHR  /  AI INFRASTRUCTURE',15,'#bcd1df',True)
    c.text(44,126,'Measure the system.',53,WHITE,True)
    c.text(44,188,'Verify the computation.',53,WHITE,True)
    c.text(44,235,'Distributed training · serving · numerical controls · reproducible evidence',20,'#bcd1df')
    gain=s['a6000']['serving'][0]['dp2_advantage_pct']
    verdicts=s['a6000']['calibration']['verdicts']; passed=sum(v=='pass' for v in verdicts.values())
    sizes=s['titan']['training_world_sizes']
    cards=[(f'{gain:.2f}%','DP2 over TP2','Qwen1.5B · A6000 · historical'),
           (f'{passed} / {len(verdicts)}','calibration endpoints pass','CPU-replayable fixed slices'),
           (f'{min(sizes)}–{max(sizes)} GPUs','conditional update checks','Titan/1080 · one audited boundary')]
    for i,(a,b,d) in enumerate(cards):
        x=44+375*i
        c.rect(x,290,350,190,'#1b3349',12)
        c.text(x+22,351,a,42, '#b4e866' if i==0 else WHITE,True)
        c.text(x+22,398,b,21,WHITE,True)
        c.text(x+22,442,d,15,'#bcd1df')
    c.text(44,535,'Recorded results, not a universal GPU ranking. Scope and replay commands are in the repository.',16,'#bcd1df')
    return c


def serving(s,png=False):
    c=Canvas('A6000 serving throughput at matched concurrency',1200,740,png)
    c.heading('A6000 / serving','The request budget changes the comparison.','Two GPUs · aggregate cap 512 · mean and both repeats · separate panel scales')
    for j,r in enumerate(s['a6000']['serving']):
        x=42+j*565; c.rect(x,170,548,480,WHITE,12)
        c.text(x+24,208,'Qwen2.5-1.5B' if j==0 else 'Llama-3.1-8B',25,INK,True)
        c.text(x+24,241,f"+{r['dp2_advantage_pct']:.2f}% DP2 throughput",21,TEAL,True)
        top,bottom=300,559; ymax=18000 if j==0 else 4000
        for v in [0,ymax/4,ymax/2,3*ymax/4,ymax]:
            y=bottom-(bottom-top)*v/ymax
            c.line(x+75,y,x+510,y)
            c.text(x+64,y+5,f'{v/1000:g}k',14,MUTED,anchor='end')
        for i,(name,color) in enumerate([('TP2',BLUE),('DP2',TEAL)]):
            a=r['arms'][name]; bx=x+145+i*205; height=(bottom-top)*a['mean_tok_s']/ymax
            c.rect(bx,bottom-height,90,height,color,4)
            c.text(bx+45,bottom-height-18,f"{a['mean_tok_s']:,.0f}",20,INK,True,'middle')
            for k,v in enumerate(a['values_tok_s']):
                yy=bottom-(bottom-top)*v/ymax
                c.circle(bx+36+18*k,yy,5,WHITE); c.circle(bx+36+18*k,yy,2.5,INK)
            c.text(bx+45,591,name,21,INK,True,'middle')
        c.text(x+24,628,'Output tokens/s · 512 prompt / 256 output tokens',15,MUTED)
    c.footer('Historical, two repeats per arm. Llama8B is not revalidated. Source: serving comparison CSV + raw result.json.')
    return c


def precision(s,png=False):
    c=Canvas('A6000 first-boundary precision-policy contrasts',1200,690,png)
    c.heading('A6000 / numerical diagnostics','Precision policy matters.','One boundary · matched TF32-off / math-SDPA policy · zero optimizer updates')
    labels=['FP32, permuted accumulation','FP32 parameters + BF16 autocast','Native BF16 policy','DeepSpeed BF16 policy']
    rows={r['label']:r for r in s['a6000']['precision']['rows']}
    left,right=420,1085
    def xx(v): return left+(math.log10(v)+8)/8*(right-left)
    for exp in (-8,-6,-4,-2,0):
        x=xx(10**exp); c.line(x,182,x,529); c.text(x,561,f'10^{exp}',15,MUTED,anchor='middle')
    for i,label in enumerate(labels):
        r=rows[label]; y=228+i*86; color=TEAL if i<2 else BLUE
        c.text(42,y+6,label,19,INK,True)
        c.line(left,y,xx(r['rel_l2']),y,color,9); c.circle(xx(r['rel_l2']),y,8,color)
        c.text(xx(r['rel_l2'])+16,y+6,f"{r['rel_l2']:.3g}",20,INK,True)
    c.text(left,601,'Relative L2 from the FP32 control (log scale)',17,MUTED)
    c.footer('Native / DeepSpeed BF16 recorded hashes match for 338 tensors. This is a specific contrast, not framework certification.')
    return c


def calibration(s,png=False):
    c=Canvas('Replayable five-update A6000 calibration',1200,660,png)
    c.heading('A6000 / control C','A control you can replay.','Same 20 global inputs · FP32 DDP · one GPU / accumulation 4 vs two GPUs / accumulation 2')
    cal=s['a6000']['calibration']; left,right=475,1090
    def xx(v): return left+(math.log10(v)+5)/5*(right-left)
    for exp in (-5,-4,-3,-2,-1,0):
        x=xx(10**exp); c.line(x,192,x,480,ORANGE if exp==0 else GRID,2 if exp==0 else 1,dash=exp==0)
        c.text(x,513,f'{10**exp:g}',15,MUTED,anchor='middle')
    for i,(label,key) in enumerate([('First-boundary gradient','G1_rel_l2'),('Maximum five-step loss difference','L_max_k0_4_rel'),('Five-update parameter change','U_rel_l2')]):
        y=245+i*92; ratio=cal['endpoints'][key]/cal['limits'][key][1]
        c.text(42,y+5,label,20,INK,True)
        c.circle(xx(ratio),y,9,TEAL)
        c.text(xx(ratio)+18,y+6,f'{ratio:.4g}',19,TEAL,True)
    c.text(475,551,'Observed relative error / frozen upper limit (log scale)',16,MUTED)
    c.text(42,551,'All six endpoints pass.',22,TEAL,True)
    c.footer('Three upper-bound endpoints shown; cosine endpoints also pass. Fixed slices: 787,968 elements × four arrays, 12.6 MB.')
    return c


def scaling(s,png=False):
    c=Canvas('Historical GTX 1080 Ti GPT-2 weak scaling',1200,720,png)
    c.heading('GTX 1080 Ti / historical performance','More GPUs. Less than linear speedup.','GPT-2 FP32 DDP · sequence 512 · batch 4 per GPU · same-family baseline')
    rows=s['titan']['historical_weak_scaling']; base=s['titan']['historical_baseline_tokens_s']
    left,right,top,bottom=110,1090,190,574
    xx=lambda n:left+(n-1)/5*(right-left)
    yy=lambda v:bottom-v/44000*(bottom-top)
    for v in (0,10000,20000,30000,40000):
        c.line(left,yy(v),right,yy(v)); c.text(left-18,yy(v)+5,f'{v/1000:g}k',16,MUTED,anchor='end')
    c.line(xx(1),yy(base),xx(6),yy(base*6),'#9baab6',2,True)
    pts=[(1,base,100)]+[(r['gpus'],r['tokens_s'],r['efficiency_pct']) for r in rows]
    for a,b in zip(pts,pts[1:]): c.line(xx(a[0]),yy(a[1]),xx(b[0]),yy(b[1]),TEAL,4)
    for n,v,e in pts:
        c.circle(xx(n),yy(v),7,TEAL)
        if n==1: c.text(xx(n)+18,yy(v)+30,'100% baseline',17,TEAL,True)
        else: c.text(xx(n),yy(v)-23,f'{e:.2f}%',20,TEAL,True,'middle')
        c.text(xx(n),bottom+32,n,18,INK,anchor='middle')
    c.text(110,636,'GPUs · observed token/s; labels show weak-scaling efficiency',19,MUTED)
    c.text(836,203,'- - ideal linear rate',17,MUTED)
    c.footer('Selected historical point records; no repeat intervals implied. Later update diagnostics are a separate experiment.')
    return c


def architecture(s,png=False):
    c=Canvas('The measurement-to-release evidence path',1200,520,png)
    c.heading('Engineering / evidence flow','From a run to a defensible result.','Recorded observations and reproducible checks have different responsibilities.')
    boxes=[('01','MEASURE',['Pinned recipe and inputs','Per-rank identities','Raw results and failures']),
           ('02','RECONCILE',['Expected / observed coverage','Optimizer-boundary evidence','Explicit refusal states']),
           ('03','REPLAY',['Fixed slices and logits','CPU numerical comparison','Checksums and negative tests']),
           ('04','PUBLISH',['Traceable figures and tables','Scoped conclusions','Portable verification'])]
    for i,(num,title,lines) in enumerate(boxes):
        x=42+i*287
        c.rect(x,197,260,211,WHITE,12); c.text(x+22,236,num,25,TEAL,True)
        c.text(x+22,278,title,21,INK,True)
        for j,line in enumerate(lines): c.text(x+22,317+j*29,line,15,MUTED)
        if i<3: c.text(x+269,305,'›',28,TEAL,True)
    c.footer('Architecture summary of the supplied implementation, not a claim that every historical run exercised every safeguard.')
    return c


FIGURES={'overview':hero,'a6000_serving':serving,'a6000_precision':precision,'a6000_calibration':calibration,
         'titan_scaling':scaling,'evidence_flow':architecture}


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--check',action='store_true'); ap.add_argument('--svg-only',action='store_true'); a=ap.parse_args()
    s=json.loads((ROOT/'results/showcase.json').read_text()); out=ROOT/'assets/figures'; out.mkdir(parents=True,exist_ok=True)
    for name,fn in FIGURES.items():
        c=fn(s,png=not(a.check or a.svg_only)); b=c.svg().encode()
        p=out/(name+'.svg')
        if a.check:
            if not p.exists() or p.read_bytes()!=b: raise SystemExit('Stale figure: '+name)
        else:
            p.write_bytes(b)
            if not a.svg_only: c.image.save(out/(name+'.png'),dpi=(192,192))
    if not a.check and not a.svg_only:
        frames=[]
        for name in ('a6000_serving','a6000_precision','a6000_calibration','evidence_flow'):
            c=FIGURES[name](s,png=True)
            c.text(c.w-42,40,'EVIDENCE WALKTHROUGH · NOT LIVE TELEMETRY',12,MUTED,anchor='end')
            from PIL import Image
            frame=Image.new('RGB',(1200,740),BG)
            im=c.image.resize((1200,c.h),Image.Resampling.LANCZOS); frame.paste(im,(0,0))
            frames.append(frame.quantize(colors=128))
        frames[0].save(out/'evidence_walkthrough.gif',save_all=True,append_images=frames[1:],duration=3200,loop=0,disposal=2)
    print('Figure sources verified.' if a.check else 'Rendered six figures and the evidence walkthrough.')


if __name__=='__main__': main()
