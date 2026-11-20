from pathlib import Path
from xml.sax.saxutils import escape
import argparse
import subprocess

W=720
ROOT=Path(__file__).resolve().parent
BG='#f3f6fc'; CARD='#ffffff'; INK='#14283e'; MUT='#48617b'; MID='#59738b'; LINE='#ced9e8'
BLUE='#145db2'; BS='#e8f1fe'; TEAL='#007960'; TS='#e6f7f0'
OR='#a7520c'; OS='#fff2df'; PUR='#6145ab'; PS='#f0ebff'
V={BLUE:BS,TEAL:TS,OR:OS,PUR:PS,MID:'#edf2f8'}
ALL={}
class Fig:
 def __init__(self,num,title,sub,h):
  self.e=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" viewBox="0 0 {W} {h}" role="img"><title>{escape(title)}</title><desc>{escape(sub)}</desc><defs><marker id="a" markerWidth="8" markerHeight="8" viewBox="0 0 10 10" refX="9" refY="5" orient="auto"><path d="M0,0 10,5 0,10z" fill="#59738b"/></marker></defs>']
  self.r(0,0,W,h,BG)
  self.t(30,40,'MIRAGE   /   ENGINEERING ATLAS   /   '+num,16,BLUE,True)
  self.t(29,88,title,36,INK,True)
  self.t(31,122,sub,19,MUT)
  self.h=h
 def r(self,x,y,w,h,fill=CARD,stroke='none',rx=0):
  self.e.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="1.8" rx="{rx}"/>')
 def t(self,x,y,label,s=22,c=INK,b=False,a='start'):
  self.e.append(f'<text x="{x}" y="{y}" text-anchor="{a}" font-family="Noto Sans CJK SC, sans-serif" font-weight="{700 if b else 400}" font-size="{s}" fill="{c}">{escape(label)}</text>')
 def p(self,pts,arrow=True):
  self.e.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in pts)+'" fill="none" stroke="'+MID+'" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" '+('marker-end="url(#a)"' if arrow else '')+'/>')
 def frame(self,y,h,number,label):
  self.r(29,y,662,h,CARD,LINE,18)
  self.r(46,y+18,44,33,INK,rx=9)
  self.t(68,y+41,number,16,CARD,True,'middle')
  self.t(105,y+42,label,22,INK,True)
 def mod(self,x,y,w,h,head,desc,c=BLUE):
  self.r(x,y,w,h,V[c],c,11)
  self.t(x+w/2,y+30,head,22,c,True,'middle')
  if desc:self.t(x+w/2,y+58,desc,18,INK,False,'middle')
 def data(self,x,y,w,h,head,shape,c=TEAL):
  self.r(x,y,w,h,'#ffffff',c,9)
  for i in range(3):
   for j in range(3):
    self.r(x+11+j*6,y+12+i*6,4,4,c if i==j else V[c])
  self.t(x+w/2+8,y+27,head,21,c,True,'middle')
  self.t(x+w/2,y+54,shape,20,INK,False,'middle')
 def sum(self,x,y):
  self.e.append(f'<circle cx="{x}" cy="{y}" r="17" fill="white" stroke="{OR}" stroke-width="2"/>')
  self.t(x,y+8,'+',25,OR,True,'middle')
 def note(self,y,text,c=INK):
  self.r(47,y-22,626,48,V[BLUE])
  self.t(63,y+8,text,19,c)
 def legend(self,line):
  self.t(33,self.h-29,line,18,MUT)
 def save(self,n):
  self.e.append('</svg>'); ALL[n]='\n'.join(self.e)

f=Fig('01','Stem 到 E：视觉主干如何接线','五级卷积输出三条旁路，最深一级走共享主干。',1045)
f.frame(151,392,'01','五级卷积 Stem')
f.data(75,209,176,65,'RGB 画布','512² × 3')
f.mod(322,202,322,72,'首级 Conv 3×3','stride 2，直接混通道')
f.p([(252,242),(314,242)])
f.p([(483,277),(483,294),(159,294),(159,307)])
for x,tag,sz in [(53,'F1','256² × 32'),(265,'F2 / S4','128² × 64'),(477,'F3 / S8','64² × 128')]:
 f.data(x,313,172,66,tag,sz)
f.p([(227,346),(257,346)]); f.p([(439,346),(469,346)])
f.p([(563,381),(563,414)])
f.data(477,421,172,66,'F4 / S16','32² × 256')
f.p([(473,454),(443,454)])
f.data(265,421,172,66,'F5','16² × 512')
f.t(57,511,'F2–F5：DW 3×3 s2 → PW',19,MUT)
f.p([(351,490),(351,548)])
f.frame(553,304,'02','共享编码网络')
f.mod(49,620,207,76,'Body ×16','16² × 512 → 同形')
f.mod(282,620,151,76,'展平 + 4','256 → 260')
f.mod(459,620,211,76,'Attention ×12','260 × 512 → 同形',PUR)
f.p([(351,553),(351,604),(152,604),(152,617)])
f.p([(260,658),(275,658)]);f.p([(438,658),(451,658)])
f.p([(563,700),(563,722),(417,722),(417,735)])
f.mod(205,738,421,74,'按层融合 · 只保留图像格','末层 + 逐通道门控的早期网格',PUR)
f.p([(417,815),(417,886)])
f.data(257,891,320,71,'E ＝ 粗图输出','16² × 512 → 256 × 512',PUR)
f.t(58,977,'E 送视觉记忆与 FPN；不是额外的网络。',20,MUT)
f.legend('白框 = 张量   ·   色底 = 操作   ·   ² 表示高 × 宽的二维网格')
f.save('16_stem_backbone')

f=Fig('02','D 来自两次逐元素融合','E 向上采样，S16 和 S8 分别补回较细的空间信息。',1115)
f.frame(151,382,'01','16² → 32²：E 与 S16 在 128 通道处相加')
f.data(50,213,250,66,'E · 主干末端','16² × 512',PUR)
f.data(421,213,250,66,'S16 · Stem F4','32² × 256',TEAL)
f.mod(50,300,250,79,'W_E：512 → 128','再上采样 2×：32² × 128',PUR)
f.mod(421,300,250,79,'W16：256 → 128','空间尺寸不变：32² × 128',TEAL)
f.p([(175,281),(175,293)]);f.p([(545,281),(545,293)])
f.p([(174,383),(174,416),(339,416)])
f.p([(548,383),(548,416),(381,416)])
f.sum(360,416)
f.p([(360,435),(360,446)])
f.mod(213,450,294,66,'局部残差块 B ×1','T32：32² × 128',BLUE)
f.frame(554,475,'02','32² → 64²：T32 与 S8 再融合一次')
f.data(50,615,250,66,'T32 · 第一级输出','32² × 128',BLUE)
f.data(421,615,250,66,'S8 · Stem F3','64² × 128',TEAL)
f.p([(360,520),(360,542),(42,542),(42,646),(49,646)])
f.mod(50,706,250,79,'上采样 2×','64² × 128',BLUE)
f.mod(421,706,250,79,'W8：128 → 128','等宽仍需学习投影',TEAL)
f.p([(175,683),(175,699)]);f.p([(545,683),(545,699)])
f.p([(174,790),(174,812),(339,812)])
f.p([(548,790),(548,812),(381,812)])
f.sum(360,812)
f.p([(360,831),(360,847)])
f.mod(139,852,442,51,'B ×3 → RMS 归一化 → SiLU','',PUR)
f.p([(360,907),(360,932)])
f.data(228,937,265,65,'D · 细空间输出','64² × 128',PUR)
f.note(1053,'加法前两路 H、W、C 都相同；上采样只改 H、W。')
f.legend('网格边长与通道分开标注。两个相加都不是 concat。')
f.save('17_d_fusion')

f=Fig('03','分割：从共享原型到每框掩码','U 是给原型头的融合输入，不是共享原型 P。',1004)
f.frame(151,362,'01','先在 128² 尺度融合原型头输入')
f.data(55,211,240,67,'D · 细空间输出','64² × 128',PUR)
f.data(425,211,240,67,'S4 · Stem F2','128² × 64',TEAL)
f.mod(55,300,240,73,'D 上采样 2×','128² × 128',PUR)
f.mod(425,300,240,73,'W4：64 → 128','128² × 128',TEAL)
f.p([(175,280),(175,294)]);f.p([(545,280),(545,294)])
f.p([(175,377),(175,404),(340,404)]);f.p([(545,377),(545,404),(382,404)])
f.sum(360,404)
f.p([(360,423),(360,437)])
f.data(228,440,264,64,'U ＝ 融合张量','128² × 128',OR)
f.frame(534,372,'02','两头分工：共享图案 × 每框系数')
f.data(54,593,265,66,'U · 原型头输入','128² × 128',OR)
f.data(401,593,265,66,'D · 系数头输入','64² × 128',PUR)
f.p([(187,662),(187,678)]);f.p([(534,662),(534,678)])
f.mod(54,685,265,72,'原型头 H_P','P：128² × 32',BLUE)
f.mod(401,685,265,72,'系数头 H_C','C：64² × 32',PUR)
f.p([(187,762),(187,792),(229,792),(229,814)])
f.p([(534,762),(534,792),(491,792),(491,814)])
f.mod(104,819,512,72,'按框中心抽 C 的 32 维 → 与 P 点积','逐像素 sigmoid → 该实例的掩码',TEAL)
f.note(933,'P 在所有实例间共享；不同框取不同的 C 向量。')
f.legend('掩码裁剪/输出格式属于下游后处理，不是原型头。')
f.save('18_mask_fusion')

f=Fig('04','同一个 512 维，怎样进入不同模块','尺寸在模块内部暂时改变，残差接回时回到 512 维。',1030)
f.frame(151,325,'01','Body：位置不展成注意力序列')
f.data(50,216,192,66,'输入 X','16² × 512')
f.mod(266,210,184,77,'DW 门控','空间尺寸不变')
f.mod(477,210,192,77,'HConvGLU','每位置 512→512',PUR)
f.p([(245,247),(259,247)]);f.p([(454,247),(470,247)])
f.p([(572,291),(572,325),(440,325),(440,341)])
f.mod(248,346,359,77,'GRN + 残差写回','输出仍是 16² × 512',BLUE)
f.t(57,452,'计数：256 个位置 × 512 个通道。',19,MUT)
f.frame(497,438,'02','Attention：4 个 Q 头共享 1 组 K / V')
f.data(50,561,620,67,'输入 Z（256 图像格 + 4 register）','260 × 512',TEAL)
f.p([(360,632),(360,649)],False)
f.p([(360,649),(131,649),(131,664)])
f.p([(360,649),(584,649),(584,664)])
f.p([(360,649),(360,664)])
for x,title,v,c in [(51,'Q 投影','4 × 32 = 128',BLUE),(280,'K 投影','1 × 32 = 32',PUR),(507,'V 投影','1 × 64 = 64',TEAL)]:
 f.mod(x,670,163,68,title,v,c)
for x in (132,362,588): f.p([(x,742),(x,758)],False)
f.p([(132,758),(588,758)],False); f.p([(362,758),(362,771)])
f.mod(143,778,439,72,'4 头 × 64 → 拼接为 256','门控后 W_O：256 → 512',PUR)
f.p([(361,854),(361,868)])
f.data(193,873,336,60,'残差输出 Z⁺','260 × 512',TEAL)
f.note(971,'256 有两种含义：图像格个数 ≠ 四头拼接通道数。')
f.legend('只画投影尺寸；省略 norm、RoPE、mask 和残差门控细节。')
f.save('15_block_dimensions')

if __name__=='__main__':
 p=argparse.ArgumentParser(description='Render editable SVGs; PNG export optionally needs Inkscape')
 p.add_argument('--out',type=Path,default=ROOT)
 p.add_argument('--png',action='store_true')
 a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=True)
 for name,svg in ALL.items():
  path=a.out/(name+'.svg')
  path.write_text(svg,encoding='utf8')
  if a.png:
   subprocess.run(['inkscape',str(path),'--export-width=1440',
                   '--export-filename='+str(a.out/(name+'.png'))],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
