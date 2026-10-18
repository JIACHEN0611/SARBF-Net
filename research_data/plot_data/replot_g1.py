#!/usr/bin/env python3
"""Replot SARBF-Net Fig. G1 from the original exported numerical arrays.

No training, checkpoint loading, data rescaling, or window reselection is done.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

ARMS=('sa_lcm','no_saer_bcif_kept','saer_only')
LABEL={'sa_lcm':'SARBF-Net','no_saer_bcif_kept':'w/o SAER','saer_only':'w/o BCIF'}
COLOR={'sa_lcm':'#2166ac','no_saer_bcif_kept':'#d95f02','saer_only':'#25854a'}
EXPECTED=[('PEMS04-96','calm',0,15033,96),('PEMS04-96','volatile',0,16665,96),
          ('Weather-192','calm',1,47149,192),('Weather-192','volatile',1,47533,192)]

def load_panels(path):
    raw=path.read_bytes()
    data=json.loads(raw)
    panels=data['panels']
    if len(panels)!=4:
        raise ValueError('Expected exactly the four original panels.')
    found={(p['task'],p['kind']):p for p in panels}
    ordered=[]
    for task,kind,channel,origin,horizon in EXPECTED:
        p=found[(task,kind)]
        if (p['channel'],p['global_first_target'],p['horizon'])!=(channel,origin,horizon):
            raise ValueError(f'Window identity mismatch: {task}, {kind}')
        for name,seq,length in [('past',p['past'],96),('truth',p['truth'],horizon)]+[
            (arm,p['predictions'][arm],horizon) for arm in ARMS]:
            array=np.asarray(seq,dtype=float)
            if array.shape!=(length,) or not np.isfinite(array).all():
                raise ValueError(f'Invalid {task}/{kind}/{name} array.')
        ordered.append(p)
    return ordered,hashlib.sha256(raw).hexdigest()

def choose_font(requested):
    if requested:
        path=font_manager.findfont(requested,fallback_to_default=False)
    else:
        try:
            path=font_manager.findfont('Times New Roman',fallback_to_default=False)
        except ValueError:
            path=font_manager.findfont('DejaVu Serif',fallback_to_default=False)
    return font_manager.FontProperties(fname=path).get_name()

def draw(panels,font):
    # Geometry, colors, line styles, and insets reproduce the original section55.py plot.
    plt.rcParams.update({'font.family':font,'font.size':9,'axes.labelsize':9,
        'axes.titlesize':9,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,
        'pdf.use14corefonts':False,'svg.fonttype':'path','axes.linewidth':.7,
        'path.simplify':False,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(2,2,figsize=(7.2,4.7))
    for ax,p in zip(axes.ravel(),panels):
        h=p['horizon'];x=np.arange(h)
        ax.plot(np.arange(-96,h),p['past']+p['truth'],color='#222222',lw=1.2,label='Ground truth')
        for arm in ARMS:
            ax.plot(x,p['predictions'][arm],color=COLOR[arm],lw=1.3 if arm=='sa_lcm' else 1,
                ls='-' if arm=='sa_lcm' else '--',label=LABEL[arm])
        ax.axvline(-.5,color='#777777',ls=':',lw=.9)
        ax.set_title(f"{p['task']} · {p['kind']} input · {p['variable']}")
        ax.set_xlabel('Steps relative to forecast start');ax.set_ylabel('Standardized value')
        ax.grid(alpha=.15)
        for side in ['top','right']:ax.spines[side].set_visible(False)
        lo,hi=h//4,h//2
        inset=ax.inset_axes([.63,.63,.34,.30])
        inset.plot(x[lo:hi],p['truth'][lo:hi],color='#222222',lw=.9)
        for arm in ARMS:
            inset.plot(x[lo:hi],p['predictions'][arm][lo:hi],color=COLOR[arm],lw=.8,
                       ls='-' if arm=='sa_lcm' else '--')
        inset.tick_params(labelsize=6,length=2);inset.set_xlim(lo,hi-1)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,-.01))
    fig.tight_layout(rect=(0,.045,1,1))
    return fig

def main():
    base=Path(__file__).resolve().parent
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=base/'curve_data.json')
    ap.add_argument('--out',type=Path,default=base/'figures')
    ap.add_argument('--font',help='Optional installed font family, e.g. Times New Roman.')
    args=ap.parse_args()
    panels,digest=load_panels(args.input)
    font=choose_font(args.font)
    args.out.mkdir(parents=True,exist_ok=True)
    fig=draw(panels,font)
    output=args.out/'Fig_G1_prediction_curves'
    fig.savefig(output.with_suffix('.pdf'),format='pdf',bbox_inches='tight',
                metadata={'Title':'SARBF-Net forecasting cases (Fig. G1)'})
    fig.savefig(output.with_suffix('.svg'),format='svg',bbox_inches='tight')
    fig.savefig(args.out/'Fig_G1_prediction_curves_600dpi.png',dpi=600,bbox_inches='tight')
    # Render line artwork directly from its vector paths; do not upscale a Word image.
    buffer=io.BytesIO()
    fig.savefig(buffer,format='png',dpi=1000,bbox_inches='tight')
    buffer.seek(0)
    with Image.open(buffer) as im:
        im.convert('RGB').save(args.out/'Fig_G1_prediction_curves_1000dpi.tiff',
                              compression='tiff_lzw',dpi=(1000,1000))
    plt.close(fig)
    rows=[]
    for p in panels:
        for arm in ARMS:
            err=np.asarray(p['predictions'][arm])-np.asarray(p['truth'])
            rows.append({'task':p['task'],'window':p['kind'],'channel':p['channel'],
                'first_target':p['global_first_target'],'model':LABEL[arm],
                'window_MSE':float(np.mean(err**2)),'window_MAE':float(np.mean(abs(err)))})
    with (args.out/'window_metrics_check.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report={'input_sha256':digest,'font_family':font,'panels':4,
        'layout_inches':[7.2,4.7],'pdf_fonttype':42,'svg_text_as_paths':True,
        'raster_dpi':{'PNG':600,'TIFF':1000},'training_or_inference':False,
        'data_rescaling':False,'window_reselection':False,
        'note':'Metrics are univariate single-window errors, not full-test-set results.'}
    (args.out/'export_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Input SHA-256:',digest)
    print('Embedded PDF font family:',font)
    print('Output directory:',args.out.resolve())
    for p in sorted(args.out.glob('Fig_G1*')):print(p.name,p.stat().st_size,'bytes')

if __name__=='__main__':main()
