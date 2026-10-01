import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import DATA,FAMILIES
plt.rcParams.update({'font.size':9,'savefig.dpi':180})
raw=json.loads((DATA/'holdout.json').read_text())['rows'];names=['F1_FULL_4679','F1_FULL_3459_Q0','F1_3459_UNGROUNDED_Q0'];base={r['name']:r for r in json.loads((DATA/'F1.json').read_text())['rows']};pc={r['name']:r for r in json.loads((DATA/'PORT_CONDITIONED_QUOTIENT.json').read_text())['rows']}
fig,axes=plt.subplots(1,3,figsize=(12.5,4.0),layout='constrained')
for family,color in zip(FAMILIES,plt.cm.tab10.colors):
 rows=[r for r in raw if r['family']==family];y=[np.log1p(r['forms']['shortest']['points']['128']['gap_lo']) for r in rows]
 for ax,k in zip(axes,['m','hub_edge_share']):
  ax.scatter([r['forms']['shortest']['statistics'][k] for r in rows],y,s=13,alpha=.7,color=color,label=family)
axes[0].set(xlabel='Edges in terminal component',ylabel='log(1 + certified relative gap)',title='200 fresh graphs: Spearman rho = 0.539')
axes[1].set(xlabel='Maximum incident-edge share',ylabel='log(1 + certified relative gap)',title='Hub share: Spearman rho = -0.361')
x=np.arange(3);w=.24
for offset,key,label,color in [(-1,'shortest','SP','#4575b4'),(0,'edge_disjoint','ED','#e1ad36'),(1,'conditioned','Port-conditioned ED','#168c65')]:
 y=[pc[n]['forms']['quotient_cycle_edge_disjoint']['points']['128']['gap_lo'] if key=='conditioned' else base[n]['forms'][key]['points']['128']['gap_lo'] for n in names]
 axes[2].bar(x+offset*w,y,w,label=label,color=color)
axes[2].set(yscale='log',ylabel='Certified relative gap (U-L)/L',title='F1: identical 128 local attempts')
axes[2].set_xticks(x,['Full bank','Full3459 Q0','Ungrounded Q0']);axes[2].legend(frameon=False,loc='upper right',fontsize=8)
fig.suptitle('T9: fixed refinement budget; quotient startup work is extra and separately charged',fontsize=11)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.33,-.13),ncol=5,fontsize=7,frameon=False)
fig.savefig(DATA/'SUMMARY.png',bbox_inches='tight');fig.savefig(DATA/'SUMMARY.pdf',bbox_inches='tight');plt.close(fig)
print('SUMMARY.png and SUMMARY.pdf written.')
