
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

font = { 'size'   : 15}
plt.rc('font', **font)

colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
styles = ['o','s','v','^','D',">","<","*","h","H","+","1","2","3","4","8","p","d","|","_",".",","]

def roofline(tag, flag, FLOPS, AIHBM, AIL2=None, AIL1=None, LABELS=None, NormDim=None):

    if not FLOPS:
        print('FLOPS can not be empty!')
        return
    if max(FLOPS)==0:
        print('FLOPS are all 0s!')
        return
    if (not AIHBM) and (not AIL2) and (not AIL1):
        print('AIHBM, AIL2 and AIL1 can not all be empty!')
        return
    if (len(FLOPS) != len(AIHBM)) or (len(FLOPS) != len(AIL2)) or (len(FLOPS) != len(AIL1)):
        print('FLOPS needs to have the same length as AI!')
        return
    if (flag != 'HBM') and (flag != 'L2') and (flag != 'L1') and (flag != 'all'):
        print('flag needs to be one of HBM, L2, L1, and all!')
        return
    if NormDim:
        tmp= np.array(NormDim)
        tmpp=tmp[np.nonzero(FLOPS)]
        markersize = np.array(NormDim)/max(tmpp)*15
    else:
        markersize  = np.ones(len(NormDim))*5
    base_markersize = 5
    markerwidth = 2

    memRoofs = [('L1', 54000.), ('L2', 2996.77), ('HBM', 828.76)] #('L1', 14336.)
#     cmpRoofs = [('Tensor', 125000), ('FMA', 7068.86), ('No-FMA', 3535.79)]
    cmpRoofs = [('Tensor', 96.9), ('SP', 14.0)]

    fig = plt.figure(1,figsize=(10.67,6.6))
    plt.clf()
    ax = fig.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Arithmetic Intensity [FLOPs/Byte]')
    ax.set_ylabel('Performance [GFLOP/sec]')

    nx   = 10000
    xmin = -3 #-0.65
    xmax = 3
    ymin = 0.001
    ymax = 200000

    tmp= np.array(FLOPS)
    tmpp=tmp[np.nonzero(FLOPS)]
    ymin = 1 if tmpp.min()<1 else tmpp.min()
#     ymax = tmpp.max()
#     xmax = np.log10(max(AIHBM))
    
    ax.set_xlim(10**xmin, 10**xmax)
    ax.set_ylim(ymin, ymax)

    ixx = int(nx*0.02)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    scomp_x_elbow  = []
    scomp_ix_elbow = []
    smem_x_elbow   = []
    smem_ix_elbow  = []

    x = np.logspace(xmin,xmax,nx)
    for roof in cmpRoofs:
        for ix in range(1,nx):
            if float(memRoofs[0][1] * x[ix]) >= roof[1]*1024 and (memRoofs[0][1] * x[ix-1]) < roof[1]*1024:
                scomp_x_elbow.append(x[ix-1])
                scomp_ix_elbow.append(ix-1)
                break


    for roof in memRoofs:
        for ix in range(1,nx):
            if (cmpRoofs[0][1]*1024 <= roof[1] * x[ix] and cmpRoofs[0][1]*1024 > roof[1] * x[ix-1]):
                smem_x_elbow.append(x[ix-1])
                smem_ix_elbow.append(ix-1)
                break

    for i in range(len(cmpRoofs)):
        roof = cmpRoofs[i][1]*1024
        y = np.ones(len(x)) * roof
        ax.plot(x[scomp_ix_elbow[i]:],y[scomp_ix_elbow[i]:],c='k',ls='-',lw='2')

    for i in range(len(memRoofs)):
        roof = memRoofs[i][1]
        y = x * roof
        ax.plot(x[:smem_ix_elbow[i]+1],y[:smem_ix_elbow[i]+1],c='k',ls='-',lw='2')

    marker_handles = []  

    for i in range(len(AIHBM)):
        if flag == 'L1':
            ax.plot(float(AIL1[i]),float(FLOPS[i]),c=colors[i%10],marker=styles[0],\
                    linestyle='None',ms=markersize[i],markerfacecolor='none',\
                    markeredgewidth=markerwidth)#,label=LABELS[i])
        elif flag == 'L2':
            ax.plot(float(AIL2[i]),float(FLOPS[i]),c=colors[i%10],marker=styles[1],\
                    linestyle='None',ms=markersize[i],markerfacecolor='none',\
                    markeredgewidth=markerwidth)#,label=LABELS[i])
        elif flag == 'HBM':
            ax.plot(float(AIHBM[i]),float(FLOPS[i]),c=colors[i%10],marker=styles[2],\
                    linestyle='None',ms=markersize[i],markerfacecolor='none',\
                    markeredgewidth=markerwidth)#,label=LABELS[i])
        elif flag == 'all':
            ax.plot(float(AIL1[i]),float(FLOPS[i]),c=colors[0],marker=styles[0],linestyle='None',\
                    ms=markersize[i],markerfacecolor='none',markeredgewidth=markerwidth)#,label=LABELS[i])
            ax.plot(float(AIL2[i]),float(FLOPS[i]),c=colors[1],marker=styles[1],linestyle='None',\
                    ms=markersize[i],markerfacecolor='none',markeredgewidth=markerwidth)#,label=LABELS[i])
            ax.plot(float(AIHBM[i]),float(FLOPS[i]),c=colors[2],marker=styles[2],linestyle='None',
                    ms=markersize[i],markerfacecolor='none',markeredgewidth=markerwidth)#,label=LABELS[i])


            
    for i in range(len(memRoofs)):
#         marker_handles.append(ax.plot([],[],c='k',marker=styles[i],linestyle='None',ms=np.mean(markersize),\
        marker_handles.append(ax.plot([],[],c='k',marker=styles[i],linestyle='None',ms=base_markersize,\
                                      markerfacecolor='none',markeredgewidth=markerwidth,label=memRoofs[i][0])[0])

    for roof in cmpRoofs:
        ax.text(x[-ixx],roof[1]*1024,
              roof[0] + ': ' + '{0:.1f}'.format(roof[1]) + ' TFLOP/s',
              horizontalalignment='right',
              verticalalignment='bottom')

    for roof in memRoofs:
        ang = np.arctan(np.log10(xlim[1]/xlim[0]) / np.log10(ylim[1]/ylim[0])
                                   * fig.get_size_inches()[1]/fig.get_size_inches()[0] )
        if x[ixx]*roof[1] >ymin:
            ax.text(x[ixx],x[ixx]*roof[1]*(1+0.25*np.sin(ang)**2),
              roof[0] + ': ' + '{0:.1f}'.format(float(roof[1])) + ' GB/s',
              horizontalalignment='left',
              verticalalignment='bottom',
              rotation=180/np.pi*ang)
        else:
            ymin_ix_elbow=list()
            ymin_x_elbow=list()
            for ix in range(1,nx):
                if (ymin <= roof[1] * x[ix] and ymin > roof[1] * x[ix-1]):
                    ymin_x_elbow.append(x[ix-1])
                    ymin_ix_elbow.append(ix-1)
                    break
            ax.text(x[ixx+ymin_ix_elbow[0]],x[ixx+ymin_ix_elbow[0]]*roof[1]*(1+0.25*np.sin(ang)**2),
              roof[0] + ': ' + '{0:.1f}'.format(float(roof[1])) + ' GB/s',
              horizontalalignment='left',
              verticalalignment='bottom',
              rotation=180/np.pi*ang)



    leg1 = plt.legend(handles = marker_handles,loc='lower right', ncol=1,bbox_to_anchor = (1,0))
    ax.add_artist(leg1)

    #   patch_handles = list()
    #   for i in range(0,len(AIHBM)):
    #       patch_handles.append(mpatches.Patch(color=colors[i%10]))#,label = LABELS[i]))

    #   leg2 = plt.legend(handles = patch_handles,loc=4,ncol=2,bbox_to_anchor = (1,0),scatterpoints = 1)

    ax.text(xlim[0]*1.1,ylim[1]/1.1, '-'.join([tag,flag]), horizontalalignment='left',verticalalignment='top')

#     plt.savefig(tag+'.png')
#     plt.savefig(tag+'.eps')

    plt.show()
