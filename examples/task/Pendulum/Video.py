import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import matplotlib.animation as animation
from IPython import display

def get_video(vedio_revive,vedio_old):
    fps = 24
    nSeconds = 6
    shots = [ np.random.rand(5,5) for _ in range( nSeconds * fps ) ]
    fig = plt.figure( figsize=(10,5) )
    # gs0 = gridspec.GridSpec(1, 2, figure=fig, wspace=0, hspace=0)
    ax = fig.add_subplot()

    a = shots[0]
    im = ax.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1)
    ax.plot([0.5, 0.5], [0.08, 1-0.08], 'k-', transform=ax.transAxes)
    ax.text(0.25,0.9,'REVIVE policy', ha='center',va='center',color = 'k',transform=ax.transAxes)
    ax.text(0.75,0.9,'Old policy', ha='center',va='center',color = 'k',transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    
    time_text = ax.text(0.5,0.03,'', ha='center',va='center',color = 'k',transform=ax.transAxes)
    rew_text_1= ax.text(0.25,0.07,'', ha='center',va='center',color = 'k',transform=ax.transAxes)
    rew_text_2 = ax.text(0.75,0.07,'', ha='center',va='center',color = 'k',transform=ax.transAxes)

    cumu_rew_text_1= ax.text(0.25,0.03,'', ha='center',va='center',color = 'k',transform=ax.transAxes)
    cumu_rew_text_2 = ax.text(0.75,0.03,'', ha='center',va='center',color = 'k',transform=ax.transAxes)


    rew_array_1 = vedio_revive[1]
    rew_array_2 = vedio_old[1]

    cumu_rew_1 = np.convolve(vedio_revive[1],np.array([1]*len(vedio_revive[1])),
                            mode='full')[:len(vedio_revive[1])]
    cumu_rew_2 = np.convolve(vedio_old[1],np.array([1]*len(vedio_old[1])),
                            mode='full')[:len(vedio_old[1])]

    def animate_func(i):
        im.set_array(np.concatenate((vedio_revive[0][i], vedio_old[0][i]),axis=1))
        time_text.set_text('time: %.2f s'%(i / fps))
        rew_text_1.set_text('reward: %.2f'%rew_array_1[i])
        rew_text_2.set_text('reward: %.2f'%rew_array_2[i])

        cumu_rew_text_1.set_text('return: %.2f'%cumu_rew_1[i])
        cumu_rew_text_2.set_text('return: %.2f'%cumu_rew_2[i])
        return [im]

    anim = animation.FuncAnimation(
                                    fig, 
                                    animate_func, 
                                    frames = nSeconds * fps,
                                    interval = 1000 / fps, # in ms
                                    )
    # anim.save('test_anim.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])
    anim.save('url/result.gif', writer='imagemagick', fps=fps)
    video = anim.to_jshtml()

    html = display.HTML(video)
    return html