import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



age_bins = ['0-4', '5-18', '19-29', '30-64', '65+']


def plot_samples(runs, pars):
    for i in range(len(runs)):
        run=runs[i]
        par=pars[i]
        fig = plt.figure(figsize=(8,8))
        for j in range(6):
            if j<5:
                plt.subplot(4,2,j+1)
                plt.imshow(run[j,:,:])
                plt.annotate(age_bins[j], (2,25), color='white')
                plt.xlabel('Day')
                plt.ylabel('Tract')
            else:
                plt.subplot(4,2,j+1)
                plt.imshow(np.zeros(run[:,0,:].shape), cmap='Greys')
                R0 = 2.5*par[0] + 2.5
                TriggerDay = int(183 + 182*par[1])
                plt.title('R0=%f, TriggerDay=%d'%(R0, TriggerDay))
                #plt.annotate('R0=%f, TriggerDay=%d'%(R0, TriggerDay),
                #             (62, 60),
                #             color='black')
                plt.axis('off')
        plt.tight_layout(pad=0.25)
    return fig


def compare_curves(reals, fakes, pars, norm='log', peakshift=True):
    if norm=='log':
        reals = np.exp(reals) - 1.
        fakes = np.exp(fakes) - 1.
    # Map parameters back to un-normalized units
    pars[:,0] = 2.5*pars[:,0] + 2.5
    pars[:,1] = 183 + np.multiply(pars[:,1], 182).astype(np.int)
    
    pars = pars
    realsum = np.sum(reals, axis=(1,2))
    fakesum = np.sum(fakes, axis=(1,2))
    days = np.arange(realsum.shape[1])
    idx = np.random.randint(realsum.shape[0])
    true = realsum[idx].astype(np.int)
    gen = fakesum[idx].astype(np.int)
    par = pars[idx]
    R0 = par[0]
    TriggerDay = par[1]

    if peakshift:
        true_shift = days - np.argmax(true)
        gen_shift = days - np.argmax(gen)
        offset = np.argmax(true) - np.argmax(gen)
        if offset > 0: 
            shifted = np.pad(gen,(offset,0), mode='constant')[:-offset]
        else:
            shifted = np.pad(gen,(0,-offset),  mode='constant')[-offset:]
        
        f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1], 'hspace':0}, figsize=(8,6))
        a0.plot(days, true, 'k-', label='True')
        a0.plot(days, shifted, 'r-', label='cGAN (shifted)')
        a0.set_title('R0=%f, TriggerDay=%d, peakshift=%d'%(R0, TriggerDay,offset))
        a0.set_ylabel('Daily new symptomatic individuals')
        a0.set_yscale('log')
        a0.legend()
        a0.tick_params(axis='x', direction='in', labelbottom=False, which='both')


        a1.plot([days[0], days[-1]], [0., 0.], 'k-')
        normed = (shifted.astype(np.float) - true.astype(np.float))/true.astype(np.float)
        a1.plot(days, normed, 'r-')
        a1.set_xlabel('Days ')
        a1.set_ylabel('Relative error')
        a1.set_ylim(-1,1)
        a1.tick_params(axis='x', top=True, direction='inout', labeltop=False, which='both')
    else:
        f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1], 'hspace':0}, figsize=(8,6))
        a0.plot(days, true, 'k-', label='True')
        a0.plot(days, gen, 'r-', label='cGAN')
        a0.set_title('R0=%f, TriggerDay=%d'%(R0, TriggerDay))
        a0.set_ylabel('Daily new symptomatic individuals')
        a0.set_yscale('log')
        a0.legend()
        a0.tick_params(axis='x', direction='in', labelbottom=False, which='both')


        a1.plot([days[0], days[-1]], [0., 0.], 'k-')
        normed = (gen.astype(np.float) - true.astype(np.float))/true.astype(np.float)
        a1.plot(days, normed, 'r-')
        a1.set_xlabel('Day')
        a1.set_ylabel('Relative error')
        a1.set_ylim(-1,1)
        a1.tick_params(axis='x', top=True, direction='inout', labeltop=False, which='both')

    # Compute peak-shifted chi square
    scores = np.zeros((realsum.shape[0],))
    for i in range(realsum.shape[0]):
        true = realsum[i].astype(np.int)
        gen = fakesum[i].astype(np.int)
        mask = np.where(true>=10, 1, 0) # only compare to where truth >=10
        true=true*mask
        gen=np.where(gen>=10, gen, 0)
        offset = np.argmax(true) - np.argmax(gen)
        if offset > 0:
            shifted = np.pad(gen,(offset,0), mode='constant')[:-offset]
        else:
            shifted = np.pad(gen,(0,-offset),  mode='constant')[-offset:]
        score = np.abs(true - mask*shifted)
        scores[i] = score.mean()
    f2 = plt.figure()
    plt.hist(scores, bins=100)
    plt.yscale('log')
    plt.show()
    return f, scores.mean(), f2
