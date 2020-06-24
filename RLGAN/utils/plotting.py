import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys



age_bins = ['0-4', '5-18', '19-29', '30-64', '65+']


def get_label(tag, par):
    if tag=='3parA':
        return 'R0=%f, WFHcompl=%f, WFHdays=%d'%(par[0], par[1], par[2])
    else:
        return 'R0=%f, TriggerDay=%d'%(par[0], par[1])

def plot_samples(runs, pars, tag):
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
                txt = get_label(tag, par)
                plt.title(txt)
                plt.axis('off')
        plt.tight_layout(pad=0.25)
    return fig


def compare_curves(reals, fakes, pars, tag, norm='log', peakshift=True):
    if norm=='log':
        reals = np.exp(reals) - 1.
        fakes = np.exp(fakes) - 1.
    
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
        txt = get_label(tag, par)
        a0.set_title(txt+', pkshift=%d'%offset)
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
        txt = get_label(tag, par)
        a0.set_title(txt)
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


def plot_snapspread(inp, tar, gen, pars, states, bios, params):
    if params.norm=='log':
        inp = np.exp(inp) - 1.
        tar = np.exp(tar) - 1.
        gen = np.exp(gen) - 1.
    inp = np.sum(inp, axis=(2,3))
    tar = np.sum(tar, axis=(2,3))
    gen = np.sum(gen, axis=(2,3))

    needsplot = True
    f = plt.figure()
    errs = []
    for sim in range(inp.shape[0]):
        samp_t1 = inp[sim]
        samp_t2 = tar[sim]
        samp_pars = pars[sim]
        if samp_t1[0]==0 or samp_t2[0]==0:
            continue

        idxs = np.where(np.any(bios - samp_pars, axis=1), False, True)
        matches = states[idxs]
        if matches.shape[0]==0:
            continue

        close05 = []
        L1 = []
        
        for i in range(matches.shape[0]):
            sim2 = matches[i]
            err = np.abs((samp_t1 - sim2)/samp_t1)
            day2 = np.argmin(np.mean(err[:357,:], axis=1))
            if np.abs(samp_t1[0] - sim2[day2,0]) < 1e-1:
                continue
            mean = err[day2].mean()
            if mean<0.05:
                close05.append(sim2[day2])
                new = sim2[day2+7]
                if needsplot:
                    plt.plot([0,1], np.log1p([sim2[day2,0], new[0]]), color='g', alpha=0.1)
                    plt.plot([0,1], np.log1p([sim2[day2,1], new[1]]), color='b', alpha=0.1)
                diff = np.abs((samp_t2 - new)/samp_t2)
                L1.append(diff)
            else:
                continue
                
        if needsplot and L1:
            plt.plot([0,1], np.log1p([samp_t1[0], samp_t2[0]]), 'k-', label='new sympt')
            plt.plot([0,1], np.log1p([samp_t1[1], samp_t2[1]]), 'k--', label='cum sympt')
            plt.plot([0,1], np.log1p([samp_t1[0], gen[sim,0]]), 'r-', label='new sympt (gen)')
            plt.plot([0,1], np.log1p([samp_t1[1], gen[sim,1]]), 'r--', label='cum sympt (gen)')
            plt.ylabel('$\log (1+n)$')
            plt.xlabel('Prediction interval')
            plt.ylim((0,12.5))
            plt.title('LLcompliance=%f, WFHcompliance=%f'%(pars[sim,0], pars[sim,1]))
            plt.legend()
            needsplot=False
        if not L1:
            continue
        L1 = np.stack(L1, axis=0)
        sympt_err = L1[:,0].mean()
        cum_err = L1[:,1].mean()
        diff = np.abs((samp_t2 - gen[sim])/samp_t2)
        errs.append(diff/L1.mean(axis=0))
        
    errs = np.stack(errs, axis=0)
    return f, errs


def plot_spread_ts(reals, fakes, pars, states, bios, params):
    reals = np.sum(reals, axis=(3,4))
    fakes = np.sum(fakes, axis=(3,4))
    inp = reals[:,:,0]
    tar = reals[:,:,-1]
    gen = fakes[:,:,-1]

    days = np.arange(params.pred_interval)
    needsplot = True
    f = plt.figure()
    errs = []
    for sim in range(inp.shape[0]):
        samp_t1 = inp[sim]
        samp_t2 = tar[sim]
        samp_pars = pars[sim]
        if samp_t1[0]==0 or samp_t2[0]==0:
            continue

        idxs = np.where(np.any(bios - samp_pars, axis=1), False, True)
        matches = states[idxs]
        if matches.shape[0]==0:
            continue

        close05 = []
        L1 = []

        for i in range(matches.shape[0]):
            sim2 = matches[i]
            err = np.abs((samp_t1 - sim2)/samp_t1)
            day2 = np.argmin(np.mean(err[:357,:], axis=1))
            if np.abs(samp_t1[0] - sim2[day2,0]) < 1e-1:
                continue
            mean = err[day2].mean()
            if mean<0.05:
                close05.append(sim2[day2])
                new = sim2[day2+params.pred_interval]
                if needsplot:
                    #plt.plot(days, np.log1p([sim2[day2,0], new[0]]), color='g', alpha=0.1)
                    plt.plot(days, np.log1p(sim2[day2:day2+params.pred_interval,0]), color='g', alpha=0.1)
                    #plt.plot(days, np.log1p([sim2[day2,1], new[1]]), color='b', alpha=0.1)
                    plt.plot(days, np.log1p(sim2[day2:day2+params.pred_interval,1]), color='g', alpha=0.1)
                diff = np.abs((samp_t2 - new)/samp_t2)
                L1.append(diff)
            else:
                continue

        if needsplot and L1:
            #plt.plot(days, np.log1p([samp_t1[0], samp_t2[0]]), 'k-', label='new sympt')
            plt.plot(days, np.log1p(reals[sim,0,:]), 'k-', label='new sympt')
            plt.plot(days, np.log1p(reals[sim,1,:]), 'k--', label='cum sympt')
            plt.plot(days, np.log1p(fakes[sim,0,:]), 'r-', label='new sympt (gen)')
            plt.plot(days, np.log1p(fakes[sim,1,:]), 'r--', label='cum sympt (gen)')
            plt.ylabel('$\log (1+n)$')
            plt.xlabel('Days from prediction time')
            plt.ylim((0,12.5))
            plt.title('LLcompliance=%f, WFHcompliance=%f'%(pars[sim,0], pars[sim,1]))
            plt.legend()
            needsplot=False
        if not L1:
            continue
        L1 = np.stack(L1, axis=0)
        sympt_err = L1[:,0].mean()
        cum_err = L1[:,1].mean()
        diff = np.abs((samp_t2 - gen[sim])/samp_t2)
        errs.append(diff/L1.mean(axis=0))

    errs = np.stack(errs, axis=0)
    return f, errs


def plot_snapshots(inp, tar, gen, pars, params):
    if params.norm=='log':
        inp = np.exp(inp) - 1.
        tar = np.exp(tar) - 1.
        gen = np.exp(gen) - 1.
    inp = np.log1p(np.sum(inp, axis=(1,2)))
    tar = np.log1p(np.sum(tar, axis=(1,2)))
    gen = np.log1p(np.sum(gen, axis=(1,2)))
    f = plt.figure()
    plt.plot([0,1], [inp[0], tar[0]], 'k-', label='true newsympt')
    plt.plot([0,1], [inp[1], tar[1]], 'k--', label='true cumsympt')
    plt.plot([0,1], [inp[0], gen[0]], 'r-', label='gen newsympt')
    plt.plot([0,1], [inp[1], gen[1]], 'r--', label='gen cumsympt')
    plt.legend()
    plt.ylabel(r'$\log(1+n)$')
    plt.xlabel('Time since snapshot (prediciton interval)')
    plt.ylim((0,12.5))
    plt.title('LLcompliance=%f, WFHcompliance=%f'%(pars[0],pars[1]))
    return f
