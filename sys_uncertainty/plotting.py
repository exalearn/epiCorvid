import numpy as np
import matplotlib.pyplot as plt

def plot_timeseries(agg, figsz, agebins):
    # ignore total number of indivuals fields
    filter1 ='Total individuals by age'
    filter2 ='Total unvaccinated individuals by age'
    fieldnames = sorted(list(filter(lambda x: x!=filter1  and x!=filter2, agg.keys())))

    plt.figure(figsize=(figsz,figsz))
    idx = 0
    for ix in range(3):
        for iy in range(3):
            key = fieldnames[idx]
            v = agg[key]
            dim = v.shape[-1]
            Ndays=v.shape[1]
            d=np.float if 'rate' in key else np.int
            
            plt.subplot(3,3,idx+1)
            if dim==5:
                for i in range(5):
                    means = np.mean(v[:,:,i], axis=0).astype(d)
                    stds = np.std(v[:,:,i], axis=0).astype(d)
                    plt.plot(means)
                    plt.fill_between(np.arange(Ndays), means + stds, means-stds, alpha=0.15, label=agebins[i+1])
                    plt.legend()
            else:
                means = np.mean(v[:,:,0], axis=0).astype(d)
                stds = np.std(v[:,:,0], axis=0).astype(d)
                plt.plot(means)
                plt.fill_between(np.arange(Ndays), means + stds, means-stds, alpha=0.15)
            plt.yscale('log')
            plt.title(key)
            plt.xlabel('Day')
            idx +=1
    plt.tight_layout(pad=2.0)
    plt.show() 


def plot_end(agg, figsz, agebins):
    # ignore total number of indivuals fields
    filter1 ='Total individuals by age'
    filter2 ='Total unvaccinated individuals by age'
    fieldnames = sorted(list(filter(lambda x: x!=filter1  and x!=filter2, agg.keys())))

    plt.figure(figsize=(figsz,figsz))
    idx = 0
    for ix in range(3):
        for iy in range(3):
            key = fieldnames[idx]
            v = agg[key]
            dim = v.shape[-1]
            Ndays=v.shape[1]
            d=np.float if 'rate' in key else np.int

            plt.subplot(3,3,idx+1)
            if dim==5:
                plt.boxplot(v[:,-1,:].astype(d))
                plt.title(key)
                plt.gca().set_xticklabels(agebins)
                plt.xlabel('Age bin')
                if 'rate' not in key: plt.yscale('log') # log scale for individual counts
            else:
                means = np.mean(v[:,:,0], axis=0).astype(d)
                stds = np.std(v[:,:,0], axis=0).astype(d)
                plt.plot(means)
                plt.fill_between(np.arange(Ndays), means + stds, means-stds, alpha=0.15)
                plt.yscale('log')
            plt.title(key)
            idx +=1
    plt.tight_layout(pad=2.0)
    plt.show()



def plot_daily_new_symptomatic(agg, plot_peakshift=True):
    ts = agg['Cumulative symptomatic (daily)'][:,:,0]
    daily = np.diff(ts, axis=1)
    means = np.mean(daily, axis=0).astype(np.int)
    median = np.median(daily, axis=0).astype(np.int)
    stds = np.std(daily, axis=0).astype(np.int)
    days = np.arange(ts.shape[1])[1:]
    

    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1], 'hspace':0}, figsize=(8,6))
    a0.plot(days, means, '-')
    a0.plot(days, median, '-')
    for i in range(daily.shape[0]):
        a0.plot(days, daily[i,:], 'k-', alpha=0.05)
    a0.fill_between(days, means+stds, means-stds, alpha=0.3)
    a0.set_title('Daily new symptomatic individuals')
    a0.set_ylabel('Daily new symptomatic individuals')
    a0.set_yscale('log')
    a0.tick_params(axis='x', direction='in', labelbottom=False, which='both')
    
    
    for i in range(daily.shape[0]):
        normed = (daily[i,:].astype(np.float) - means.astype(np.float))/means.astype(np.float)
        a1.plot(days, normed, 'k-', alpha=0.05)
    normed = stds.astype(np.float)/means.astype(np.float)
    a1.fill_between(days, normed, -normed, alpha=0.3)
    a1.plot([days[0], days[-1]], [0., 0.])
    normed = (median.astype(np.float) - means.astype(np.float))/means.astype(np.float)
    a1.plot(days, normed, '-')
    a1.set_xlabel('Day')
    a1.set_ylabel('Relative error')
    a1.set_ylim(-1,1)
    a1.tick_params(axis='x', top=True, direction='inout', labeltop=False, which='both')
    plt.show()

    if plot_peakshift:
        f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1], 'hspace':0}, figsize=(8,6))
        meanpeak=np.argmax(means)

        reldays = days - meanpeak
        sumshifted = np.zeros((2*daily.shape[1]))
        numshifted = np.zeros((2*daily.shape[1]))
        aggshifted = np.zeros((daily.shape[0], 2*daily.shape[1]))
        Ndays = daily.shape[1]
        for i in range(daily.shape[0]):
            shifted = days - np.argmax(daily[i,:])
            aggshifted[i,shifted+Ndays] = daily[i,:]
            sumshifted[shifted+Ndays]+=daily[i,:]
            numshifted[shifted+Ndays]+=np.ones((daily.shape[1]))
            a0.plot(shifted, daily[i,:], 'k-', alpha=0.04)

        rolled_days = np.arange(-means.shape[0], means.shape[0])
        stdshifted = np.std(aggshifted, axis=0)
        meanshifted = sumshifted/numshifted
        a0.plot(rolled_days, meanshifted.astype(np.int), '-')
        a0.fill_between(rolled_days, meanshifted.astype(np.int) + stdshifted.astype(int),
                                     meanshifted.astype(np.int) - stdshifted.astype(int),
                                     alpha=0.3)
        a0.set_title('Daily new symptomatic individuals (shifted)')
        a0.set_ylabel('Daily new symptomatic individuals')
        a0.set_yscale('log')
        a0.set_xlim((reldays[0], reldays[-1]))
        a0.tick_params(axis='x', direction='in', labelbottom=False, which='both')

        for i in range(daily.shape[0]):        
            shifted = days - np.argmax(daily[i,:])
            thisshifted = np.zeros((2*means.shape[0]))
            thisshifted[shifted+Ndays] = daily[i,:]
            normed = (thisshifted - meanshifted)/meanshifted
            a1.plot(rolled_days, normed, 'k-', alpha=0.04)
        a1.plot([rolled_days[0], rolled_days[-1]], [0., 0.])

        relsigma = stdshifted/meanshifted
        a1.fill_between(rolled_days, relsigma, -relsigma, alpha=0.3)
        a1.set_xlabel('Day relative to run\'s peak')
        a1.set_ylabel('Relative error')
        a1.set_ylim(-1,1)
        a1.set_xlim((reldays[0], reldays[-1]))
        a1.tick_params(axis='x', top=True, direction='inout', labeltop=False, which='both')
        plt.show()


def plot_peakdist(agg):
    peaks = []
    ts = agg['Cumulative symptomatic (daily)'][:,:,0]
    daily = np.diff(ts, axis=1)
    for i in range(daily.shape[0]):
        peaks.append(np.argmax(daily[i,:]))
    plt.figure()
    plt.hist(peaks, bins=50)
    plt.yscale('log')
    plt.xlabel('Day of peak', fontsize=18)
    plt.ylabel('Counts', fontsize=18)
    plt.show()
